"""
Long text TTS job management system
"""

import asyncio
import hashlib
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from app.config import Config
from app.core.voice_library import get_voice_library
from app.models.long_text import (
    LongTextJobStatus,
    LongTextJobMetadata,
    LongTextChunk,
    LongTextProgress,
    LongTextJobResponse,
    LongTextJobListItem,
    LongTextJobList,
    LongTextSSEEvent
)

logger = logging.getLogger(__name__)


class LongTextJobManager:
    """Manages long text TTS jobs with filesystem persistence"""

    def __init__(self):
        self.data_dir = Path(Config.LONG_TEXT_DATA_DIR)
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.active_jobs_lock = threading.Lock()
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(Config.LONG_TEXT_MAX_CONCURRENT_JOBS)
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory structure exists"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Long text data directory: {self.data_dir}")

    def _get_job_directory(self, job_id: str) -> Path:
        """Get the directory path for a specific job"""
        return self.data_dir / job_id

    def _get_job_file_paths(self, job_id: str) -> Dict[str, Path]:
        """Get all file paths for a job"""
        job_dir = self._get_job_directory(job_id)
        return {
            'metadata': job_dir / 'metadata.json',
            'input_text': job_dir / 'input_text.txt',
            'chunks': job_dir / 'chunks.json',
            'progress': job_dir / 'progress.json',
            'chunks_dir': job_dir / 'chunks',
            'output_dir': job_dir / 'output'
        }

    def _generate_text_hash(self, text: str) -> str:
        """Generate SHA256 hash of input text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _create_job_directories(self, job_id: str):
        """Create directory structure for a new job"""
        paths = self._get_job_file_paths(job_id)
        job_dir = self._get_job_directory(job_id)

        job_dir.mkdir(parents=True, exist_ok=True)
        paths['chunks_dir'].mkdir(exist_ok=True)
        paths['output_dir'].mkdir(exist_ok=True)

    def _save_job_metadata(self, metadata: LongTextJobMetadata):
        """Save job metadata to filesystem"""
        paths = self._get_job_file_paths(metadata.job_id)

        # Update timestamp
        metadata.updated_at = datetime.utcnow()

        with open(paths['metadata'], 'w') as f:
            json.dump(metadata.dict(), f, indent=2, default=str)

    def _load_job_metadata(self, job_id: str) -> Optional[LongTextJobMetadata]:
        """Load job metadata from filesystem"""
        paths = self._get_job_file_paths(job_id)

        if not paths['metadata'].exists():
            return None

        try:
            with open(paths['metadata'], 'r') as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            for field in ['created_at', 'updated_at', 'processing_started_at',
                         'processing_paused_at', 'processing_completed_at']:
                if data.get(field):
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))

            return LongTextJobMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load metadata for job {job_id}: {e}")
            return None

    def _save_chunks_data(self, job_id: str, chunks: List[LongTextChunk]):
        """Save chunks data to filesystem"""
        paths = self._get_job_file_paths(job_id)

        chunks_data = [chunk.dict() for chunk in chunks]
        with open(paths['chunks'], 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)

    def _load_chunks_data(self, job_id: str) -> List[LongTextChunk]:
        """Load chunks data from filesystem"""
        paths = self._get_job_file_paths(job_id)

        if not paths['chunks'].exists():
            return []

        try:
            with open(paths['chunks'], 'r') as f:
                data = json.load(f)

            chunks = []
            for chunk_data in data:
                # Convert datetime strings back to datetime objects
                for field in ['processing_started_at', 'processing_completed_at']:
                    if chunk_data.get(field):
                        chunk_data[field] = datetime.fromisoformat(chunk_data[field].replace('Z', '+00:00'))
                chunks.append(LongTextChunk(**chunk_data))

            return chunks
        except Exception as e:
            logger.error(f"Failed to load chunks data for job {job_id}: {e}")
            return []

    def _save_input_text(self, job_id: str, text: str):
        """Save input text to filesystem"""
        paths = self._get_job_file_paths(job_id)

        with open(paths['input_text'], 'w', encoding='utf-8') as f:
            f.write(text)

    def _load_input_text(self, job_id: str) -> Optional[str]:
        """Load input text from filesystem"""
        paths = self._get_job_file_paths(job_id)

        if not paths['input_text'].exists():
            return None

        try:
            with open(paths['input_text'], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load input text for job {job_id}: {e}")
            return None

    def create_job(
        self,
        text: str,
        voice: Optional[str] = None,
        output_format: str = "mp3",
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        session_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        silence_padding: Optional[int] = None,
        quality_preset: Optional[str] = None,
        enable_pauses: Optional[bool] = None,
        custom_pauses: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, int]:
        """
        Create a new long text job

        Returns:
            Tuple of (job_id, estimated_chunks)
        """
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Calculate text hash for potential deduplication
        text_hash = self._generate_text_hash(text)

        # Resolve chunking configuration
        resolved_chunk_size = chunk_size or Config.LONG_TEXT_CHUNK_SIZE
        if resolved_chunk_size <= 0:
            resolved_chunk_size = Config.LONG_TEXT_CHUNK_SIZE

        resolved_chunking_strategy = chunking_strategy or Config.LONG_TEXT_CHUNKING_STRATEGY
        resolved_silence_padding = (
            silence_padding
            if silence_padding is not None and silence_padding >= 0
            else Config.LONG_TEXT_SILENCE_PADDING_MS
        )
        resolved_quality_preset = quality_preset or Config.LONG_TEXT_QUALITY_PRESET

        resolved_enable_pauses = (
            Config.ENABLE_PUNCTUATION_PAUSES if enable_pauses is None else bool(enable_pauses)
        )

        resolved_custom_pauses = None
        if custom_pauses:
            resolved_custom_pauses = {}
            for key, value in custom_pauses.items():
                try:
                    resolved_custom_pauses[str(key)] = int(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring invalid custom pause value %r=%r", key, value)
            if not resolved_custom_pauses:
                resolved_custom_pauses = None

        # Estimate number of chunks
        estimated_chunks = max(1, (len(text) + resolved_chunk_size - 1) // resolved_chunk_size)

        # Create job directories
        self._create_job_directories(job_id)

        # Resolve voice name for storage (use default if no voice specified)
        resolved_voice_name = voice
        if not voice:
            # Get default voice name from voice library
            voice_lib = get_voice_library()
            default_voice = voice_lib.get_default_voice()
            resolved_voice_name = default_voice or "Default"

        # Create metadata
        metadata = LongTextJobMetadata(
            job_id=job_id,
            text_length=len(text),
            text_hash=text_hash,
            total_chunks=estimated_chunks,
            voice=resolved_voice_name,
            parameters={
                'exaggeration': exaggeration,
                'cfg_weight': cfg_weight,
                'temperature': temperature,
                'output_format': output_format,
                'chunking_strategy': resolved_chunking_strategy,
                'chunk_size': resolved_chunk_size,
                'quality_preset': resolved_quality_preset,
                'silence_padding_ms': resolved_silence_padding,
                'pause_settings': {
                    'enable': resolved_enable_pauses,
                    'custom': resolved_custom_pauses,
                },
            },
            output_format=output_format,
            user_session_id=session_id
        )

        # Save to filesystem
        self._save_job_metadata(metadata)
        self._save_input_text(job_id, text)

        logger.info(f"Created job {job_id} for {len(text)} characters ({estimated_chunks} chunks)")
        return job_id, estimated_chunks

    def get_job_status(self, job_id: str) -> Optional[LongTextJobResponse]:
        """Get current status and progress of a job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return None

        chunks = self._load_chunks_data(job_id)

        # Calculate progress
        progress = self._calculate_progress(metadata, chunks)

        # Determine available actions
        can_pause = metadata.status == LongTextJobStatus.PROCESSING
        can_resume = metadata.status == LongTextJobStatus.PAUSED
        can_cancel = metadata.status in [LongTextJobStatus.PENDING,
                                       LongTextJobStatus.PROCESSING,
                                       LongTextJobStatus.PAUSED]

        # Generate download URL if completed
        download_url = None
        if metadata.status == LongTextJobStatus.COMPLETED and metadata.output_path:
            download_url = f"/v1/audio/speech/long/{job_id}/download"

        return LongTextJobResponse(
            job_id=job_id,
            status=metadata.status,
            progress=progress,
            metadata=metadata,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            download_url=download_url,
            can_pause=can_pause,
            can_resume=can_resume,
            can_cancel=can_cancel
        )

    def _calculate_progress(self, metadata: LongTextJobMetadata, chunks: List[LongTextChunk]) -> LongTextProgress:
        """Calculate current progress for a job"""
        completed_chunks = [chunk for chunk in chunks if chunk.audio_file is not None]
        current_chunk = None

        # Find current processing chunk
        for chunk in chunks:
            if chunk.processing_started_at and not chunk.processing_completed_at:
                current_chunk = chunk
                break

        # Calculate overall progress percentage
        if metadata.total_chunks > 0:
            overall_progress = (len(completed_chunks) / metadata.total_chunks) * 100
        else:
            overall_progress = 0.0

        # Estimate remaining time based on completed chunks
        estimated_remaining = None
        if completed_chunks and metadata.current_chunk is not None:
            avg_time_per_chunk = sum(c.duration_ms or 0 for c in completed_chunks) / len(completed_chunks)
            remaining_chunks = metadata.total_chunks - len(completed_chunks)
            estimated_remaining = int((avg_time_per_chunk * remaining_chunks) / 1000)

        return LongTextProgress(
            job_id=metadata.job_id,
            overall_progress=overall_progress,
            current_chunk=current_chunk,
            completed_chunks=completed_chunks,
            estimated_remaining_seconds=estimated_remaining,
            status=metadata.status,
            error=metadata.error
        )

    def list_jobs(self, session_id: Optional[str] = None, limit: int = 50) -> LongTextJobList:
        """List all jobs, optionally filtered by session ID"""
        jobs = []
        active_count = 0
        completed_count = 0

        # Scan all job directories
        if self.data_dir.exists():
            for job_dir in self.data_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                metadata = self._load_job_metadata(job_dir.name)
                if not metadata:
                    continue

                # Session ID filtering removed - show all jobs for better UX

                # Load input text for preview
                input_text = self._load_input_text(job_dir.name) or ""
                text_preview = input_text[:100] + ("..." if len(input_text) > 100 else "")

                # Calculate progress
                chunks = self._load_chunks_data(job_dir.name)
                progress = self._calculate_progress(metadata, chunks)

                # Generate download URL if completed
                download_url = None
                if metadata.status == LongTextJobStatus.COMPLETED:
                    download_url = f"/v1/audio/speech/long/{job_dir.name}/download"

                # Count job types
                if metadata.status in [LongTextJobStatus.PENDING, LongTextJobStatus.PROCESSING]:
                    active_count += 1
                elif metadata.status == LongTextJobStatus.COMPLETED:
                    completed_count += 1

                jobs.append(LongTextJobListItem(
                    job_id=job_dir.name,
                    status=metadata.status,
                    text_preview=text_preview,
                    text_length=metadata.text_length,
                    progress_percentage=progress.overall_progress,
                    created_at=metadata.created_at,
                    completed_at=metadata.processing_completed_at,
                    download_url=download_url,
                    can_resume=metadata.status == LongTextJobStatus.PAUSED,
                    voice=metadata.voice,
                    parameters=metadata.parameters
                ))

        # Sort by creation date (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        # Apply limit
        jobs = jobs[:limit]

        return LongTextJobList(
            jobs=jobs,
            total_jobs=len(jobs),
            active_jobs=active_count,
            completed_jobs=completed_count
        )

    def list_history_jobs(self, session_id: Optional[str] = None,
                         status_filter: Optional[LongTextJobStatus] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         search_text: Optional[str] = None,
                         is_archived: Optional[bool] = None,
                         sort_by: str = "completed_desc",
                         limit: int = 50, offset: int = 0) -> LongTextJobList:
        """List jobs for history view with advanced filtering and sorting"""
        jobs = []
        active_count = 0
        completed_count = 0

        if not self.data_dir.exists():
            return LongTextJobList(jobs=[], total_jobs=0, active_jobs=0, completed_jobs=0)

        # Collect all jobs first
        all_jobs = []
        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name in ['history']:
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata:
                continue

            # Session ID filtering removed - show all jobs for better UX

            # Status filter
            if status_filter and metadata.status != status_filter:
                continue

            # Date filters
            comparison_date = metadata.completion_timestamp or metadata.created_at
            if start_date and comparison_date and comparison_date < start_date:
                continue
            if end_date and comparison_date and comparison_date > end_date:
                continue

            # Archive filter
            if is_archived is not None and metadata.is_archived != is_archived:
                continue

            # Search filter
            if search_text:
                input_text = self._load_input_text(job_dir.name) or ""
                display_name = metadata.display_name or ""
                if (search_text.lower() not in input_text.lower() and
                    search_text.lower() not in display_name.lower()):
                    continue

            # Load input text for preview
            input_text = self._load_input_text(job_dir.name) or ""
            text_preview = input_text[:100] + ("..." if len(input_text) > 100 else "")

            # Calculate progress
            chunks = self._load_chunks_data(job_dir.name)
            progress = self._calculate_progress(metadata, chunks)

            # Generate download URL if completed
            download_url = None
            if metadata.status == LongTextJobStatus.COMPLETED:
                download_url = f"/v1/audio/speech/long/{job_dir.name}/download"

            # Count job types
            if metadata.status in [LongTextJobStatus.PENDING, LongTextJobStatus.PROCESSING]:
                active_count += 1
            elif metadata.status == LongTextJobStatus.COMPLETED:
                completed_count += 1

            job_item = LongTextJobListItem(
                job_id=job_dir.name,
                status=metadata.status,
                text_preview=text_preview,
                text_length=metadata.text_length,
                progress_percentage=progress.overall_progress,
                created_at=metadata.created_at,
                completed_at=metadata.completion_timestamp or metadata.processing_completed_at,
                download_url=download_url,
                can_resume=metadata.status == LongTextJobStatus.PAUSED,
                voice=metadata.voice,
                total_duration_seconds=metadata.total_duration_seconds,
                audio_file_size=metadata.audio_file_size,
                retry_count=metadata.retry_count,
                is_archived=metadata.is_archived,
                display_name=metadata.display_name,
                tags=metadata.tags,
                last_accessed=metadata.last_accessed,
                parameters=metadata.parameters
            )
            all_jobs.append(job_item)

        # Sort jobs
        if sort_by == "created_desc":
            all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "created_asc":
            all_jobs.sort(key=lambda x: x.created_at)
        elif sort_by == "completed_desc":
            all_jobs.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
        elif sort_by == "completed_asc":
            all_jobs.sort(key=lambda x: x.completed_at or datetime.min)
        elif sort_by == "duration_desc":
            all_jobs.sort(key=lambda x: x.total_duration_seconds or 0, reverse=True)
        elif sort_by == "duration_asc":
            all_jobs.sort(key=lambda x: x.total_duration_seconds or 0)
        elif sort_by == "name_asc":
            all_jobs.sort(key=lambda x: (x.display_name or x.text_preview).lower())
        elif sort_by == "name_desc":
            all_jobs.sort(key=lambda x: (x.display_name or x.text_preview).lower(), reverse=True)
        elif sort_by == "size_desc":
            all_jobs.sort(key=lambda x: x.audio_file_size or 0, reverse=True)
        elif sort_by == "size_asc":
            all_jobs.sort(key=lambda x: x.audio_file_size or 0)

        # Apply pagination
        total_count = len(all_jobs)
        jobs = all_jobs[offset:offset + limit]

        return LongTextJobList(
            jobs=jobs,
            total_jobs=total_count,
            active_jobs=active_count,
            completed_jobs=completed_count
        )

    def get_history_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for job history"""
        if not self.data_dir.exists():
            return {
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "total_audio_duration_seconds": 0.0,
                "total_storage_bytes": 0,
                "average_processing_time_seconds": 0.0,
                "success_rate_percentage": 0.0,
                "most_used_voice": None,
                "jobs_by_month": {}
            }

        total_jobs = 0
        completed_jobs = 0
        failed_jobs = 0
        total_audio_duration = 0.0
        total_storage_bytes = 0
        total_processing_time = 0
        voice_counts = {}
        jobs_by_month = {}

        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name in ['history']:
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata:
                continue

            # Session ID filtering removed - show all jobs for better UX

            total_jobs += 1

            if metadata.status == LongTextJobStatus.COMPLETED:
                completed_jobs += 1
                if metadata.total_duration_seconds:
                    total_audio_duration += metadata.total_duration_seconds
                if metadata.audio_file_size:
                    total_storage_bytes += metadata.audio_file_size
                if metadata.total_processing_time_ms:
                    total_processing_time += metadata.total_processing_time_ms

            elif metadata.status == LongTextJobStatus.FAILED:
                failed_jobs += 1

            # Track voice usage
            if metadata.voice:
                voice_counts[metadata.voice] = voice_counts.get(metadata.voice, 0) + 1

            # Track jobs by month
            month_key = metadata.created_at.strftime("%Y-%m")
            jobs_by_month[month_key] = jobs_by_month.get(month_key, 0) + 1

        # Calculate averages and percentages
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
        avg_processing_time = (total_processing_time / completed_jobs / 1000) if completed_jobs > 0 else 0.0
        most_used_voice = max(voice_counts, key=voice_counts.get) if voice_counts else None

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "total_audio_duration_seconds": total_audio_duration,
            "total_storage_bytes": total_storage_bytes,
            "average_processing_time_seconds": avg_processing_time,
            "success_rate_percentage": success_rate,
            "most_used_voice": most_used_voice,
            "jobs_by_month": jobs_by_month
        }

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata or metadata.status != LongTextJobStatus.PROCESSING:
            return False

        # Cancel the processing task if it exists (thread-safe)
        with self.active_jobs_lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]

        # Update metadata
        metadata.status = LongTextJobStatus.PAUSED
        metadata.processing_paused_at = datetime.utcnow()
        self._save_job_metadata(metadata)

        logger.info(f"Paused job {job_id}")
        return True

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata or metadata.status != LongTextJobStatus.PAUSED:
            return False

        # Update metadata
        metadata.status = LongTextJobStatus.PENDING
        metadata.processing_paused_at = None
        self._save_job_metadata(metadata)

        # Add back to queue for processing
        await self.job_queue.put(job_id)

        logger.info(f"Resumed job {job_id}")
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        if metadata.status in [LongTextJobStatus.COMPLETED, LongTextJobStatus.FAILED]:
            return False

        # Cancel the processing task if it exists (thread-safe)
        with self.active_jobs_lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]

        # Update metadata
        metadata.status = LongTextJobStatus.CANCELLED
        self._save_job_metadata(metadata)

        logger.info(f"Cancelled job {job_id}")
        return True

    def complete_job(self, job_id: str, output_path: str, output_size_bytes: int,
                    output_duration_seconds: float) -> bool:
        """Mark a job as completed and set up for history persistence"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        # Update completion fields
        metadata.status = LongTextJobStatus.COMPLETED
        metadata.processing_completed_at = datetime.utcnow()
        metadata.completion_timestamp = metadata.processing_completed_at
        metadata.output_path = output_path
        metadata.output_size_bytes = output_size_bytes
        metadata.output_duration_seconds = output_duration_seconds
        metadata.total_duration_seconds = output_duration_seconds

        # Set up persistent storage for history
        persistent_path = self._setup_persistent_storage(job_id, output_path)
        if persistent_path:
            metadata.audio_file_path = persistent_path
            metadata.audio_file_size = output_size_bytes

        # Generate display name if not set
        if not metadata.display_name:
            # Load input text for preview
            input_text = self._load_input_text(job_id) or ""
            preview = input_text[:50].strip()
            if len(input_text) > 50:
                preview += "..."
            metadata.display_name = f"Long Text: {preview}"

        # Update processing time
        if metadata.processing_started_at:
            metadata.total_processing_time_ms = int(
                (metadata.processing_completed_at - metadata.processing_started_at).total_seconds() * 1000
            )

        self._save_job_metadata(metadata)
        logger.info(f"Completed job {job_id} - Duration: {output_duration_seconds:.1f}s, Size: {output_size_bytes:,} bytes")
        return True

    def _setup_persistent_storage(self, job_id: str, output_path: str) -> Optional[str]:
        """Set up persistent storage for completed job audio"""
        try:
            paths = self._get_job_file_paths(job_id)
            source_file = paths['output_dir'] / Path(output_path).name

            if not source_file.exists():
                logger.warning(f"Output file not found for job {job_id}: {source_file}")
                return None

            # Create persistent storage directory
            persistent_dir = self.data_dir / "history" / job_id
            persistent_dir.mkdir(parents=True, exist_ok=True)

            # Copy file to persistent location
            persistent_file = persistent_dir / source_file.name
            shutil.copy2(source_file, persistent_file)

            return str(persistent_file.relative_to(self.data_dir))

        except Exception as e:
            logger.error(f"Failed to set up persistent storage for job {job_id}: {e}")
            return None

    def archive_job(self, job_id: str) -> bool:
        """Archive a job (mark as archived without deleting)"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        metadata.is_archived = True
        metadata.last_accessed = datetime.utcnow()
        self._save_job_metadata(metadata)

        logger.info(f"Archived job {job_id}")
        return True

    def unarchive_job(self, job_id: str) -> bool:
        """Unarchive a job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        metadata.is_archived = False
        metadata.last_accessed = datetime.utcnow()
        self._save_job_metadata(metadata)

        logger.info(f"Unarchived job {job_id}")
        return True

    def update_job_metadata(self, job_id: str, display_name: Optional[str] = None,
                           tags: Optional[List[str]] = None, is_archived: Optional[bool] = None) -> bool:
        """Update job metadata fields"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        if display_name is not None:
            metadata.display_name = display_name
        if tags is not None:
            metadata.tags = tags
        if is_archived is not None:
            metadata.is_archived = is_archived

        metadata.last_accessed = datetime.utcnow()
        self._save_job_metadata(metadata)

        return True

    def track_job_access(self, job_id: str) -> bool:
        """Track when a job was last accessed"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        metadata.last_accessed = datetime.utcnow()
        self._save_job_metadata(metadata)
        return True

    def retry_job(self, job_id: str, preserve_chunks: bool = True,
                  new_parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Retry a failed job, optionally with new parameters"""
        original_metadata = self._load_job_metadata(job_id)
        if not original_metadata:
            return None

        if original_metadata.status not in [LongTextJobStatus.FAILED, LongTextJobStatus.CANCELLED]:
            logger.warning(f"Cannot retry job {job_id} with status {original_metadata.status}")
            return None

        # Load original input text
        input_text = self._load_input_text(job_id)
        if not input_text:
            logger.error(f"Cannot retry job {job_id}: input text not found")
            return None

        # Create new job with original parameters
        parameters = original_metadata.parameters.copy()
        if new_parameters:
            parameters.update(new_parameters)

        new_job_id, _ = self.create_job(
            text=input_text,
            voice=original_metadata.voice,
            output_format=original_metadata.output_format,
            exaggeration=parameters.get('exaggeration'),
            cfg_weight=parameters.get('cfg_weight'),
            temperature=parameters.get('temperature'),
            session_id=original_metadata.user_session_id,
            chunking_strategy=parameters.get('chunking_strategy'),
            chunk_size=parameters.get('chunk_size'),
            silence_padding=parameters.get('silence_padding_ms'),
            quality_preset=parameters.get('quality_preset'),
            enable_pauses=(parameters.get('pause_settings') or {}).get('enable'),
            custom_pauses=(parameters.get('pause_settings') or {}).get('custom'),
        )

        # Update metadata to link to original job
        new_metadata = self._load_job_metadata(new_job_id)
        if new_metadata:
            new_metadata.original_job_id = job_id
            new_metadata.retry_count = original_metadata.retry_count + 1
            self._save_job_metadata(new_metadata)

        # If preserving chunks, copy successful ones
        if preserve_chunks:
            try:
                original_chunks = self._load_chunks_data(job_id)
                successful_chunks = [chunk for chunk in original_chunks if chunk.audio_file and not chunk.error]

                if successful_chunks:
                    # Copy successful chunk files to new job
                    new_paths = self._get_job_file_paths(new_job_id)
                    original_paths = self._get_job_file_paths(job_id)

                    for chunk in successful_chunks:
                        if chunk.audio_file:
                            original_file = original_paths['chunks_dir'] / chunk.audio_file
                            if original_file.exists():
                                new_file = new_paths['chunks_dir'] / chunk.audio_file
                                shutil.copy2(original_file, new_file)

                    logger.info(f"Copied {len(successful_chunks)} successful chunks to retry job {new_job_id}")

            except Exception as e:
                logger.warning(f"Failed to preserve chunks for retry job {new_job_id}: {e}")

        logger.info(f"Created retry job {new_job_id} for original job {job_id}")
        return new_job_id

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and all its files"""
        job_dir = self._get_job_directory(job_id)

        if not job_dir.exists():
            return False

        # Cancel if still running
        self.cancel_job(job_id)

        # Remove all files
        try:
            shutil.rmtree(job_dir)
            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def cleanup_old_jobs(self, retention_days: Optional[int] = None, max_storage_bytes: Optional[int] = None):
        """Clean up old jobs based on retention policy and storage limits"""
        if not self.data_dir.exists():
            return

        retention_days = retention_days or Config.LONG_TEXT_JOB_RETENTION_DAYS
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        deleted_count = 0
        freed_bytes = 0

        # First pass: Delete jobs past retention period
        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name == 'history':
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata:
                continue

            # Delete based on retention policy
            should_delete = False
            comparison_date = metadata.completion_timestamp or metadata.created_at

            if metadata.status == LongTextJobStatus.COMPLETED:
                # Keep completed jobs longer if they're not archived
                if metadata.is_archived and comparison_date < cutoff_date:
                    should_delete = True
            elif metadata.status in [LongTextJobStatus.FAILED, LongTextJobStatus.CANCELLED]:
                # Delete failed/cancelled jobs sooner
                failed_cutoff = datetime.utcnow() - timedelta(days=max(7, retention_days // 4))
                if comparison_date < failed_cutoff:
                    should_delete = True

            if should_delete:
                job_size = self._calculate_job_size(job_dir.name)
                if self.delete_job(job_dir.name):
                    deleted_count += 1
                    freed_bytes += job_size

        # Second pass: If storage limit exceeded, delete oldest completed jobs
        if max_storage_bytes:
            current_storage = self._calculate_total_storage()
            if current_storage > max_storage_bytes:
                excess_bytes = current_storage - max_storage_bytes
                oldest_jobs = self._get_oldest_jobs_by_storage()

                for job_id, job_size in oldest_jobs:
                    if excess_bytes <= 0:
                        break

                    metadata = self._load_job_metadata(job_id)
                    if metadata and metadata.status == LongTextJobStatus.COMPLETED:
                        if self.delete_job(job_id):
                            deleted_count += 1
                            freed_bytes += job_size
                            excess_bytes -= job_size

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old jobs, freed {freed_bytes:,} bytes")

    def _calculate_job_size(self, job_id: str) -> int:
        """Calculate total size of job files"""
        total_size = 0
        job_dir = self._get_job_directory(job_id)

        if not job_dir.exists():
            return 0

        for file_path in job_dir.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    continue

        return total_size

    def _calculate_total_storage(self) -> int:
        """Calculate total storage used by all jobs"""
        total_size = 0

        if not self.data_dir.exists():
            return 0

        for job_dir in self.data_dir.iterdir():
            if job_dir.is_dir():
                total_size += self._calculate_job_size(job_dir.name)

        return total_size

    def _get_oldest_jobs_by_storage(self) -> List[Tuple[str, int]]:
        """Get jobs sorted by age (oldest first) with their storage sizes"""
        jobs_with_size = []

        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name == 'history':
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata or metadata.status != LongTextJobStatus.COMPLETED:
                continue

            job_size = self._calculate_job_size(job_dir.name)
            comparison_date = metadata.completion_timestamp or metadata.created_at
            jobs_with_size.append((job_dir.name, job_size, comparison_date))

        # Sort by date (oldest first)
        jobs_with_size.sort(key=lambda x: x[2])
        return [(job_id, size) for job_id, size, _ in jobs_with_size]

    def cleanup_orphaned_files(self):
        """Clean up orphaned files that don't belong to valid jobs"""
        if not self.data_dir.exists():
            return

        cleaned_count = 0

        for item in self.data_dir.iterdir():
            if item.is_file():
                # Remove any loose files in the data directory
                try:
                    item.unlink()
                    cleaned_count += 1
                except OSError:
                    continue
            elif item.is_dir() and item.name != 'history':
                # Check if directory has valid metadata
                metadata = self._load_job_metadata(item.name)
                if not metadata:
                    # Remove directory with invalid/missing metadata
                    try:
                        shutil.rmtree(item)
                        cleaned_count += 1
                    except OSError:
                        continue

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} orphaned files/directories")

    def auto_archive_old_completed_jobs(self, archive_days: int = 30):
        """Automatically archive old completed jobs"""
        if not self.data_dir.exists():
            return

        archive_cutoff = datetime.utcnow() - timedelta(days=archive_days)
        archived_count = 0

        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name == 'history':
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata:
                continue

            # Auto-archive old completed jobs that aren't already archived
            if (metadata.status == LongTextJobStatus.COMPLETED and
                not metadata.is_archived and
                (metadata.completion_timestamp or metadata.created_at) < archive_cutoff):

                if self.archive_job(job_dir.name):
                    archived_count += 1

        if archived_count > 0:
            logger.info(f"Auto-archived {archived_count} old completed jobs")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        if not self.data_dir.exists():
            return {
                "total_storage_bytes": 0,
                "job_count": 0,
                "avg_job_size_bytes": 0,
                "completed_jobs_storage": 0,
                "failed_jobs_storage": 0,
                "active_jobs_storage": 0
            }

        total_storage = 0
        job_count = 0
        completed_storage = 0
        failed_storage = 0
        active_storage = 0

        for job_dir in self.data_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name == 'history':
                continue

            metadata = self._load_job_metadata(job_dir.name)
            if not metadata:
                continue

            job_size = self._calculate_job_size(job_dir.name)
            total_storage += job_size
            job_count += 1

            if metadata.status == LongTextJobStatus.COMPLETED:
                completed_storage += job_size
            elif metadata.status == LongTextJobStatus.FAILED:
                failed_storage += job_size
            elif metadata.status in [LongTextJobStatus.PENDING, LongTextJobStatus.PROCESSING]:
                active_storage += job_size

        return {
            "total_storage_bytes": total_storage,
            "job_count": job_count,
            "avg_job_size_bytes": total_storage // job_count if job_count > 0 else 0,
            "completed_jobs_storage": completed_storage,
            "failed_jobs_storage": failed_storage,
            "active_jobs_storage": active_storage
        }

    def get_job_file_path(self, job_id: str, file_type: str = 'output') -> Optional[Path]:
        """Get path to a specific job file"""
        paths = self._get_job_file_paths(job_id)

        if file_type == 'output':
            metadata = self._load_job_metadata(job_id)
            if metadata and metadata.output_path:
                return self._get_job_directory(job_id) / metadata.output_path
        elif file_type in paths:
            return paths[file_type]

        return None

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return False

        return True

    def get_progress(self, job_id: str) -> Optional[LongTextProgress]:
        """Get current progress for a job"""
        metadata = self._load_job_metadata(job_id)
        if not metadata:
            return None

        chunks = self._load_chunks_data(job_id)
        return self._calculate_progress(metadata, chunks)

# Global job manager instance
_job_manager: Optional[LongTextJobManager] = None


def get_job_manager() -> LongTextJobManager:
    """Get the global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = LongTextJobManager()
    return _job_manager