"""
Background task processing for long text TTS jobs
"""

import asyncio
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from app.config import Config
from app.core.long_text_jobs import get_job_manager
from app.core.text_processing import split_text_for_long_generation
from app.core.audio_processing import concatenate_audio_files, AudioConcatenationError
from app.api.endpoints.speech import generate_speech_internal, resolve_voice_path_and_language
from app.models.long_text import (
    LongTextJobStatus,
    LongTextJobMetadata,
    LongTextChunk
)

logger = logging.getLogger(__name__)


class LongTextProcessor:
    """Processes long text TTS jobs in the background"""

    def __init__(self):
        self.job_manager = get_job_manager()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background processor"""
        if self.is_running:
            return

        self.is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Long text processor started")

    async def stop(self):
        """Stop the background processor"""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel all active tasks
        for job_id, task in list(self.active_tasks.items()):
            logger.info(f"Cancelling active job: {job_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel the worker task
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self.active_tasks.clear()
        logger.info("Long text processor stopped")

    async def submit_job(self, job_id: str):
        """Submit a job for background processing"""
        if not self.is_running:
            raise RuntimeError("Processor is not running")

        await self.job_manager.job_queue.put(job_id)
        logger.info(f"Job {job_id} submitted for processing")

    async def _worker_loop(self):
        """Main worker loop that processes jobs from the queue"""
        logger.info("Background worker loop started")

        while self.is_running:
            try:
                # Wait for a job (with timeout to allow graceful shutdown)
                try:
                    job_id = await asyncio.wait_for(
                        self.job_manager.job_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Check if we have capacity to process more jobs
                if len(self.active_tasks) >= Config.LONG_TEXT_MAX_CONCURRENT_JOBS:
                    # Re-queue the job for later
                    await self.job_manager.job_queue.put(job_id)
                    await asyncio.sleep(1)  # Brief delay before checking again
                    continue

                # Start processing the job
                task = asyncio.create_task(self._process_job(job_id))
                self.active_tasks[job_id] = task

                # Set up cleanup when task completes
                task.add_done_callback(lambda t, jid=job_id: self._cleanup_task(jid))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)

        logger.info("Background worker loop stopped")

    def _cleanup_task(self, job_id: str):
        """Clean up completed task"""
        if job_id in self.active_tasks:
            del self.active_tasks[job_id]

    async def _process_job(self, job_id: str):
        """Process a single long text job with batched GPU inference"""
        logger.info(f"Starting processing for job {job_id}")

        try:
            # Load job metadata
            metadata = self.job_manager._load_job_metadata(job_id)
            if not metadata:
                logger.error(f"Job {job_id} metadata not found")
                return

            # Update status to processing
            metadata.status = LongTextJobStatus.PROCESSING
            metadata.processing_started_at = datetime.utcnow()
            self.job_manager._save_job_metadata(metadata)

            # Load input text
            input_text = self.job_manager._load_input_text(job_id)
            if not input_text:
                await self._fail_job(job_id, "Input text not found")
                return

            parameters = metadata.parameters or {}
            chunk_size = int(parameters.get('chunk_size', Config.LONG_TEXT_CHUNK_SIZE))
            if chunk_size <= 0:
                chunk_size = Config.LONG_TEXT_CHUNK_SIZE

            chunking_strategy = parameters.get(
                'chunking_strategy', Config.LONG_TEXT_CHUNKING_STRATEGY
            )

            # Phase 1: Text chunking
            await self._update_job_status(job_id, LongTextJobStatus.CHUNKING, "Splitting text into chunks")

            chunks = split_text_for_long_generation(
                input_text,
                max_chunk_size=chunk_size,
                strategy=chunking_strategy
            )

            if not chunks:
                await self._fail_job(job_id, "Failed to split text into chunks")
                return

            # Update metadata with actual chunk count
            metadata.total_chunks = len(chunks)
            self.job_manager._save_job_metadata(metadata)
            self.job_manager._save_chunks_data(job_id, chunks)

            logger.info(f"Job {job_id}: Split into {len(chunks)} chunks")

            # Phase 2: Generate audio for all chunks with batching
            await self._update_job_status(
                job_id,
                LongTextJobStatus.PROCESSING,
                f"Generating audio for {len(chunks)} chunks",
            )

            voice_path, language_id = resolve_voice_path_and_language(metadata.voice)

            batch_size = int(parameters.get('batch_size', Config.LONG_TEXT_BATCH_SIZE))
            if batch_size <= 0:
                batch_size = Config.LONG_TEXT_BATCH_SIZE

            chunk_audio_data: List[Tuple[int, Any, LongTextChunk]] = []

            for batch_start in range(0, len(chunks), batch_size):
                current_metadata = self.job_manager._load_job_metadata(job_id)
                if current_metadata and current_metadata.status in [LongTextJobStatus.PAUSED, LongTextJobStatus.CANCELLED]:
                    logger.info(f"Job {job_id} was paused/cancelled, stopping processing")
                    return

                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                logger.info(
                    f"Job {job_id}: Processing batch {batch_start // batch_size + 1} "
                    f"(chunks {batch_start + 1}-{batch_end}/{len(chunks)})"
                )

                batch_tasks = []
                for i, chunk in enumerate(batch_chunks, start=batch_start):
                    batch_tasks.append(
                        self._generate_chunk_audio(
                            job_id=job_id,
                            chunk=chunk,
                            chunk_index=i,
                            voice_path=voice_path,
                            language_id=language_id,
                            parameters=parameters,
                        )
                    )

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                current_metadata = self.job_manager._load_job_metadata(job_id)
                if not current_metadata:
                    current_metadata = metadata

                for i, result in enumerate(batch_results, start=batch_start):
                    chunk = chunks[i]
                    if isinstance(result, Exception):
                        logger.error(f"Job {job_id}: Failed to process chunk {i + 1}: {result}")
                        chunk.error = str(result)
                        chunks[i] = chunk
                        if i not in current_metadata.failed_chunks:
                            current_metadata.failed_chunks.append(i)
                    else:
                        audio_buffer, updated_chunk = result
                        chunk_audio_data.append((i, audio_buffer, updated_chunk))
                        chunks[i] = updated_chunk

                completed_chunks = len([c for c in chunks if c.audio_file])
                current_metadata.completed_chunks = completed_chunks
                current_metadata.current_chunk = min(batch_end, len(chunks)) - 1
                self.job_manager._save_job_metadata(current_metadata)
                self.job_manager._save_chunks_data(job_id, chunks)

            if not chunk_audio_data:
                await self._fail_job(job_id, "No chunks were successfully generated")
                return

            if len(chunk_audio_data) < len(chunks):
                logger.warning(
                    f"Job {job_id}: Only {len(chunk_audio_data)}/{len(chunks)} chunks generated successfully"
                )

            logger.info(f"Job {job_id}: Writing {len(chunk_audio_data)} audio files to disk")
            chunk_audio_files = await self._batch_write_audio_files(job_id, chunk_audio_data)

            successful_chunks = [path for path in chunk_audio_files if path.exists()]
            if not successful_chunks:
                await self._fail_job(job_id, "No chunks were successfully generated")
                return

            # Phase 3: Concatenate audio chunks
            await self._update_job_status(job_id, LongTextJobStatus.PROCESSING, "Combining audio chunks")

            try:
                output_filename = f"final.{metadata.output_format}"
                output_path = self.job_manager._get_job_file_paths(job_id)['output_dir'] / output_filename

                silence_padding_ms = parameters.get(
                    'silence_padding_ms', Config.LONG_TEXT_SILENCE_PADDING_MS
                )
                if silence_padding_ms is None or silence_padding_ms < 0:
                    silence_padding_ms = Config.LONG_TEXT_SILENCE_PADDING_MS

                concatenation_metadata = concatenate_audio_files(
                    audio_files=successful_chunks,
                    output_path=output_path,
                    output_format=metadata.output_format,
                    silence_duration_ms=silence_padding_ms,
                    # normalize_volume=True,
                    normalize_volume=False,
                    remove_source_files=False  # Keep source chunks for debugging
                )

                # Mark job as completed with history persistence
                self.job_manager.complete_job(
                    job_id=job_id,
                    output_path=f"output/{output_filename}",
                    output_size_bytes=concatenation_metadata['file_size_bytes'],
                    output_duration_seconds=concatenation_metadata['duration_seconds']
                )

                logger.info(f"Job {job_id} completed successfully: {concatenation_metadata['duration_seconds']:.1f}s audio, "
                          f"{concatenation_metadata['file_size_bytes']:,} bytes")

            except AudioConcatenationError as e:
                await self._fail_job(job_id, f"Audio concatenation failed: {e}")
                return
            except Exception as e:
                await self._fail_job(job_id, f"Unexpected error during concatenation: {e}")
                return

        except asyncio.CancelledError:
            # Job was cancelled
            logger.info(f"Job {job_id} processing was cancelled")
            await self._update_job_status(job_id, LongTextJobStatus.CANCELLED, "Processing was cancelled")
            raise

        except Exception as e:
            logger.error(f"Unexpected error processing job {job_id}: {e}")
            logger.error(traceback.format_exc())
            await self._fail_job(job_id, f"Unexpected error: {e}")

    async def _generate_chunk_audio(
        self,
        job_id: str,
        chunk: LongTextChunk,
        chunk_index: int,
        voice_path: str,
        language_id: str,
        parameters: Dict[str, Any],
    ):
        """Generate audio for a single chunk (executed in parallel within a batch)"""

        chunk.processing_started_at = datetime.utcnow()
        chunk.error = None

        logger.debug(
            f"Job {job_id}: Processing chunk {chunk_index + 1} ({len(chunk.text)} chars)"
        )

        try:
            pause_settings = (parameters.get('pause_settings') or {}) if parameters else {}

            audio_buffer = await generate_speech_internal(
                text=chunk.text,
                voice_sample_path=voice_path,
                language_id=language_id,
                exaggeration=(parameters or {}).get('exaggeration'),
                cfg_weight=(parameters or {}).get('cfg_weight'),
                temperature=(parameters or {}).get('temperature'),
                enable_pauses=pause_settings.get('enable'),
                custom_pauses=pause_settings.get('custom'),
            )

            chunk.audio_file = f"chunk_{chunk_index + 1:03d}.wav"
            chunk.processing_completed_at = datetime.utcnow()
            chunk.duration_ms = int(
                (chunk.processing_completed_at - chunk.processing_started_at).total_seconds() * 1000
            )

            logger.debug(
                f"Job {job_id}: Completed chunk {chunk_index + 1} in {chunk.duration_ms}ms"
            )

            return audio_buffer, chunk

        except Exception as exc:
            logger.error(f"Job {job_id}: Error processing chunk {chunk_index + 1}: {exc}")
            raise

    async def _batch_write_audio_files(
        self,
        job_id: str,
        chunk_audio_data: List[Tuple[int, Any, LongTextChunk]],
    ) -> List[Path]:
        """Write generated audio buffers to disk after GPU work completes"""

        job_paths = self.job_manager._get_job_file_paths(job_id)
        written_paths: List[Path] = []

        for chunk_index, audio_buffer, chunk in chunk_audio_data:
            chunk_audio_path = job_paths['chunks_dir'] / chunk.audio_file

            await asyncio.to_thread(
                self._write_audio_file,
                chunk_audio_path,
                audio_buffer.getvalue(),
            )

            written_paths.append(chunk_audio_path)

        logger.info(f"Job {job_id}: Wrote {len(written_paths)} audio files to disk")
        return written_paths

    def _write_audio_file(self, path: Path, data: bytes) -> None:
        """Write binary audio data to disk"""
        with open(path, 'wb') as file_obj:
            file_obj.write(data)

    async def _update_job_status(self, job_id: str, status: LongTextJobStatus, message: str = ""):
        """Update job status"""
        try:
            metadata = self.job_manager._load_job_metadata(job_id)
            if metadata:
                metadata.status = status
                if message:
                    logger.info(f"Job {job_id}: {message}")
                self.job_manager._save_job_metadata(metadata)
        except Exception as e:
            logger.error(f"Failed to update status for job {job_id}: {e}")

    async def _fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed"""
        try:
            logger.error(f"Job {job_id} failed: {error_message}")

            metadata = self.job_manager._load_job_metadata(job_id)
            if metadata:
                metadata.status = LongTextJobStatus.FAILED
                metadata.error = error_message
                metadata.processing_completed_at = datetime.utcnow()
                if metadata.processing_started_at:
                    metadata.total_processing_time_ms = int(
                        (metadata.processing_completed_at - metadata.processing_started_at).total_seconds() * 1000
                    )
                self.job_manager._save_job_metadata(metadata)
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")

    def get_active_job_count(self) -> int:
        """Get the number of currently active jobs"""
        return len(self.active_tasks)

    def get_active_job_ids(self) -> list:
        """Get list of currently active job IDs"""
        return list(self.active_tasks.keys())

    async def pause_job(self, job_id: str) -> bool:
        """Pause a currently processing job"""
        if job_id in self.active_tasks:
            task = self.active_tasks[job_id]
            task.cancel()

            # The task cancellation will be handled by the processing loop
            # and will update the job status appropriately
            return True

        return False


# Global processor instance
_processor: Optional[LongTextProcessor] = None


def get_processor() -> LongTextProcessor:
    """Get the global processor instance"""
    global _processor
    if _processor is None:
        _processor = LongTextProcessor()
    return _processor


async def start_background_processor():
    """Start the background processor (called during app startup)"""
    processor = get_processor()
    await processor.start()


async def stop_background_processor():
    """Stop the background processor (called during app shutdown)"""
    processor = get_processor()
    await processor.stop()