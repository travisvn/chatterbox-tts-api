"""
Pydantic models for long text TTS operations
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator
from uuid import UUID

from app.config import Config


class LongTextJobStatus(str, Enum):
    """Status enum for long text jobs"""
    PENDING = "pending"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LongTextJobActionType(str, Enum):
    """Action types for job operations"""
    CANCEL = "cancel"
    DELETE = "delete"


class LongTextRequest(BaseModel):
    """Request model for long text TTS generation"""
    input: str = Field(
        ..., description="Text to convert to speech (must meet the configured minimum length)"
    )
    voice: Optional[str] = Field(None, description="Voice name from library or OpenAI voice name")
    response_format: Optional[str] = Field("mp3", description="Audio format (mp3 or wav)")
    exaggeration: Optional[float] = Field(None, ge=0.25, le=2.0, description="Emotion intensity")
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pace control")
    temperature: Optional[float] = Field(None, ge=0.05, le=5.0, description="Sampling temperature")
    session_id: Optional[str] = Field(None, description="Frontend session ID for tracking")
    chunking_strategy: Optional[Literal["sentence", "paragraph", "word", "fixed"]] = Field(
        None, description="Strategy to use when chunking the text"
    )
    quality_preset: Optional[Literal["fast", "balanced", "high"]] = Field(
        None, description="Quality preset balancing speed vs fidelity"
    )
    chunk_size: Optional[int] = Field(
        None,
        gt=0,
        description="Custom chunk size override (takes precedence over presets and defaults)",
    )
    silence_padding_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Silence padding between chunks in milliseconds",
    )
    enable_pauses: Optional[bool] = Field(
        None,
        description="Enable punctuation-based pauses when generating chunk audio",
    )
    custom_pauses: Optional[Dict[str, int]] = Field(
        None,
        description="Custom pause patterns and durations in milliseconds",
    )

    @field_validator('input')
    @classmethod
    def validate_input_length(cls, v: str) -> str:
        cleaned = v.strip()
        text_length = len(cleaned)
        min_length = Config.get_long_text_min_length()
        max_length = Config.get_long_text_max_length()

        if text_length < min_length:
            raise ValueError(
                f"Input text must be at least {min_length} characters for long text processing"
            )

        if text_length > max_length:
            raise ValueError(
                f"Input text exceeds maximum length of {max_length} characters"
            )

        return cleaned

    def get_chunking_strategy(self) -> str:
        """Return the requested chunking strategy with configuration fallback."""

        return self.chunking_strategy or Config.LONG_TEXT_CHUNKING_STRATEGY

    def get_quality_preset(self) -> str:
        """Return the requested quality preset with configuration fallback."""

        return self.quality_preset or Config.LONG_TEXT_QUALITY_PRESET

    def get_chunk_size(self, preset_config: Dict[str, Any]) -> int:
        """Resolve the chunk size using custom value, preset, then config."""

        if self.chunk_size:
            return self.chunk_size
        return int(preset_config.get("chunk_size", Config.LONG_TEXT_CHUNK_SIZE))

    def get_silence_padding(self) -> int:
        """Resolve the silence padding with fallback to configuration."""

        if self.silence_padding_ms is not None:
            return self.silence_padding_ms
        return Config.LONG_TEXT_SILENCE_PADDING_MS

    def resolve_pause_settings(self) -> Dict[str, Any]:
        """Return pause handling configuration with defaults applied."""

        enable = (
            self.enable_pauses
            if self.enable_pauses is not None
            else Config.ENABLE_PUNCTUATION_PAUSES
        )
        return {
            "enable": bool(enable),
            "custom": self.custom_pauses or None,
        }

    @field_validator('custom_pauses')
    @classmethod
    def validate_custom_pauses(cls, value: Optional[Dict[str, Any]]):
        if value is None:
            return value

        cleaned: Dict[str, int] = {}
        for key, duration in value.items():
            if duration is None:
                raise ValueError(f'Pause duration for {key!r} cannot be None')
            try:
                int_duration = int(duration)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid pause duration for {key!r}: {duration!r}') from exc
            if int_duration < 0:
                raise ValueError(f'Pause duration for {key!r} must be non-negative')
            cleaned[str(key)] = int_duration

        return cleaned


class LongTextChunk(BaseModel):
    """Model for individual text chunk"""
    index: int = Field(..., ge=0, description="Chunk index (0-based)")
    text: str = Field(..., min_length=1, description="Chunk text content")
    text_preview: str = Field(..., description="First 50 characters for display")
    character_count: int = Field(..., ge=1, description="Number of characters in chunk")
    audio_file: Optional[str] = Field(None, description="Path to generated audio file")
    duration_ms: Optional[int] = Field(None, ge=0, description="Duration in milliseconds")
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    error: Optional[str] = None


class LongTextJobMetadata(BaseModel):
    """Metadata for a long text TTS job"""
    job_id: str = Field(..., description="Unique job identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: LongTextJobStatus = Field(default=LongTextJobStatus.PENDING)
    text_length: int = Field(..., ge=1, description="Total characters in input text")
    text_hash: str = Field(..., description="SHA256 hash of input text for deduplication")
    total_chunks: int = Field(..., ge=1, description="Total number of chunks")
    completed_chunks: int = Field(default=0, ge=0, description="Number of completed chunks")
    failed_chunks: List[int] = Field(default_factory=list, description="Indices of failed chunks")
    current_chunk: Optional[int] = Field(None, description="Currently processing chunk index")
    voice: Optional[str] = Field(None, description="Voice used for generation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="TTS parameters used")
    processing_started_at: Optional[datetime] = None
    processing_paused_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    total_processing_time_ms: int = Field(default=0, ge=0, description="Total processing time")
    output_format: str = Field(default="mp3", description="Output audio format")
    output_path: Optional[str] = Field(None, description="Path to final concatenated audio")
    output_size_bytes: Optional[int] = Field(None, ge=0, description="Final audio file size")
    output_duration_seconds: Optional[float] = Field(None, ge=0, description="Final audio duration")
    error: Optional[str] = None
    user_session_id: Optional[str] = Field(None, description="Frontend session ID")

    # History-specific fields
    completion_timestamp: Optional[datetime] = Field(None, description="When job fully completed successfully")
    audio_file_path: Optional[str] = Field(None, description="Persistent path to final audio file for history")
    audio_file_size: Optional[int] = Field(None, ge=0, description="Size of final audio file in bytes")
    total_duration_seconds: Optional[float] = Field(None, ge=0, description="Total audio duration in seconds")
    retry_count: int = Field(default=0, ge=0, description="Number of times job has been retried")
    original_job_id: Optional[str] = Field(None, description="Original job ID if this is a retry")
    is_archived: bool = Field(default=False, description="Whether job is archived in history")
    last_accessed: Optional[datetime] = Field(None, description="When user last interacted with this job")
    display_name: Optional[str] = Field(None, description="User-friendly name for the job")
    tags: List[str] = Field(default_factory=list, description="User-defined tags for organization")


class LongTextProgress(BaseModel):
    """Progress information for a long text job"""
    job_id: str = Field(..., description="Job identifier")
    overall_progress: float = Field(..., ge=0, le=100, description="Overall completion percentage")
    current_chunk: Optional[LongTextChunk] = None
    completed_chunks: List[LongTextChunk] = Field(default_factory=list)
    estimated_remaining_seconds: Optional[int] = Field(None, ge=0)
    status: LongTextJobStatus = Field(..., description="Current job status")
    error: Optional[str] = None


class LongTextJobResponse(BaseModel):
    """Response model for job status and progress"""
    job_id: str
    status: LongTextJobStatus
    progress: LongTextProgress
    metadata: LongTextJobMetadata
    created_at: datetime
    updated_at: datetime
    download_url: Optional[str] = Field(None, description="URL to download completed audio")
    can_pause: bool = Field(default=False, description="Whether job can be paused")
    can_resume: bool = Field(default=False, description="Whether job can be resumed")
    can_cancel: bool = Field(default=True, description="Whether job can be cancelled")


class LongTextJobListItem(BaseModel):
    """Simplified model for job listing"""
    job_id: str
    status: LongTextJobStatus
    text_preview: str = Field(..., description="First 100 characters of input text")
    text_length: int
    progress_percentage: float = Field(..., ge=0, le=100)
    created_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    can_resume: bool = Field(default=False)

    # Enhanced fields for history
    voice: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None
    total_duration_seconds: Optional[float] = None
    audio_file_size: Optional[int] = None
    retry_count: int = Field(default=0, ge=0)
    is_archived: bool = Field(default=False)
    display_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    last_accessed: Optional[datetime] = None
    parameters: Dict[str, Any] = Field(default_factory=dict, description="TTS parameters used")


class LongTextJobList(BaseModel):
    """Response model for listing jobs"""
    jobs: List[LongTextJobListItem]
    total_jobs: int = Field(..., ge=0)
    active_jobs: int = Field(..., ge=0, description="Jobs that are pending or processing")
    completed_jobs: int = Field(..., ge=0, description="Successfully completed jobs")


class LongTextJobCreateResponse(BaseModel):
    """Response model for job creation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: LongTextJobStatus = Field(default=LongTextJobStatus.PENDING)
    message: str = Field(default="Job created successfully")
    estimated_processing_time_seconds: Optional[int] = Field(None, ge=0)
    total_chunks: int = Field(..., ge=1)
    status_url: str = Field(..., description="URL to check job status")
    sse_url: str = Field(..., description="URL for real-time progress updates")


class LongTextJobAction(BaseModel):
    """Generic model for job actions (pause, resume, cancel)"""
    success: bool = Field(..., description="Whether action was successful")
    message: str = Field(..., description="Action result message")
    status: LongTextJobStatus = Field(..., description="Job status after action")


class LongTextSSEEvent(BaseModel):
    """Server-sent event model for real-time updates"""
    job_id: str
    event_type: str = Field(..., description="Event type: progress, status_change, error, completed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LongTextHistoryFilter(BaseModel):
    """Filter options for history listing"""
    status: Optional[LongTextJobStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    voice: Optional[str] = None
    search_text: Optional[str] = None
    is_archived: Optional[bool] = None
    tags: List[str] = Field(default_factory=list)


class LongTextHistorySort(str, Enum):
    """Sort options for history listing"""
    CREATED_DESC = "created_desc"
    CREATED_ASC = "created_asc"
    COMPLETED_DESC = "completed_desc"
    COMPLETED_ASC = "completed_asc"
    DURATION_DESC = "duration_desc"
    DURATION_ASC = "duration_asc"
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    SIZE_DESC = "size_desc"
    SIZE_ASC = "size_asc"


class LongTextJobUpdateRequest(BaseModel):
    """Request model for updating job metadata"""
    display_name: Optional[str] = Field(None, max_length=200)
    is_archived: Optional[bool] = None
    tags: Optional[List[str]] = Field(None, max_items=20)

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if v is not None:
            for tag in v:
                if len(tag) > 50:
                    raise ValueError('Tag length cannot exceed 50 characters')
            return [tag.strip() for tag in v]
        return v


class LongTextJobRetryRequest(BaseModel):
    """Request model for retrying a failed job"""
    preserve_chunks: bool = Field(default=True, description="Whether to preserve successful chunks")
    new_parameters: Optional[Dict[str, Any]] = Field(None, description="Updated TTS parameters")


class LongTextJobDetails(BaseModel):
    """Detailed view of a job including chunk information"""
    metadata: LongTextJobMetadata
    chunks: List[LongTextChunk]
    input_text: str
    error_log: List[str] = Field(default_factory=list, description="Detailed error history")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class LongTextHistoryStats(BaseModel):
    """Statistics for user's long text TTS history"""
    total_jobs: int = Field(..., ge=0)
    completed_jobs: int = Field(..., ge=0)
    failed_jobs: int = Field(..., ge=0)
    total_audio_duration_seconds: float = Field(..., ge=0)
    total_storage_bytes: int = Field(..., ge=0)
    average_processing_time_seconds: float = Field(..., ge=0)
    success_rate_percentage: float = Field(..., ge=0, le=100)
    most_used_voice: Optional[str] = None
    jobs_by_month: Dict[str, int] = Field(default_factory=dict)


class BulkJobAction(BaseModel):
    """Request model for bulk operations on jobs"""
    job_ids: List[str] = Field(..., min_items=1, max_items=100)
    action: str = Field(..., pattern="^(delete|archive|unarchive|retry)$")
    confirm: bool = Field(..., description="Confirmation that user wants to proceed")


class BulkJobActionResponse(BaseModel):
    """Response model for bulk operations"""
    success_count: int = Field(..., ge=0)
    failed_count: int = Field(..., ge=0)
    total_count: int = Field(..., ge=0)
    failed_jobs: List[str] = Field(default_factory=list, description="Job IDs that failed to process")
    message: str