"""Request models for API validation"""

from typing import Dict, Optional

from pydantic import BaseModel, Field, validator


class TTSRequest(BaseModel):
    """Text-to-speech request model"""

    input: str = Field(..., description="The text to generate audio for", min_length=1, max_length=3000)
    model: Optional[str] = Field(None, description="Model version to use: chatterbox-v1, chatterbox-v2, chatterbox-multilingual-v1, chatterbox-multilingual-v2")
    voice: Optional[str] = Field("alloy", description="Voice to use (ignored - uses voice sample)")
    response_format: Optional[str] = Field("wav", description="Audio format (always returns WAV)")
    speed: Optional[float] = Field(1.0, description="Speed of speech (ignored)")
    stream_format: Optional[str] = Field("audio", description="Streaming format: 'audio' for raw audio stream, 'sse' for Server-Side Events")
    
    # Custom TTS parameters
    exaggeration: Optional[float] = Field(None, description="Emotion intensity", ge=0.25, le=2.0)
    cfg_weight: Optional[float] = Field(None, description="Pace control", ge=0.0, le=1.0)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.05, le=5.0)
    
    # Streaming-specific parameters
    streaming_chunk_size: Optional[int] = Field(None, description="Characters per streaming chunk", ge=50, le=500)
    streaming_strategy: Optional[str] = Field(None, description="Chunking strategy for streaming")
    streaming_buffer_size: Optional[int] = Field(None, description="Number of chunks to buffer", ge=1, le=10)
    streaming_quality: Optional[str] = Field(None, description="Speed vs quality trade-off")

    # Pause handling parameters
    enable_pauses: Optional[bool] = Field(
        None,
        description="Enable punctuation-based pauses (defaults to server configuration)",
    )
    custom_pauses: Optional[Dict[str, int]] = Field(
        None,
        description="Custom pause durations in milliseconds keyed by punctuation",
    )
    
    @validator('input')
    def validate_input(cls, v):
        if not v or not v.strip():
            raise ValueError('Input text cannot be empty')
        return v.strip()

    @validator('model')
    def validate_model(cls, v):
        if v is not None:
            allowed_models = [
                'chatterbox-v1',
                'chatterbox-v2',
                'chatterbox-multilingual-v1',
                'chatterbox-multilingual-v2',
                'indextts-2',  # IndexTTS-2 model
                'higgs-audio-v2',  # Higgs Audio V2 model
                'vibevoice-1.5b',  # VibeVoice 1.5B model
                'vibevoice-7b',  # VibeVoice 7B model
                'tts-1',  # OpenAI compatibility - maps to default
                'tts-1-hd'  # OpenAI compatibility - maps to default
            ]
            if v not in allowed_models:
                raise ValueError(f'model must be one of: {", ".join(allowed_models)}')
        return v
    
    @validator('stream_format')
    def validate_stream_format(cls, v):
        if v is not None:
            allowed_formats = ['audio', 'sse']
            if v not in allowed_formats:
                raise ValueError(f'stream_format must be one of: {", ".join(allowed_formats)}')
        return v
    
    @validator('streaming_strategy')
    def validate_streaming_strategy(cls, v):
        if v is not None:
            allowed_strategies = ['sentence', 'paragraph', 'fixed', 'word']
            if v not in allowed_strategies:
                raise ValueError(f'streaming_strategy must be one of: {", ".join(allowed_strategies)}')
        return v
    
    @validator('streaming_quality')
    def validate_streaming_quality(cls, v):
        if v is not None:
            allowed_qualities = ['fast', 'balanced', 'high']
            if v not in allowed_qualities:
                raise ValueError(f'streaming_quality must be one of: {", ".join(allowed_qualities)}')
        return v

    @validator('custom_pauses')
    def validate_custom_pauses(cls, value):
        if value is None:
            return value

        cleaned: Dict[str, int] = {}
        for key, duration in value.items():
            if duration is None:
                raise ValueError('custom pause duration cannot be None')
            try:
                int_duration = int(duration)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'invalid pause duration for {key!r}: {duration!r}') from exc
            if int_duration < 0:
                raise ValueError(f'pause duration for {key!r} must be non-negative')
            cleaned[str(key)] = int_duration

        return cleaned