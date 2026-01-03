"""
Text-to-speech endpoint
"""

import io
import os
import asyncio
import tempfile
import torch
import torchaudio as ta
import base64
import json
import struct
from typing import Optional, List, Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from fastapi.responses import StreamingResponse

from app.models import (
    TTSRequest,
    ErrorResponse,
    SSEAudioDelta,
    SSEAudioDone,
    SSEUsageInfo,
    SSEAudioInfo,
)
from app.config import Config
from app.core import (
    get_memory_info,
    cleanup_memory,
    safe_delete_tensors,
    split_text_into_chunks,
    concatenate_audio_chunks,
    add_route_aliases,
    TTSStatus,
    start_tts_request,
    update_tts_status,
    get_voice_library,
)
from app.core.tts_model import get_model, is_multilingual, is_turbo
from app.core.text_processing import split_text_for_streaming, get_streaming_settings

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)

# Request counter for memory management
REQUEST_COUNTER = 0

# Supported audio formats for voice uploads
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def create_wav_header(
    sample_rate: int, channels: int, bits_per_sample: int, data_size: int = 0xFFFFFFFF
) -> bytes:
    """Creates a WAV header for streaming."""
    header = io.BytesIO()
    header.write(b"RIFF")
    # Use a large, but not max, value for chunk size to avoid overflow issues in some players
    chunk_size = 36 + data_size if data_size != 0xFFFFFFFF else 0x7FFFFFFF - 36
    header.write(struct.pack("<I", chunk_size))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write(struct.pack("<I", 16))  # Subchunk1Size for PCM
    header.write(struct.pack("<H", 1))  # AudioFormat (1 for PCM)
    header.write(struct.pack("<H", channels))
    header.write(struct.pack("<I", sample_rate))
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    header.write(struct.pack("<I", byte_rate))
    block_align = channels * (bits_per_sample // 8)
    header.write(struct.pack("<H", block_align))
    header.write(struct.pack("<H", bits_per_sample))
    header.write(b"data")
    header.write(struct.pack("<I", data_size))  # Subchunk2Size
    return header.getvalue()


def resolve_voice_path_and_language(voice_name: Optional[str]) -> tuple[str, str]:
    """
    Resolve a voice name or alias to a file path and language.

    Args:
        voice_name: Voice name or alias from the request (can be None for default)

    Returns:
        Tuple of (path to the voice file, language code)
    """
    # If no voice specified, use default
    if not voice_name:
        return Config.VOICE_SAMPLE_PATH, "en"

    # Try to resolve from voice library (handles both names and aliases)
    voice_lib = get_voice_library()
    voice_path = voice_lib.get_voice_path(voice_name)
    voice_language = voice_lib.get_voice_language(voice_name)

    if voice_path is None:
        # Check if it's an OpenAI voice name without an alias mapping
        openai_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        if voice_name.lower() in openai_voices:
            print(
                f"🎵 Using default voice for OpenAI voice '{voice_name}' (no alias mapping)"
            )
            return Config.VOICE_SAMPLE_PATH, "en"

        # Voice not found, fall back to default voice and log a warning
        print(
            f"⚠️ Warning: Voice '{voice_name}' not found in voice library, using default voice"
        )
        return Config.VOICE_SAMPLE_PATH, "en"

    return voice_path, voice_language or "en"


def resolve_voice_path(voice_name: Optional[str]) -> str:
    """
    Resolve a voice name or alias to a file path (backward compatibility).

    Args:
        voice_name: Voice name or alias from the request (can be None for default)

    Returns:
        Path to the voice file (falls back to default if voice not found)
    """
    path, _ = resolve_voice_path_and_language(voice_name)
    return path


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "No filename provided",
                    "type": "invalid_request_error",
                }
            },
        )

    # Check file extension
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
                    "type": "invalid_request_error",
                }
            },
        )

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file, "size") and file.size and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"File too large. Maximum size: {max_size // (1024 * 1024)}MB",
                    "type": "invalid_request_error",
                }
            },
        )


async def generate_speech_internal(
    text: str,
    voice_sample_path: str,
    language_id: str = "en",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
) -> io.BytesIO:
    """Internal function to generate speech with given parameters"""
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1

    # Start TTS request tracking
    voice_source = (
        "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "default"
    )
    request_id = start_tts_request(
        text=text,
        voice_source=voice_source,
        parameters={
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "voice_sample_path": voice_sample_path,
        },
    )

    update_tts_status(request_id, TTSStatus.INITIALIZING, "Checking model availability")

    model = get_model()
    if model is None:
        update_tts_status(request_id, TTSStatus.ERROR, error_message="Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Model not loaded", "type": "model_error"}},
        )

    # Log memory usage before processing
    initial_memory = None
    if Config.ENABLE_MEMORY_MONITORING:
        initial_memory = get_memory_info()
        update_tts_status(
            request_id,
            TTSStatus.INITIALIZING,
            "Monitoring initial memory",
            memory_usage=initial_memory,
        )
        print(
            f"📊 Request #{REQUEST_COUNTER} - Initial memory: CPU {initial_memory['cpu_memory_mb']:.1f}MB",
            end="",
        )
        if torch.cuda.is_available():
            print(f", GPU {initial_memory['gpu_memory_allocated_mb']:.1f}MB allocated")
        else:
            print()

    # Validate total text length
    update_tts_status(request_id, TTSStatus.PROCESSING_TEXT, "Validating text length")
    if len(text) > Config.MAX_TOTAL_LENGTH:
        update_tts_status(
            request_id,
            TTSStatus.ERROR,
            error_message=f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )

    audio_chunks = []
    final_audio = None
    buffer = None

    try:
        # Get parameters with defaults
        exaggeration = exaggeration if exaggeration is not None else Config.EXAGGERATION
        cfg_weight = cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT
        temperature = temperature if temperature is not None else Config.TEMPERATURE

        # Split text into chunks
        update_tts_status(request_id, TTSStatus.CHUNKING, "Splitting text into chunks")
        chunks = split_text_into_chunks(text, Config.MAX_CHUNK_LENGTH)

        voice_source = (
            "uploaded file"
            if voice_sample_path != Config.VOICE_SAMPLE_PATH
            else "configured sample"
        )
        print(
            f"Processing {len(chunks)} text chunks with {voice_source} and parameters:"
        )
        print(f"  - Exaggeration: {exaggeration}")
        print(f"  - CFG Weight: {cfg_weight}")
        print(f"  - Temperature: {temperature}")

        # Update status with chunk information
        update_tts_status(
            request_id,
            TTSStatus.GENERATING_AUDIO,
            "Starting audio generation",
            current_chunk=0,
            total_chunks=len(chunks),
        )

        # Generate audio for each chunk with memory management
        loop = asyncio.get_event_loop()

        for i, chunk in enumerate(chunks):
            # Update progress
            current_step = f"Generating audio for chunk {i + 1}/{len(chunks)}"
            update_tts_status(
                request_id,
                TTSStatus.GENERATING_AUDIO,
                current_step,
                current_chunk=i + 1,
                total_chunks=len(chunks),
            )

            print(
                f"Generating audio for chunk {i + 1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'"
            )

            with torch.no_grad():
                if is_turbo():
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                    }
                    if exaggeration or cfg_weight or temperature:
                        print(
                            f"Note: exaggeration/cfg_weight/temperature ignored for Turbo model"
                        )
                else:
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "temperature": temperature,
                    }
                    if is_multilingual():
                        generate_kwargs["language_id"] = language_id

                audio_tensor = await loop.run_in_executor(
                    None, lambda: model.generate(**generate_kwargs)
                )

                # Ensure tensor is on the correct device and detached
                if hasattr(audio_tensor, "detach"):
                    audio_tensor = audio_tensor.detach()

                audio_chunks.append(audio_tensor)

            # Periodic memory cleanup during generation
            if i > 0 and i % 3 == 0:  # Every 3 chunks
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate all chunks with memory management
        if len(audio_chunks) > 1:
            update_tts_status(
                request_id, TTSStatus.CONCATENATING, "Concatenating audio chunks"
            )
            print("Concatenating audio chunks...")
            with torch.no_grad():
                final_audio = concatenate_audio_chunks(audio_chunks, model.sr)
        else:
            final_audio = audio_chunks[0]

        # Convert to WAV format
        update_tts_status(request_id, TTSStatus.FINALIZING, "Converting to WAV format")
        buffer = io.BytesIO()

        # Ensure final_audio is on CPU for saving
        if hasattr(final_audio, "cpu"):
            final_audio_cpu = final_audio.cpu()
        else:
            final_audio_cpu = final_audio

        ta.save(buffer, final_audio_cpu, model.sr, format="wav")
        buffer.seek(0)

        # Mark as completed
        update_tts_status(request_id, TTSStatus.COMPLETED, "Audio generation completed")
        print(f"✓ Audio generation completed. Size: {len(buffer.getvalue()):,} bytes")

        return buffer

    except Exception as e:
        # Update status with error
        update_tts_status(
            request_id,
            TTSStatus.ERROR,
            error_message=f"TTS generation failed: {str(e)}",
        )
        print(f"✗ TTS generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"TTS generation failed: {str(e)}",
                    "type": "generation_error",
                }
            },
        )

    finally:
        # Comprehensive cleanup
        try:
            # Clean up all audio chunks
            for chunk in audio_chunks:
                safe_delete_tensors(chunk)

            # Clean up final audio tensor
            if final_audio is not None:
                safe_delete_tensors(final_audio)
                if "final_audio_cpu" in locals():
                    safe_delete_tensors(final_audio_cpu)

            # Clear the list
            audio_chunks.clear()

            # Periodic memory cleanup
            if REQUEST_COUNTER % Config.MEMORY_CLEANUP_INTERVAL == 0:
                cleanup_memory()

            # Log memory usage after processing
            if Config.ENABLE_MEMORY_MONITORING:
                final_memory = get_memory_info()
                print(
                    f"📊 Request #{REQUEST_COUNTER} - Final memory: CPU {final_memory['cpu_memory_mb']:.1f}MB",
                    end="",
                )
                if torch.cuda.is_available():
                    print(
                        f", GPU {final_memory['gpu_memory_allocated_mb']:.1f}MB allocated"
                    )
                else:
                    print()

                # Calculate memory difference
                if "initial_memory" in locals():
                    cpu_diff = (
                        final_memory["cpu_memory_mb"] - initial_memory["cpu_memory_mb"]
                    )
                    print(f"📈 Memory change: CPU {cpu_diff:+.1f}MB", end="")
                    if torch.cuda.is_available():
                        gpu_diff = (
                            final_memory["gpu_memory_allocated_mb"]
                            - initial_memory["gpu_memory_allocated_mb"]
                        )
                        print(f", GPU {gpu_diff:+.1f}MB")
                    else:
                        print()

        except Exception as cleanup_error:
            print(f"⚠️ Warning during cleanup: {cleanup_error}")


async def generate_speech_streaming(
    text: str,
    voice_sample_path: str,
    language_id: str = "en",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    streaming_chunk_size: Optional[int] = None,
    streaming_strategy: Optional[str] = None,
    streaming_quality: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Streaming function to generate speech with real-time chunk yielding"""
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1

    # Start TTS request tracking
    voice_source = (
        "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "default"
    )
    request_id = start_tts_request(
        text=text,
        voice_source=voice_source,
        parameters={
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "voice_sample_path": voice_sample_path,
            "streaming": True,
            "streaming_chunk_size": streaming_chunk_size,
            "streaming_strategy": streaming_strategy,
            "streaming_quality": streaming_quality,
        },
    )

    update_tts_status(
        request_id, TTSStatus.INITIALIZING, "Checking model availability (streaming)"
    )

    model = get_model()
    if model is None:
        update_tts_status(request_id, TTSStatus.ERROR, error_message="Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Model not loaded", "type": "model_error"}},
        )

    # Log memory usage before processing
    initial_memory = None
    if Config.ENABLE_MEMORY_MONITORING:
        initial_memory = get_memory_info()
        update_tts_status(
            request_id,
            TTSStatus.INITIALIZING,
            "Monitoring initial memory (streaming)",
            memory_usage=initial_memory,
        )
        print(
            f"📊 Streaming Request #{REQUEST_COUNTER} - Initial memory: CPU {initial_memory['cpu_memory_mb']:.1f}MB",
            end="",
        )
        if torch.cuda.is_available():
            print(f", GPU {initial_memory['gpu_memory_allocated_mb']:.1f}MB allocated")
        else:
            print()

    # Validate total text length
    update_tts_status(request_id, TTSStatus.PROCESSING_TEXT, "Validating text length")
    if len(text) > Config.MAX_TOTAL_LENGTH:
        update_tts_status(
            request_id,
            TTSStatus.ERROR,
            error_message=f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )

    # WAV header info for streaming
    sample_rate = model.sr
    channels = 1
    bits_per_sample = 16

    # Generate and yield WAV header first
    try:
        # Get parameters with defaults
        exaggeration = exaggeration if exaggeration is not None else Config.EXAGGERATION
        cfg_weight = cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT
        temperature = temperature if temperature is not None else Config.TEMPERATURE

        # Get optimized streaming settings
        streaming_settings = get_streaming_settings(
            streaming_chunk_size, streaming_strategy, streaming_quality
        )

        # Split text using streaming-optimized chunking
        update_tts_status(
            request_id, TTSStatus.CHUNKING, "Splitting text for streaming"
        )
        chunks = split_text_for_streaming(
            text,
            chunk_size=streaming_settings["chunk_size"],
            strategy=streaming_settings["strategy"],
            quality=streaming_settings["quality"],
        )

        voice_source = (
            "uploaded file"
            if voice_sample_path != Config.VOICE_SAMPLE_PATH
            else "configured sample"
        )
        print(
            f"Streaming {len(chunks)} text chunks with {voice_source} and parameters:"
        )
        print(f"  - Exaggeration: {exaggeration}")
        print(f"  - CFG Weight: {cfg_weight}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Streaming Strategy: {streaming_settings['strategy']}")
        print(f"  - Streaming Chunk Size: {streaming_settings['chunk_size']}")
        print(f"  - Streaming Quality: {streaming_settings['quality']}")

        # Update status with chunk information
        update_tts_status(
            request_id,
            TTSStatus.GENERATING_AUDIO,
            "Starting streaming audio generation",
            current_chunk=0,
            total_chunks=len(chunks),
        )

        # Yield a proper WAV header for streaming
        wav_header = create_wav_header(sample_rate, channels, bits_per_sample)
        yield wav_header

        # Generate and stream audio for each chunk
        loop = asyncio.get_event_loop()
        total_samples = 0

        for i, chunk in enumerate(chunks):
            # Update progress
            current_step = f"Streaming audio for chunk {i + 1}/{len(chunks)} ({streaming_settings['strategy']} strategy)"
            update_tts_status(
                request_id,
                TTSStatus.GENERATING_AUDIO,
                current_step,
                current_chunk=i + 1,
                total_chunks=len(chunks),
            )

            print(
                f"Streaming audio for chunk {i + 1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'"
            )

            with torch.no_grad():
                if is_turbo():
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                    }
                else:
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "temperature": temperature,
                    }
                    if is_multilingual():
                        generate_kwargs["language_id"] = language_id

                audio_tensor = await loop.run_in_executor(
                    None,
                    lambda: model.generate(**generate_kwargs),
                )

                # Ensure tensor is on CPU for streaming
                if hasattr(audio_tensor, "cpu"):
                    audio_tensor = audio_tensor.cpu()

                # Convert tensor to raw 16-bit PCM data
                # Clamp values to [-1, 1] before conversion
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                audio_tensor_int = (audio_tensor * 32767).to(torch.int16)

                # Yield the raw audio data as bytes
                pcm_data = audio_tensor_int.numpy().tobytes()
                yield pcm_data

                total_samples += audio_tensor.shape[1]

                # Clean up this chunk
                safe_delete_tensors(audio_tensor, audio_tensor_int)
                del pcm_data

            # Periodic memory cleanup during generation
            if i > 0 and i % 3 == 0:  # Every 3 chunks
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Mark as completed
        update_tts_status(
            request_id, TTSStatus.COMPLETED, "Streaming audio generation completed"
        )
        print(
            f"✓ Streaming audio generation completed. Total samples: {total_samples:,}"
        )

    except Exception as e:
        # Update status with error
        update_tts_status(
            request_id, TTSStatus.ERROR, error_message=f"TTS streaming failed: {str(e)}"
        )
        print(f"✗ TTS streaming failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"TTS streaming failed: {str(e)}",
                    "type": "generation_error",
                }
            },
        )

    finally:
        # Periodic memory cleanup
        if REQUEST_COUNTER % Config.MEMORY_CLEANUP_INTERVAL == 0:
            cleanup_memory()

        # Log memory usage after processing
        if Config.ENABLE_MEMORY_MONITORING:
            final_memory = get_memory_info()
            print(
                f"📊 Streaming Request #{REQUEST_COUNTER} - Final memory: CPU {final_memory['cpu_memory_mb']:.1f}MB",
                end="",
            )
            if torch.cuda.is_available():
                print(
                    f", GPU {final_memory['gpu_memory_allocated_mb']:.1f}MB allocated"
                )
            else:
                print()


async def generate_speech_sse(
    text: str,
    voice_sample_path: str,
    language_id: str = "en",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    streaming_chunk_size: Optional[int] = None,
    streaming_strategy: Optional[str] = None,
    streaming_quality: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Generate Server-Side Events for speech streaming (OpenAI compatible format)"""
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1

    # Start TTS request tracking
    voice_source = (
        "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "default"
    )
    request_id = start_tts_request(
        text=text,
        voice_source=voice_source,
        parameters={
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "voice_sample_path": voice_sample_path,
            "streaming": True,
            "streaming_format": "sse",
            "streaming_chunk_size": streaming_chunk_size,
            "streaming_strategy": streaming_strategy,
            "streaming_quality": streaming_quality,
        },
    )

    update_tts_status(
        request_id,
        TTSStatus.INITIALIZING,
        "Checking model availability (SSE streaming)",
    )

    model = get_model()
    if model is None:
        update_tts_status(request_id, TTSStatus.ERROR, error_message="Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Model not loaded", "type": "model_error"}},
        )

    # Log memory usage before processing
    initial_memory = None
    if Config.ENABLE_MEMORY_MONITORING:
        initial_memory = get_memory_info()
        update_tts_status(
            request_id,
            TTSStatus.INITIALIZING,
            "Monitoring initial memory (SSE streaming)",
            memory_usage=initial_memory,
        )
        print(
            f"📊 SSE Request #{REQUEST_COUNTER} - Initial memory: CPU {initial_memory['cpu_memory_mb']:.1f}MB",
            end="",
        )
        if torch.cuda.is_available():
            print(f", GPU {initial_memory['gpu_memory_allocated_mb']:.1f}MB allocated")
        else:
            print()

    # Validate total text length
    update_tts_status(request_id, TTSStatus.PROCESSING_TEXT, "Validating text length")
    if len(text) > Config.MAX_TOTAL_LENGTH:
        update_tts_status(
            request_id,
            TTSStatus.ERROR,
            error_message=f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )

    # WAV header info for conversion
    sample_rate = model.sr
    channels = 1
    bits_per_sample = 16
    total_audio_chunks = 0
    total_input_tokens = len(text.split())  # Rough token count

    try:
        # Get parameters with defaults
        exaggeration = exaggeration if exaggeration is not None else Config.EXAGGERATION
        cfg_weight = cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT
        temperature = temperature if temperature is not None else Config.TEMPERATURE

        # Get optimized streaming settings
        streaming_settings = get_streaming_settings(
            streaming_chunk_size, streaming_strategy, streaming_quality
        )

        # Split text using streaming-optimized chunking
        update_tts_status(
            request_id, TTSStatus.CHUNKING, "Splitting text for SSE streaming"
        )
        chunks = split_text_for_streaming(
            text,
            chunk_size=streaming_settings["chunk_size"],
            strategy=streaming_settings["strategy"],
            quality=streaming_settings["quality"],
        )

        voice_source = (
            "uploaded file"
            if voice_sample_path != Config.VOICE_SAMPLE_PATH
            else "configured sample"
        )
        print(
            f"SSE Streaming {len(chunks)} text chunks with {voice_source} and parameters:"
        )
        print(f"  - Exaggeration: {exaggeration}")
        print(f"  - CFG Weight: {cfg_weight}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Streaming Strategy: {streaming_settings['strategy']}")
        print(f"  - Streaming Chunk Size: {streaming_settings['chunk_size']}")
        print(f"  - Streaming Quality: {streaming_settings['quality']}")

        # Update status with chunk information
        update_tts_status(
            request_id,
            TTSStatus.GENERATING_AUDIO,
            "Starting SSE audio generation",
            current_chunk=0,
            total_chunks=len(chunks),
        )

        # First, send an info event with audio parameters
        info_event = SSEAudioInfo(
            sample_rate=sample_rate, channels=channels, bits_per_sample=bits_per_sample
        )
        yield f"data: {info_event.model_dump_json()}\n\n"

        # Generate and stream audio for each chunk as SSE events
        loop = asyncio.get_event_loop()

        for i, chunk in enumerate(chunks):
            # Update progress
            current_step = f"SSE streaming audio for chunk {i + 1}/{len(chunks)} ({streaming_settings['strategy']} strategy)"
            update_tts_status(
                request_id,
                TTSStatus.GENERATING_AUDIO,
                current_step,
                current_chunk=i + 1,
                total_chunks=len(chunks),
            )

            print(
                f"SSE streaming audio for chunk {i + 1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'"
            )

            with torch.no_grad():
                if is_turbo():
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                    }
                else:
                    generate_kwargs = {
                        "text": chunk,
                        "audio_prompt_path": voice_sample_path,
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "temperature": temperature,
                    }
                    if is_multilingual():
                        generate_kwargs["language_id"] = language_id

                audio_tensor = await loop.run_in_executor(
                    None,
                    lambda: model.generate(**generate_kwargs),
                )

                # Ensure tensor is on CPU for processing
                if hasattr(audio_tensor, "cpu"):
                    audio_tensor = audio_tensor.cpu()

                # Convert tensor to raw 16-bit PCM data
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                audio_tensor_int = (audio_tensor * 32767).to(torch.int16)
                pcm_data = audio_tensor_int.numpy().tobytes()

                # Base64 encode the raw PCM data
                audio_base64 = base64.b64encode(pcm_data).decode("utf-8")

                # Create SSE event for this audio chunk
                sse_event = SSEAudioDelta(audio=audio_base64)

                # Format as SSE event
                sse_data = f"data: {sse_event.model_dump_json()}\n\n"
                yield sse_data

                total_audio_chunks += 1

                # Clean up this chunk
                safe_delete_tensors(audio_tensor, audio_tensor_int)
                del pcm_data

            # Periodic memory cleanup during generation
            if i > 0 and i % 3 == 0:  # Every 3 chunks
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Send completion event
        total_output_tokens = total_audio_chunks * 50  # Rough estimate
        total_tokens = total_input_tokens + total_output_tokens

        usage_info = SSEUsageInfo(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens,
        )
        completion_event = SSEAudioDone(usage=usage_info)

        # Format final SSE event
        final_sse_data = f"data: {completion_event.model_dump_json()}\n\n"
        yield final_sse_data

        # Mark as completed
        update_tts_status(
            request_id, TTSStatus.COMPLETED, "SSE audio generation completed"
        )
        print(f"✓ SSE audio generation completed. Total chunks: {total_audio_chunks}")

    except Exception as e:
        # Update status with error
        update_tts_status(
            request_id,
            TTSStatus.ERROR,
            error_message=f"TTS SSE streaming failed: {str(e)}",
        )
        print(f"✗ TTS SSE streaming failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"TTS SSE streaming failed: {str(e)}",
                    "type": "generation_error",
                }
            },
        )

    finally:
        # Periodic memory cleanup
        if REQUEST_COUNTER % Config.MEMORY_CLEANUP_INTERVAL == 0:
            cleanup_memory()

        # Log memory usage after processing
        if Config.ENABLE_MEMORY_MONITORING:
            final_memory = get_memory_info()
            print(
                f"📊 SSE Request #{REQUEST_COUNTER} - Final memory: CPU {final_memory['cpu_memory_mb']:.1f}MB",
                end="",
            )
            if torch.cuda.is_available():
                print(
                    f", GPU {final_memory['gpu_memory_allocated_mb']:.1f}MB allocated"
                )
            else:
                print()


@router.post(
    "/audio/speech",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}, "text/event-stream": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Generate speech from text",
    description="Generate speech audio from input text. Supports voice names from the voice library or defaults to configured voice sample. Use stream_format='sse' for Server-Side Events streaming.",
)
async def text_to_speech(request: TTSRequest):
    """Generate speech from text using Chatterbox TTS with voice selection support"""

    # Resolve voice name to file path and language
    voice_sample_path, language_id = resolve_voice_path_and_language(request.voice)

    # Check if SSE streaming is requested
    if request.stream_format == "sse":
        # Return SSE streaming response
        return StreamingResponse(
            generate_speech_sse(
                text=request.input,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                streaming_chunk_size=request.streaming_chunk_size,
                streaming_strategy=request.streaming_strategy,
                streaming_quality=request.streaming_quality,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        # Standard audio generation
        buffer = await generate_speech_internal(
            text=request.input,
            voice_sample_path=voice_sample_path,
            language_id=language_id,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        )

        # Create response
        response = StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

        return response


@router.post(
    "/audio/speech/upload",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}, "text/event-stream": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Generate speech with custom voice upload or library selection",
    description="Generate speech audio from input text with voice library selection or optional custom voice file upload. Use stream_format='sse' for Server-Side Events streaming.",
)
async def text_to_speech_with_upload(
    input: str = Form(
        ..., description="The text to generate audio for", min_length=1, max_length=3000
    ),
    voice: Optional[str] = Form(
        "alloy",
        description="Voice name from library or OpenAI voice name (defaults to configured sample)",
    ),
    response_format: Optional[str] = Form(
        "wav", description="Audio format (always returns WAV)"
    ),
    speed: Optional[float] = Form(1.0, description="Speed of speech (ignored)"),
    stream_format: Optional[str] = Form(
        "audio",
        description="Streaming format: 'audio' for raw audio stream, 'sse' for Server-Side Events",
    ),
    exaggeration: Optional[float] = Form(
        None, description="Emotion intensity (0.25-2.0)", ge=0.25, le=2.0
    ),
    cfg_weight: Optional[float] = Form(
        None, description="Pace control (0.0-1.0)", ge=0.0, le=1.0
    ),
    temperature: Optional[float] = Form(
        None, description="Sampling temperature (0.05-5.0)", ge=0.05, le=5.0
    ),
    streaming_chunk_size: Optional[int] = Form(
        None, description="Characters per streaming chunk (50-500)", ge=50, le=500
    ),
    streaming_strategy: Optional[str] = Form(
        None, description="Chunking strategy (sentence, paragraph, fixed, word)"
    ),
    streaming_quality: Optional[str] = Form(
        None, description="Quality preset (fast, balanced, high)"
    ),
    voice_file: Optional[UploadFile] = File(
        None, description="Optional voice sample file for custom voice cloning"
    ),
):
    """Generate speech from text using Chatterbox TTS with optional voice file upload"""

    # Validate input text
    if not input or not input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Input text cannot be empty",
                    "type": "invalid_request_error",
                }
            },
        )

    input = input.strip()

    # Validate stream_format
    if stream_format not in ["audio", "sse"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "stream_format must be 'audio' or 'sse'",
                    "type": "validation_error",
                }
            },
        )

    # Validate streaming parameters for SSE
    if stream_format == "sse":
        if streaming_strategy and streaming_strategy not in [
            "sentence",
            "paragraph",
            "fixed",
            "word",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "streaming_strategy must be one of: sentence, paragraph, fixed, word",
                        "type": "validation_error",
                    }
                },
            )

        if streaming_quality and streaming_quality not in ["fast", "balanced", "high"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "streaming_quality must be one of: fast, balanced, high",
                        "type": "validation_error",
                    }
                },
            )

    # Handle voice selection and file upload
    temp_voice_path = None
    voice_sample_path = Config.VOICE_SAMPLE_PATH  # Default
    language_id = "en"  # Default language

    # First, try to resolve voice name from library if no file uploaded
    if not voice_file:
        voice_sample_path, language_id = resolve_voice_path_and_language(voice)

    # If a file is uploaded, it takes priority over voice name
    if voice_file:
        try:
            # Validate the uploaded file
            validate_audio_file(voice_file)

            # Create temporary file for the voice sample
            file_ext = os.path.splitext(voice_file.filename.lower())[1]
            temp_voice_fd, temp_voice_path = tempfile.mkstemp(
                suffix=file_ext, prefix="voice_sample_"
            )

            # Read and save the uploaded file
            file_content = await voice_file.read()
            with os.fdopen(temp_voice_fd, "wb") as temp_file:
                temp_file.write(file_content)

            voice_sample_path = temp_voice_path
            print(
                f"Using uploaded voice file: {voice_file.filename} ({len(file_content):,} bytes)"
            )

        except HTTPException:
            raise
        except Exception as e:
            # Clean up temp file if it was created
            if temp_voice_path and os.path.exists(temp_voice_path):
                try:
                    os.unlink(temp_voice_path)
                except:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": f"Failed to process voice file: {str(e)}",
                        "type": "file_processing_error",
                    }
                },
            )

    try:
        # Check if SSE streaming is requested
        if stream_format == "sse":
            # Create async generator that handles cleanup
            async def sse_streaming_with_cleanup():
                try:
                    async for sse_event in generate_speech_sse(
                        text=input,
                        voice_sample_path=voice_sample_path,
                        language_id=language_id,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        streaming_chunk_size=streaming_chunk_size,
                        streaming_strategy=streaming_strategy,
                        streaming_quality=streaming_quality,
                    ):
                        yield sse_event
                finally:
                    # Clean up temporary voice file
                    if temp_voice_path and os.path.exists(temp_voice_path):
                        try:
                            os.unlink(temp_voice_path)
                            print(
                                f"🗑️ Cleaned up temporary voice file: {temp_voice_path}"
                            )
                        except Exception as e:
                            print(
                                f"⚠️ Warning: Failed to clean up temporary voice file: {e}"
                            )

            # Return SSE streaming response
            return StreamingResponse(
                sse_streaming_with_cleanup(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )
        else:
            # Generate speech using internal function
            buffer = await generate_speech_internal(
                text=input,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

            # Create response
            response = StreamingResponse(
                io.BytesIO(buffer.getvalue()),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"},
            )

            return response

    finally:
        # Clean up temporary voice file
        if temp_voice_path and os.path.exists(temp_voice_path):
            try:
                os.unlink(temp_voice_path)
                print(f"🗑️ Cleaned up temporary voice file: {temp_voice_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to clean up temporary voice file: {e}")


@router.post(
    "/audio/speech/stream",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Stream speech generation from text",
    description="Generate and stream speech audio in real-time. Supports voice names from the voice library or defaults to configured voice sample.",
)
async def stream_text_to_speech(request: TTSRequest):
    """Stream speech generation from text using Chatterbox TTS with voice selection support"""

    # Resolve voice name to file path and language
    voice_sample_path, language_id = resolve_voice_path_and_language(request.voice)

    # Create streaming response
    return StreamingResponse(
        generate_speech_streaming(
            text=request.input,
            voice_sample_path=voice_sample_path,
            language_id=language_id,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            streaming_chunk_size=request.streaming_chunk_size,
            streaming_strategy=request.streaming_strategy,
            streaming_quality=request.streaming_quality,
        ),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech_stream.wav",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for true streaming
        },
    )


@router.post(
    "/audio/speech/stream/upload",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Stream speech generation with custom voice upload",
    description="Generate and stream speech audio in real-time with optional custom voice file upload",
)
async def stream_text_to_speech_with_upload(
    input: str = Form(
        ..., description="The text to generate audio for", min_length=1, max_length=3000
    ),
    voice: Optional[str] = Form(
        "alloy",
        description="Voice name from library or OpenAI voice name (defaults to configured sample)",
    ),
    response_format: Optional[str] = Form(
        "wav", description="Audio format (always returns WAV)"
    ),
    speed: Optional[float] = Form(1.0, description="Speed of speech (ignored)"),
    exaggeration: Optional[float] = Form(
        None, description="Emotion intensity (0.25-2.0)", ge=0.25, le=2.0
    ),
    cfg_weight: Optional[float] = Form(
        None, description="Pace control (0.0-1.0)", ge=0.0, le=1.0
    ),
    temperature: Optional[float] = Form(
        None, description="Sampling temperature (0.05-5.0)", ge=0.05, le=5.0
    ),
    streaming_chunk_size: Optional[int] = Form(
        None, description="Characters per streaming chunk (50-500)", ge=50, le=500
    ),
    streaming_strategy: Optional[str] = Form(
        None, description="Chunking strategy (sentence, paragraph, fixed, word)"
    ),
    streaming_quality: Optional[str] = Form(
        None, description="Quality preset (fast, balanced, high)"
    ),
    voice_file: Optional[UploadFile] = File(
        None, description="Optional voice sample file for custom voice cloning"
    ),
):
    """Stream speech generation from text using Chatterbox TTS with optional voice file upload"""

    # Validate input text
    if not input or not input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Input text cannot be empty",
                    "type": "invalid_request_error",
                }
            },
        )

    input = input.strip()

    # Validate streaming parameters
    if streaming_strategy and streaming_strategy not in [
        "sentence",
        "paragraph",
        "fixed",
        "word",
    ]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "streaming_strategy must be one of: sentence, paragraph, fixed, word",
                    "type": "validation_error",
                }
            },
        )

    if streaming_quality and streaming_quality not in ["fast", "balanced", "high"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "streaming_quality must be one of: fast, balanced, high",
                    "type": "validation_error",
                }
            },
        )

    # Handle voice selection and file upload
    temp_voice_path = None
    voice_sample_path = Config.VOICE_SAMPLE_PATH  # Default
    language_id = "en"  # Default language

    # First, try to resolve voice name from library if no file uploaded
    if not voice_file:
        voice_sample_path, language_id = resolve_voice_path_and_language(voice)

    # If a file is uploaded, it takes priority over voice name
    if voice_file:
        try:
            # Validate the uploaded file
            validate_audio_file(voice_file)

            # Create temporary file for the voice sample
            file_ext = os.path.splitext(voice_file.filename.lower())[1]
            temp_voice_fd, temp_voice_path = tempfile.mkstemp(
                suffix=file_ext, prefix="voice_sample_"
            )

            # Read and save the uploaded file
            file_content = await voice_file.read()
            with os.fdopen(temp_voice_fd, "wb") as temp_file:
                temp_file.write(file_content)

            voice_sample_path = temp_voice_path
            print(
                f"Using uploaded voice file for streaming: {voice_file.filename} ({len(file_content):,} bytes)"
            )

        except HTTPException:
            raise
        except Exception as e:
            # Clean up temp file if it was created
            if temp_voice_path and os.path.exists(temp_voice_path):
                try:
                    os.unlink(temp_voice_path)
                except:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": f"Failed to process voice file: {str(e)}",
                        "type": "file_processing_error",
                    }
                },
            )

    # Create async generator that handles cleanup
    async def streaming_with_cleanup():
        try:
            async for chunk in generate_speech_streaming(
                text=input,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                streaming_chunk_size=streaming_chunk_size,
                streaming_strategy=streaming_strategy,
                streaming_quality=streaming_quality,
            ):
                yield chunk
        finally:
            # Clean up temporary voice file
            if temp_voice_path and os.path.exists(temp_voice_path):
                try:
                    os.unlink(temp_voice_path)
                    print(f"🗑️ Cleaned up temporary voice file: {temp_voice_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to clean up temporary voice file: {e}")

    # Create streaming response
    return StreamingResponse(
        streaming_with_cleanup(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech_stream.wav",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for true streaming
        },
    )


# Export the base router for the main app to use
__all__ = ["base_router"]
