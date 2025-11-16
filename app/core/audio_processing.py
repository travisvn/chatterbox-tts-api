"""
Audio processing utilities for long text TTS concatenation
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    from pydub.utils import make_chunks
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    # Log the import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"pydub import failed: {e}")
except Exception as e:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    # Log any other errors for debugging
    import logging
    logging.getLogger(__name__).error(f"Unexpected error importing pydub: {e}")

from app.config import Config

logger = logging.getLogger(__name__)


class AudioConcatenationError(Exception):
    """Exception raised when audio concatenation fails"""
    pass


def check_pydub_availability():
    """Check if pydub is available and properly configured"""
    if not PYDUB_AVAILABLE:
        raise AudioConcatenationError(
            "pydub is not available. Please install it with: pip install pydub"
        )

    # Test basic functionality
    try:
        # Create a small test audio segment
        test_audio = AudioSegment.silent(duration=100)  # 100ms of silence
        return True
    except Exception as e:
        raise AudioConcatenationError(f"pydub is not properly configured: {e}")


def concatenate_audio_files(audio_files: List[Union[str, Path]],
                          output_path: Union[str, Path],
                          output_format: str = "mp3",
                          silence_duration_ms: Optional[int] = None,
                          crossfade_duration_ms: int = 0,
                          normalize_volume: bool = True,
                          remove_source_files: bool = False) -> dict:
    """
    Concatenate multiple audio files into a single output file.

    Args:
        audio_files: List of paths to audio files to concatenate
        output_path: Path where the concatenated audio will be saved
        output_format: Output format ('mp3', 'wav', etc.)
        silence_duration_ms: Duration of silence between chunks (defaults to config)
        crossfade_duration_ms: Duration of crossfade between chunks (0 for no crossfade)
        normalize_volume: Whether to normalize volume across all chunks
        remove_source_files: Whether to delete source files after concatenation

    Returns:
        Dictionary with metadata about the concatenated audio:
        {
            'output_path': str,
            'duration_seconds': float,
            'file_size_bytes': int,
            'sample_rate': int,
            'channels': int
        }

    Raises:
        AudioConcatenationError: If concatenation fails
    """
    check_pydub_availability()

    if not audio_files:
        raise AudioConcatenationError("No audio files provided for concatenation")

    if silence_duration_ms is None:
        silence_duration_ms = Config.LONG_TEXT_SILENCE_PADDING_MS

    logger.info(f"Concatenating {len(audio_files)} audio files with {silence_duration_ms}ms silence padding")

    try:
        # Load all audio segments
        segments = []
        for i, audio_file in enumerate(audio_files):
            file_path = Path(audio_file)
            if not file_path.exists():
                raise AudioConcatenationError(f"Audio file not found: {audio_file}")

            try:
                # Detect format from extension
                file_format = file_path.suffix.lower().lstrip('.')
                if file_format == 'wav':
                    audio = AudioSegment.from_wav(str(file_path))
                elif file_format == 'mp3':
                    audio = AudioSegment.from_mp3(str(file_path))
                elif file_format in ['m4a', 'aac']:
                    audio = AudioSegment.from_file(str(file_path), format='m4a')
                else:
                    # Try to auto-detect
                    audio = AudioSegment.from_file(str(file_path))

                segments.append(audio)
                logger.debug(f"Loaded audio segment {i+1}/{len(audio_files)}: "
                           f"{len(audio)} ms, {audio.frame_rate} Hz, {audio.channels} channels")

            except Exception as e:
                raise AudioConcatenationError(f"Failed to load audio file {audio_file}: {e}")

        if not segments:
            raise AudioConcatenationError("No valid audio segments loaded")

        # Normalize audio properties if requested
        if normalize_volume:
            segments = _normalize_audio_levels(segments)

        # Ensure all segments have the same sample rate and channels
        segments = _standardize_audio_properties(segments)

        # Create silence segment for padding
        silence = AudioSegment.silent(
            duration=silence_duration_ms,
            frame_rate=segments[0].frame_rate
        )

        # Concatenate segments with silence or crossfade
        result = segments[0]

        for segment in segments[1:]:
            if crossfade_duration_ms > 0:
                # Add crossfade between segments
                result = result.append(segment, crossfade=crossfade_duration_ms)
            else:
                # Add silence then append segment
                if silence_duration_ms > 0:
                    result = result + silence
                result = result + segment

        # Export the concatenated audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set export parameters based on format
        export_params = _get_export_parameters(output_format)

        result.export(
            str(output_path),
            format=output_format,
            **export_params
        )

        # Get file metadata
        file_size = output_path.stat().st_size
        duration_seconds = len(result) / 1000.0

        metadata = {
            'output_path': str(output_path),
            'duration_seconds': duration_seconds,
            'file_size_bytes': file_size,
            'sample_rate': result.frame_rate,
            'channels': result.channels
        }

        logger.info(f"Audio concatenation successful: {duration_seconds:.1f}s, "
                   f"{file_size:,} bytes, saved to {output_path}")

        # Clean up source files if requested
        if remove_source_files:
            for audio_file in audio_files:
                try:
                    Path(audio_file).unlink()
                    logger.debug(f"Removed source file: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove source file {audio_file}: {e}")

        return metadata

    except AudioConcatenationError:
        raise
    except Exception as e:
        raise AudioConcatenationError(f"Audio concatenation failed: {e}")


def _normalize_audio_levels(segments: List[AudioSegment]) -> List[AudioSegment]:
    """Normalize volume levels across all audio segments"""
    if not segments:
        return segments

    try:
        # Calculate average dBFS across all segments
        valid_dbfs_values = [segment.dBFS for segment in segments if segment.dBFS is not None]

        # If no valid dBFS values, return segments as is
        if not valid_dbfs_values:
            logger.debug("No valid dBFS values found, skipping normalization")
            return segments

        total_dbfs = sum(valid_dbfs_values)
        avg_dbfs = total_dbfs / len(valid_dbfs_values)

        # Target level (slightly below 0 dBFS to prevent clipping)
        target_dbfs = -3.0

        # Normalize each segment
        normalized_segments = []
        for segment in segments:
            if segment.dBFS is not None:
                # Calculate gain adjustment
                gain_adjustment = target_dbfs - segment.dBFS
                # Apply gain with some limits to prevent extreme adjustments
                gain_adjustment = max(-20, min(20, gain_adjustment))
                normalized_segment = segment.apply_gain(gain_adjustment)
            else:
                normalized_segment = segment

            normalized_segments.append(normalized_segment)

        logger.debug(f"Normalized {len(segments)} audio segments")
        return normalized_segments

    except Exception as e:
        logger.warning(f"Failed to normalize audio levels: {e}")
        return segments


def _standardize_audio_properties(segments: List[AudioSegment]) -> List[AudioSegment]:
    """Ensure all segments have the same sample rate and channel count"""
    if not segments:
        return segments

    # Use properties from the first segment as reference
    reference_segment = segments[0]
    target_frame_rate = reference_segment.frame_rate
    target_channels = reference_segment.channels

    standardized_segments = []

    for i, segment in enumerate(segments):
        standardized_segment = segment

        # Convert to target frame rate if needed
        if segment.frame_rate != target_frame_rate:
            standardized_segment = standardized_segment.set_frame_rate(target_frame_rate)
            logger.debug(f"Converted segment {i} from {segment.frame_rate} Hz to {target_frame_rate} Hz")

        # Convert to target channel count if needed
        if segment.channels != target_channels:
            if target_channels == 1 and segment.channels == 2:
                # Convert stereo to mono
                standardized_segment = standardized_segment.set_channels(1)
            elif target_channels == 2 and segment.channels == 1:
                # Convert mono to stereo
                standardized_segment = standardized_segment.set_channels(2)
            logger.debug(f"Converted segment {i} from {segment.channels} to {target_channels} channels")

        standardized_segments.append(standardized_segment)

    return standardized_segments


def _get_export_parameters(output_format: str) -> dict:
    """Get optimal export parameters for the given format"""
    export_params = {}

    if output_format.lower() == 'mp3':
        export_params.update({
            'bitrate': '128k',
            'parameters': ['-q:a', '2']  # High quality VBR
        })
    elif output_format.lower() == 'wav':
        export_params.update({
            'parameters': ['-acodec', 'pcm_s16le']  # 16-bit PCM
        })

    return export_params


def create_silence_audio(duration_ms: int,
                        sample_rate: int = 22050,
                        channels: int = 1,
                        output_path: Optional[Union[str, Path]] = None,
                        output_format: str = "wav") -> Optional[str]:
    """
    Create a silence audio file of specified duration.

    Args:
        duration_ms: Duration of silence in milliseconds
        sample_rate: Sample rate for the audio
        channels: Number of audio channels
        output_path: Path to save the silence file (optional)
        output_format: Format for the output file

    Returns:
        Path to the created silence file if output_path is specified, None otherwise
    """
    check_pydub_availability()

    try:
        silence = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=sample_rate
        ).set_channels(channels)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_params = _get_export_parameters(output_format)
            silence.export(str(output_path), format=output_format, **export_params)

            return str(output_path)

        return None

    except Exception as e:
        raise AudioConcatenationError(f"Failed to create silence audio: {e}")


def validate_audio_file(file_path: Union[str, Path]) -> dict:
    """
    Validate and get metadata for an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with audio file metadata:
        {
            'valid': bool,
            'duration_seconds': float,
            'sample_rate': int,
            'channels': int,
            'format': str,
            'file_size_bytes': int,
            'error': str (if valid=False)
        }
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {'valid': False, 'error': f'File not found: {file_path}'}

    try:
        check_pydub_availability()

        # Load the audio file
        audio = AudioSegment.from_file(str(file_path))

        return {
            'valid': True,
            'duration_seconds': len(audio) / 1000.0,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'format': file_path.suffix.lower().lstrip('.'),
            'file_size_bytes': file_path.stat().st_size,
            'error': None
        }

    except Exception as e:
        return {'valid': False, 'error': str(e)}


def estimate_concatenation_time(num_files: int, total_duration_seconds: float) -> int:
    """
    Estimate the time required to concatenate audio files.

    Args:
        num_files: Number of files to concatenate
        total_duration_seconds: Total duration of all audio files

    Returns:
        Estimated processing time in seconds
    """
    # Base processing time: 0.1 seconds per second of audio
    base_time = total_duration_seconds * 0.1

    # File I/O overhead: 1 second per file
    io_overhead = num_files * 1

    # Additional overhead for format conversion, normalization, etc.
    processing_overhead = 5

    return max(10, int(base_time + io_overhead + processing_overhead))