"""
Audio processing utilities for long text TTS concatenation.

This module is configurable via environment variables and includes features like
parallel processing, automatic retries, and a production-ready in-memory cache.

Cache Configuration:
- AUDIO_CACHE_MAX_SIZE_MB: Max in-memory cache size in MB. Evicts old entries when full. (Default: 256)
- AUDIO_CACHE_CLEAR_INTERVAL_S: Automatically clear cache periodically (in seconds). 0 disables. (Default: 3600)

Performance & Limits:
- AUDIO_SILENCE_PADDING_MS: Default silence duration in ms. (Default: 250)
- AUDIO_MAX_FILES_TO_CONCATENATE: Max number of files per job. (Default: 5000)
- AUDIO_MAX_TOTAL_SIZE_MB: Max combined file size in MB. (Default: 2048)
- AUDIO_USE_PARALLEL_PROCESSING: Set 'true' or '1' to enable parallel mode. (Default: false)
- AUDIO_MAX_PARALLEL_WORKERS: Max threads for parallel mode. (Default: CPU cores)
- AUDIO_LARGE_FILE_THRESHOLD_MB: Warn if a single file exceeds this size. (Default: 100)
"""

import concurrent.futures
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, Union

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False; AudioSegment = None
    import logging
    logging.getLogger(__name__).warning(f"pydub import failed: {e}")
except Exception as e:
    PYDUB_AVAILABLE = False; AudioSegment = None
    import logging
    logging.getLogger(__name__).error(f"Unexpected error importing pydub: {e}")

logger = logging.getLogger(__name__)


# --- Configuration from Environment Variables ---

def _get_env_var_as_int(name: str, default: int) -> int:
    value_str = os.getenv(name)
    if value_str is None: return default
    try: return int(value_str)
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for env var {name}: '{value_str}'. Using default: {default}.")
        return default

def _get_env_var_as_bool(name: str, default: bool = False) -> bool:
    value_str = os.getenv(name, '').lower()
    if value_str in ('true', '1', 'yes', 'on'): return True
    if value_str in ('false', '0', 'no', 'off', ''): return False
    logger.warning(f"Invalid value for boolean env var {name}: '{value_str}'. Using default: {default}.")
    return default

# Performance & Limits
SILENCE_PADDING_MS = _get_env_var_as_int('AUDIO_SILENCE_PADDING_MS', 250)
MAX_FILES = _get_env_var_as_int('AUDIO_MAX_FILES_TO_CONCATENATE', 5000)
MAX_SIZE_MB = _get_env_var_as_int('AUDIO_MAX_TOTAL_SIZE_MB', 2048)
MAX_TOTAL_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
USE_PARALLEL_PROCESSING_DEFAULT = _get_env_var_as_bool('AUDIO_USE_PARALLEL_PROCESSING')
MAX_WORKERS = _get_env_var_as_int('AUDIO_MAX_PARALLEL_WORKERS', os.cpu_count() or 4)
LARGE_FILE_THRESHOLD_MB = _get_env_var_as_int('AUDIO_LARGE_FILE_THRESHOLD_MB', 100)
LARGE_FILE_THRESHOLD_BYTES = LARGE_FILE_THRESHOLD_MB * 1024 * 1024

# Cache Configuration
CACHE_MAX_SIZE_MB = _get_env_var_as_int('AUDIO_CACHE_MAX_SIZE_MB', 256)
CACHE_MAX_SIZE_BYTES = CACHE_MAX_SIZE_MB * 1024 * 1024
CACHE_CLEAR_INTERVAL_S = _get_env_var_as_int('AUDIO_CACHE_CLEAR_INTERVAL_S', 3600)


# --- Type Definitions & Caching ---

class AudioMetadata(TypedDict):
    output_path: str; duration_seconds: float; file_size_bytes: int; sample_rate: int; channels: int

class AudioConcatenationError(Exception): pass

_segment_cache: Dict[Tuple, AudioSegment] = {}
_last_cache_clear_time = time.time()


# --- Core Functions ---

def concatenate_audio_files(audio_files: List[Union[str, Path]],
                          output_path: Union[str, Path],
                          output_format: str = "mp3",
                          **kwargs) -> AudioMetadata:
    """
    Concatenate multiple audio files into a single output file.
    This is the main entry point function.
    """
    # This function is now a wrapper to keep the signature clean
    # while passing all arguments to the core logic.
    return _concatenate_audio_files_core(
        audio_files, output_path, output_format, **kwargs
    )

def _concatenate_audio_files_core(
        audio_files: List[Union[str, Path]], output_path: Union[str, Path],
        output_format: str, silence_duration_ms: Optional[int] = None,
        crossfade_duration_ms: int = 0, normalize_volume: bool = True,
        remove_source_files: bool = False, quality: str = 'medium',
        use_parallel_processing: bool = USE_PARALLEL_PROCESSING_DEFAULT,
        progress_callback: Optional[Callable[[int, int], None]] = None
) -> AudioMetadata:
    start_time = time.time()
    check_pydub_availability()
    _validate_concatenation_params(audio_files)
    _check_for_large_files(audio_files)

    if use_parallel_processing and _estimate_memory_usage(audio_files) > 1 * 1024 * 1024 * 1024:
        logger.warning("High memory usage expected for this parallel job. Consider sequential mode if issues occur.")

    silence_ms = silence_duration_ms if silence_duration_ms is not None else SILENCE_PADDING_MS
    processing_mode = "parallel" if use_parallel_processing else "sequential"
    logger.info(f"Concatenating {len(audio_files)} files in {processing_mode} mode (workers={MAX_WORKERS}).")

    result = None
    try:
        all_segments = [None] * len(audio_files)
        all_segments[0] = _load_from_cache_or_disk(audio_files[0], normalize_volume)
        if progress_callback: progress_callback(1, len(audio_files))

        target_frame_rate, target_channels = all_segments[0].frame_rate, all_segments[0].channels

        if len(audio_files) > 1:
            segments_to_process = audio_files[1:]
            if use_parallel_processing:
                # Parallel processing logic
                completed_count = 1
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_map = {executor.submit(_load_from_cache_or_disk, fp, normalize_volume, target_frame_rate, target_channels): i + 1 for i, fp in enumerate(segments_to_process)}
                    for future in concurrent.futures.as_completed(future_map):
                        index = future_map[future]
                        try: all_segments[index] = future.result()
                        except Exception as exc: raise AudioConcatenationError(f"Failed to process {audio_files[index]}") from exc
                        completed_count += 1
                        if progress_callback: progress_callback(completed_count, len(audio_files))
            else:
                # Sequential processing logic
                for i, file_path in enumerate(segments_to_process):
                    all_segments[i + 1] = _load_from_cache_or_disk(file_path, normalize_volume, target_frame_rate, target_channels)
                    if progress_callback: progress_callback(i + 2, len(audio_files))

        # Concatenation and export logic...
        silence = AudioSegment.silent(duration=silence_ms, frame_rate=target_frame_rate)
        result = all_segments[0]
        for segment in all_segments[1:]:
            if crossfade_duration_ms > 0: result = result.append(segment, crossfade=crossfade_duration_ms)
            else:
                if silence_ms > 0: result += silence
                result += segment

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.export(str(output_path), format=output_format, **_get_export_parameters(output_format, quality))

        duration_s = time.time() - start_time
        metadata: AudioMetadata = {
            'output_path': str(output_path), 'duration_seconds': len(result) / 1000.0,
            'file_size_bytes': output_path.stat().st_size, 'sample_rate': result.frame_rate, 'channels': result.channels
        }
        logger.info(f"Concatenation successful in {duration_s:.2f}s. Rate: {len(audio_files) / duration_s:.1f} files/sec.")

        if remove_source_files:
            for audio_file in audio_files:
                try: Path(audio_file).unlink()
                except OSError as e: logger.warning(f"Failed to remove source file {audio_file}: {e}")
        return metadata

    except Exception as e:
        raise AudioConcatenationError(f"Audio concatenation failed: {e}")
    finally:
        del result; gc.collect()


# --- Cache Management ---

def _manage_cache_size():
    """Evicts the oldest entries from the cache if it exceeds the configured size limit."""
    if not _segment_cache or CACHE_MAX_SIZE_BYTES == 0:
        return

    # Estimate size by summing the size of each AudioSegment object.
    current_size = sum(sys.getsizeof(seg.raw_data) for seg in _segment_cache.values())

    if current_size > CACHE_MAX_SIZE_BYTES:
        logger.warning(
            f"Cache size ({current_size // 1024**2}MB) exceeds limit ({CACHE_MAX_SIZE_MB}MB). Evicting oldest entries."
        )
        # Sort keys by last modified time (the second element in the tuple key)
        sorted_keys = sorted(_segment_cache.keys(), key=lambda k: k[1])
        # Evict the oldest 20% of entries
        num_to_evict = max(1, len(sorted_keys) // 5)
        keys_to_remove = sorted_keys[:num_to_evict]
        for key in keys_to_remove:
            del _segment_cache[key]
        logger.info(f"Evicted {len(keys_to_remove)} entries from cache.")

def _load_from_cache_or_disk(file_path: Union[str, Path], normalize: bool,
                             target_rate: Optional[int] = None, target_ch: Optional[int] = None) -> AudioSegment:
    """Wrapper to use in-memory cache for loading segments, with automated management."""
    global _last_cache_clear_time
    if CACHE_CLEAR_INTERVAL_S > 0 and (time.time() - _last_cache_clear_time > CACHE_CLEAR_INTERVAL_S):
        logger.info(f"Clearing segment cache due to interval ({CACHE_CLEAR_INTERVAL_S}s).")
        _segment_cache.clear()
        _last_cache_clear_time = time.time()

    try:
        path_obj = Path(file_path)
        last_modified = path_obj.stat().st_mtime
        cache_key = (str(path_obj), last_modified, normalize, target_rate, target_ch)
        if cache_key in _segment_cache:
            logger.debug(f"Cache hit for {path_obj.name}")
            return _segment_cache[cache_key].copy()
    except FileNotFoundError: pass

    segment = _load_and_prep_segment_with_retry(file_path, normalize, target_rate, target_ch)

    if 'cache_key' in locals():
        _segment_cache[cache_key] = segment.copy()
        _manage_cache_size() # Check and manage cache size after adding a new item.

    return segment

# --- Other Helper & Internal Functions ---

def _load_and_prep_segment_with_retry(file_path: Union[str, Path], normalize: bool,
                                      target_rate: Optional[int] = None, target_ch: Optional[int] = None,
                                      max_retries: int = 2) -> AudioSegment:
    """Loads a segment, retrying on transient failures."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return _load_and_prep_segment(file_path, normalize, target_rate, target_ch)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed for {Path(file_path).name}: {e}. Retrying...")
                time.sleep(0.1 * (attempt + 1))
    raise last_exception

def _load_and_prep_segment(file_path: Union[str, Path], normalize: bool,
                           target_rate: Optional[int] = None, target_ch: Optional[int] = None) -> AudioSegment:
    """The core logic to load, standardize, and normalize a single audio segment."""
    try:
        path = Path(file_path)
        segment = AudioSegment.from_file(str(path), format=detect_audio_format(path))
        if target_rate and target_ch:
            segment = _standardize_segment_properties(segment, target_rate, target_ch)
        if normalize:
            segment = _normalize_segment_peak(segment)
        return segment
    except Exception as e:
        raise AudioConcatenationError(f"Failed to process segment {Path(file_path).name}") from e

def _normalize_segment_peak(segment: AudioSegment) -> AudioSegment:
    if segment.max_dBFS == float('-inf'): return segment
    return segment.apply_gain(-1.0 - segment.max_dBFS)

def _standardize_segment_properties(segment: AudioSegment, target_rate: int, target_ch: int) -> AudioSegment:
    if segment.frame_rate != target_rate: segment = segment.set_frame_rate(target_rate)
    if segment.channels != target_ch: segment = segment.set_channels(target_ch)
    return segment

def _get_export_parameters(output_format: str, quality: str = 'medium') -> dict:
    presets = {
        'mp3': {'low': {'bitrate': '96k'}, 'medium': {'bitrate': '128k'}, 'high': {'bitrate': '192k'}, 'lossless': {'bitrate': '320k'}},
        'opus': {'low': {'bitrate': '64k'}, 'medium': {'bitrate': '96k'}, 'high': {'bitrate': '128k'}},
        'wav': {'medium': {'parameters': ['-acodec', 'pcm_s16le']}}
    }
    fmt = output_format.lower()
    if fmt in presets: return presets[fmt].get(quality, presets[fmt].get('medium', {}))
    return {}

def check_pydub_availability():
    if not PYDUB_AVAILABLE: raise AudioConcatenationError("pydub not available. Install with: pip install pydub")
    try: AudioSegment.silent(duration=10)
    except Exception as e: raise AudioConcatenationError(f"pydub not configured correctly: {e}")

def _validate_concatenation_params(audio_files: list):
    if not audio_files: raise AudioConcatenationError("No audio files provided.")
    if len(audio_files) > MAX_FILES: raise AudioConcatenationError(f"File count exceeds limit of {MAX_FILES}.")
    try:
        total_size = sum(Path(f).stat().st_size for f in audio_files)
        if total_size > MAX_TOTAL_SIZE_BYTES: raise AudioConcatenationError(f"Total file size exceeds limit of {MAX_SIZE_MB} MB.")
    except FileNotFoundError as e: raise AudioConcatenationError(f"Audio file not found: {e.filename}")

def _check_for_large_files(audio_files: List[Union[str, Path]]):
    try:
        for fp in audio_files:
            path = Path(fp)
            if path.stat().st_size > LARGE_FILE_THRESHOLD_BYTES:
                logger.warning(f"Processing large file: {path.name} ({path.stat().st_size // (1024*1024)} MB).")
    except FileNotFoundError: pass

def _estimate_memory_usage(audio_files: list) -> int:
    try: return int(sum(Path(f).stat().st_size * 2.5 for f in audio_files))
    except FileNotFoundError: return 0

def detect_audio_format(file_path: Path) -> Optional[str]:
    ext = file_path.suffix.lower()
    if ext in {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.opus'}: return ext[1:]
    return None
