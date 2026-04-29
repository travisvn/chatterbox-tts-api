"""
Core functionality for Chatterbox TTS API
"""

from .memory import get_memory_info, cleanup_memory, empty_gpu_cache, safe_delete_tensors
from .text_processing import (
    split_text_into_chunks, 
    concatenate_audio_chunks, 
    split_text_for_streaming, 
    get_streaming_settings
)
from .tts_model import initialize_model, get_model
from .version import get_version, get_version_info
from .voice_library import get_voice_library, VoiceLibrary, SUPPORTED_VOICE_FORMATS
from .aliases import (
    alias_route, 
    add_route_aliases, 
    get_all_aliases, 
    add_custom_alias,
    add_multiple_aliases,
    remove_alias,
    get_endpoint_info,
    ENDPOINT_ALIASES
)
from .status import (
    TTSStatus,
    start_tts_request,
    update_tts_status,
    get_tts_status,
    get_tts_history,
    get_tts_statistics,
    clear_tts_history
)

__all__ = [
    "get_memory_info",
    "cleanup_memory", 
    "empty_gpu_cache",
    "safe_delete_tensors",
    "split_text_into_chunks",
    "concatenate_audio_chunks",
    "split_text_for_streaming",
    "get_streaming_settings",
    "initialize_model",
    "get_model",
    "get_version",
    "get_version_info",
    "get_voice_library",
    "VoiceLibrary", 
    "SUPPORTED_VOICE_FORMATS",
    "alias_route",
    "add_route_aliases",
    "get_all_aliases",
    "add_custom_alias",
    "add_multiple_aliases",
    "remove_alias",
    "get_endpoint_info",
    "ENDPOINT_ALIASES",
    "TTSStatus",
    "start_tts_request",
    "update_tts_status",
    "get_tts_status",
    "get_tts_history",
    "get_tts_statistics",
    "clear_tts_history"
] 