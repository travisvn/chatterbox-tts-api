"""
Configuration endpoint
"""

from fastapi import APIRouter

from app.models import ConfigResponse, ModelCapabilities
from app.config import Config
from app.core.tts_model import (
    get_device,
    get_model_capabilities,
    get_paralinguistic_tags,
    is_turbo,
)
from app.core import add_route_aliases, get_endpoint_info, get_version, get_version_info

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get configuration",
    description="Get current API configuration",
)
async def get_config():
    """Get current configuration"""
    device = get_device()
    version_info = get_version_info()

    return ConfigResponse(
        api_info={
            "name": version_info.get("name", "Chatterbox TTS API"),
            "version": version_info["version"],
            "api_version": version_info["api_version"],
            "description": version_info.get("description", ""),
            "license": version_info.get("license", "Unknown"),
            "author": version_info.get("author", "Unknown"),
            "python_version": version_info["python_version"],
            "platform": version_info["platform"],
        },
        server={"host": Config.HOST, "port": Config.PORT},
        model={
            "device": device or "unknown",
            "voice_sample_path": Config.VOICE_SAMPLE_PATH,
            "model_cache_dir": Config.MODEL_CACHE_DIR,
        },
        defaults={
            "exaggeration": Config.EXAGGERATION,
            "cfg_weight": Config.CFG_WEIGHT,
            "temperature": Config.TEMPERATURE,
            "max_chunk_length": Config.MAX_CHUNK_LENGTH,
            "max_total_length": Config.MAX_TOTAL_LENGTH,
        },
        memory_management={
            "memory_cleanup_interval": Config.MEMORY_CLEANUP_INTERVAL,
            "cuda_cache_clear_interval": Config.CUDA_CACHE_CLEAR_INTERVAL,
            "enable_memory_monitoring": Config.ENABLE_MEMORY_MONITORING,
        },
    )


@router.get(
    "/endpoints",
    summary="List all endpoints",
    description="Get information about all available endpoints and their aliases",
)
async def list_endpoints():
    endpoint_info = get_endpoint_info()

    result = {
        **endpoint_info,
        "description": "This API supports multiple endpoint formats for compatibility",
        "usage": {
            "primary_endpoints": "Clean, short paths (recommended for new integrations)",
            "v1_aliases": "OpenAI-compatible paths (for compatibility with existing tools)",
            "example": {
                "primary": "/audio/speech",
                "aliases": ["/v1/audio/speech"],
                "note": "Both paths work identically",
            },
        },
    }

    return result


@router.get(
    "/capabilities",
    response_model=ModelCapabilities,
    summary="Get model capabilities",
    description="Get current model capabilities and supported features",
)
async def get_capabilities():
    capabilities_dict = get_model_capabilities()

    return ModelCapabilities(
        model_type=capabilities_dict.get("model_type"),
        supports_paralinguistic_tags=capabilities_dict.get(
            "supports_paralinguistic_tags", False
        ),
        supports_exaggeration=capabilities_dict.get("supports_exaggeration", True),
        supports_cfg_weight=capabilities_dict.get("supports_cfg_weight", True),
        supports_temperature=capabilities_dict.get("supports_temperature", True),
        supported_languages=capabilities_dict.get("supported_languages", {}),
        is_multilingual=capabilities_dict.get("is_multilingual", False),
        paralinguistic_tags=capabilities_dict.get("paralinguistic_tags", []),
    )


@router.get(
    "/paralinguistic-tags",
    summary="Get paralinguistic tags",
    description="Get supported paralinguistic tags (Turbo model only)",
)
async def get_paralinguistic_tags_endpoint():
    if is_turbo():
        return {"supported": True, "tags": get_paralinguistic_tags()}
    return {"supported": False, "tags": []}


__all__ = ["base_router"]
