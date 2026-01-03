"""
Model listing endpoints (OpenAI compatibility)
"""

from fastapi import APIRouter

from app.models import ModelsResponse, ModelInfo, ModelCapabilities
from app.core import add_route_aliases
from app.core.tts_model import get_model_type, get_model_capabilities

base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models with current model and capabilities (OpenAI API compatibility)",
)
async def list_models():
    current_model = get_model_type()
    capabilities_dict = get_model_capabilities()

    capabilities = ModelCapabilities(
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

    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="chatterbox-tts-1",
                object="model",
                created=1677649963,
                owned_by="resemble-ai",
            ),
            ModelInfo(
                id="chatterbox-turbo",
                object="model",
                created=1735689600,
                owned_by="resemble-ai",
            ),
            ModelInfo(
                id="chatterbox-multilingual",
                object="model",
                created=1677649963,
                owned_by="resemble-ai",
            ),
        ],
        current_model=current_model,
        capabilities=capabilities,
    )


__all__ = ["base_router"]
