"""
Model listing endpoints (OpenAI compatibility)
"""

from fastapi import APIRouter

from app.models import ModelsResponse, ModelInfo
from app.core import add_route_aliases
from app.core.tts_model import get_available_models

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models (OpenAI API compatibility)"
)
async def list_models():
    """List available models (OpenAI API compatibility)"""
    available_models = get_available_models()

    # Convert to OpenAI-compatible format
    model_data = [
        ModelInfo(
            id=model["id"],
            object="model",
            created=1677649963,  # Timestamp for consistency
            owned_by="resemble-ai"
        )
        for model in available_models
    ]

    # Add OpenAI-compatible model names
    model_data.extend([
        ModelInfo(
            id="tts-1",
            object="model",
            created=1677649963,
            owned_by="openai-compatible"
        ),
        ModelInfo(
            id="tts-1-hd",
            object="model",
            created=1677649963,
            owned_by="openai-compatible"
        )
    ])

    return ModelsResponse(
        object="list",
        data=model_data
    )

# Export the base router for the main app to use
__all__ = ["base_router"] 