"""
Model listing endpoints (OpenAI compatibility)
"""

from fastapi import APIRouter

from fastapi import APIRouter, HTTPException
from app.models import ModelsResponse, ModelInfo
from app.core import add_route_aliases
from app.core.tts_model import get_model, unload_model

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/audio/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models (OpenAI API compatibility)"
)
async def list_models():
    """List available models (OpenAI API compatibility)"""
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="chatterbox-tts-1",
                object="model", 
                created=1677649963,
                owned_by="resemble-ai"
            )
        ]
    )

# -----------------------
# Load / Unload endpoints
# -----------------------
@router.post("/load_model", summary="Load the TTS model")
async def load_model_endpoint():
    """Load the TTS model into memory (auto-load if not already loaded)."""
    try:
        await get_model()
        return {"status": "loaded", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/unload_model", summary="Unload the TTS model")
async def unload_model_endpoint():
    """Unload the TTS model from memory to free resources."""
    try:
        await unload_model()
        return {"status": "unloaded", "model_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {e}")

# Export the base router for the main app to use
__all__ = ["base_router"] 