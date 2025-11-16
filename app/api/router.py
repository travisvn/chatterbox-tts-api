"""
Main API router combining all endpoints
"""

from fastapi import APIRouter

from app.api.endpoints import speech, health, models, memory, config, status, voices, long_text, language_models

# Create main router
api_router = APIRouter()

# Include all endpoint routers (using base_router for consistent aliasing)
api_router.include_router(speech.base_router, tags=["Text-to-Speech"])
api_router.include_router(long_text.base_router, tags=["Long Text TTS"])
api_router.include_router(voices.base_router, tags=["Voice Library"])
api_router.include_router(health.base_router, tags=["Health"])
api_router.include_router(models.base_router, tags=["Models"])
api_router.include_router(language_models.base_router, tags=["Language Models"])
api_router.include_router(memory.base_router, tags=["Memory Management"])
api_router.include_router(config.base_router, tags=["Configuration"])
api_router.include_router(status.base_router, tags=["Status & Processing"]) 