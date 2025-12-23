"""
Language-specific model endpoints

Provides endpoints for managing and querying language-specific ChatterBox models
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel

from app.core import add_route_aliases
from app.core.tts_model import (
    list_available_languages,
    get_model_for_language,
    get_available_models
)
from app.core.language_models import (
    list_all_models,
    get_all_supported_languages,
    MODELS_BY_LANGUAGE
)


# Response models
class LanguageInfo(BaseModel):
    """Information about a language"""
    language_code: str
    language_name: str
    available_variants: List[str]
    default_model_id: str


class LanguageModelDetail(BaseModel):
    """Detailed information about a language model"""
    id: str
    language: str
    language_code: str
    repo_id: str
    format: str
    variant: str
    is_cached: bool
    is_loaded: bool
    cache_size_mb: Optional[float] = None


class LanguagesResponse(BaseModel):
    """Response for languages listing"""
    object: str = "list"
    languages: List[LanguageInfo]


class LanguageModelsResponse(BaseModel):
    """Response for language models listing"""
    object: str = "list"
    models: List[LanguageModelDetail]


# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="List supported languages",
    description="List all supported languages with their available model variants"
)
async def list_languages():
    """List all supported languages with available models"""
    languages_data = []

    for lang_code, models in MODELS_BY_LANGUAGE.items():
        if models:
            first_model = models[0]
            languages_data.append(
                LanguageInfo(
                    language_code=lang_code,
                    language_name=first_model.language,
                    available_variants=[m.variant for m in models],
                    default_model_id=first_model.model_id
                )
            )

    # Sort by language code
    languages_data.sort(key=lambda x: x.language_code)

    return LanguagesResponse(
        object="list",
        languages=languages_data
    )


@router.get(
    "/languages/{language_code}/models",
    response_model=LanguageModelsResponse,
    summary="List models for a language",
    description="List all available models for a specific language"
)
async def list_models_for_language(language_code: str):
    """List all available models for a specific language"""
    if language_code not in MODELS_BY_LANGUAGE:
        raise HTTPException(
            status_code=404,
            detail=f"No models available for language: {language_code}"
        )

    # Get all available models
    all_models = get_available_models()

    # Filter for this language
    language_models = [
        LanguageModelDetail(
            id=model["id"],
            language=model["name"],
            language_code=model.get("language_code", ""),
            repo_id=model.get("repo_id", ""),
            format=model.get("format", ""),
            variant=model.get("variant", ""),
            is_cached=model.get("is_cached", False),
            is_loaded=model["is_loaded"],
            cache_size_mb=model.get("cache_size_mb")
        )
        for model in all_models
        if model.get("model_type") == "language_specific"
        and model.get("language_code") == language_code
    ]

    if not language_models:
        raise HTTPException(
            status_code=404,
            detail=f"No models available for language: {language_code}"
        )

    return LanguageModelsResponse(
        object="list",
        models=language_models
    )


@router.get(
    "/language-models",
    response_model=LanguageModelsResponse,
    summary="List all language-specific models",
    description="List all available language-specific models with their details"
)
async def list_all_language_models():
    """List all language-specific models"""
    all_models = get_available_models()

    # Filter for language-specific models only
    language_models = [
        LanguageModelDetail(
            id=model["id"],
            language=model["name"],
            language_code=model.get("language_code", ""),
            repo_id=model.get("repo_id", ""),
            format=model.get("format", ""),
            variant=model.get("variant", ""),
            is_cached=model.get("is_cached", False),
            is_loaded=model["is_loaded"],
            cache_size_mb=model.get("cache_size_mb")
        )
        for model in all_models
        if model.get("model_type") == "language_specific"
    ]

    return LanguageModelsResponse(
        object="list",
        models=language_models
    )


# Export the base router for the main app to use
__all__ = ["base_router"]
