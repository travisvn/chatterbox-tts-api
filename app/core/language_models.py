"""
Configuration for language-specific ChatterBox TTS models

This module defines all available language-specific models with their
HuggingFace repository locations and model file formats.
"""

from typing import Dict, List, Literal
from dataclasses import dataclass

ModelFormat = Literal["pt", "safetensors"]


@dataclass
class LanguageModelConfig:
    """Configuration for a language-specific model"""
    language: str
    language_code: str
    repo_id: str
    format: ModelFormat
    variant: str = "default"  # e.g., "default", "havok2", "SebastianBodza"

    @property
    def model_id(self) -> str:
        """Generate unique model ID"""
        if self.variant == "default":
            return f"chatterbox-{self.language_code}"
        return f"chatterbox-{self.language_code}-{self.variant}"


# Language-specific model configurations
LANGUAGE_MODELS: List[LanguageModelConfig] = [
    # English (Original)
    LanguageModelConfig(
        language="English",
        language_code="en",
        repo_id="ResembleAI/chatterbox",
        format="pt",
        variant="default"
    ),

    # German
    LanguageModelConfig(
        language="German",
        language_code="de",
        repo_id="stlohrey/chatterbox_de",
        format="safetensors",
        variant="default"
    ),
    LanguageModelConfig(
        language="German",
        language_code="de",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="havok2"
    ),
    LanguageModelConfig(
        language="German",
        language_code="de",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="SebastianBodza"
    ),

    # Italian
    LanguageModelConfig(
        language="Italian",
        language_code="it",
        repo_id="niobures/Chatterbox-TTS",
        format="pt",
        variant="default"
    ),

    # French
    LanguageModelConfig(
        language="French",
        language_code="fr",
        repo_id="Thomcles/ChatterBox-fr",
        format="safetensors",
        variant="default"
    ),

    # Russian
    LanguageModelConfig(
        language="Russian",
        language_code="ru",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="default"
    ),

    # Armenian
    LanguageModelConfig(
        language="Armenian",
        language_code="hy",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="default"
    ),

    # Georgian
    LanguageModelConfig(
        language="Georgian",
        language_code="ka",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="default"
    ),

    # Japanese
    LanguageModelConfig(
        language="Japanese",
        language_code="ja",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="default"
    ),

    # Korean
    LanguageModelConfig(
        language="Korean",
        language_code="ko",
        repo_id="niobures/Chatterbox-TTS",
        format="safetensors",
        variant="default"
    ),

    # Norwegian
    LanguageModelConfig(
        language="Norwegian",
        language_code="no",
        repo_id="akhbar/chatterbox-tts-norwegian",
        format="safetensors",
        variant="default"
    ),
]


# Create lookup dictionaries for easy access
MODELS_BY_ID: Dict[str, LanguageModelConfig] = {
    model.model_id: model for model in LANGUAGE_MODELS
}

MODELS_BY_LANGUAGE: Dict[str, List[LanguageModelConfig]] = {}
for model in LANGUAGE_MODELS:
    if model.language_code not in MODELS_BY_LANGUAGE:
        MODELS_BY_LANGUAGE[model.language_code] = []
    MODELS_BY_LANGUAGE[model.language_code].append(model)


def get_model_config(model_id: str) -> LanguageModelConfig:
    """Get model configuration by ID"""
    if model_id not in MODELS_BY_ID:
        raise ValueError(f"Unknown language model: {model_id}")
    return MODELS_BY_ID[model_id]


def get_models_for_language(language_code: str) -> List[LanguageModelConfig]:
    """Get all available models for a specific language"""
    return MODELS_BY_LANGUAGE.get(language_code, [])


def get_default_model_for_language(language_code: str) -> LanguageModelConfig:
    """Get the default (first) model for a language"""
    models = get_models_for_language(language_code)
    if not models:
        raise ValueError(f"No models available for language: {language_code}")
    return models[0]


def get_all_supported_languages() -> Dict[str, str]:
    """Get all supported languages and their names"""
    return {
        model.language_code: model.language
        for model in LANGUAGE_MODELS
    }


def is_language_model(model_id: str) -> bool:
    """Check if a model ID is a language-specific model"""
    return model_id in MODELS_BY_ID


def list_all_models() -> List[Dict[str, str]]:
    """List all available language models with their details"""
    return [
        {
            "id": model.model_id,
            "language": model.language,
            "language_code": model.language_code,
            "repo_id": model.repo_id,
            "format": model.format,
            "variant": model.variant
        }
        for model in LANGUAGE_MODELS
    ]
