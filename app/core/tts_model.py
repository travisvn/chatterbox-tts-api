"""
TTS model initialization and management with multi-version support
Supports both multilingual models and language-specific models
"""

import os
import asyncio
from enum import Enum
from typing import Optional, Dict, Any, Literal, Union
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device
from app.core.language_models import (
    LANGUAGE_MODELS,
    MODELS_BY_ID,
    get_model_config,
    is_language_model,
    get_all_supported_languages,
    list_all_models
)
from app.core.model_downloader import (
    load_language_model,
    is_model_cached,
    get_model_info as get_language_model_info
)

# Model version type (includes both base models and language-specific models)
BaseModelVersion = Literal["chatterbox-v1", "chatterbox-v2", "chatterbox-multilingual-v1", "chatterbox-multilingual-v2"]
ModelVersion = Union[BaseModelVersion, str]  # str allows for language-specific model IDs

# Global model registry - stores multiple loaded models
_model_registry: Dict[str, Any] = {}
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_default_model_version = "chatterbox-multilingual-v2"
_supported_languages_by_model: Dict[str, Dict[str, str]] = {}


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def load_model(model_version: str) -> Any:
    """Load a specific model version (base or language-specific)"""
    global _device, _model_registry, _supported_languages_by_model

    if model_version in _model_registry:
        print(f"✓ Model {model_version} already loaded")
        return _model_registry[model_version]

    print(f"Loading {model_version}...")
    loop = asyncio.get_event_loop()

    try:
        if model_version in ["chatterbox-v1", "chatterbox-v2"]:
            # Standard English-only models
            model = await loop.run_in_executor(
                None,
                lambda: ChatterboxTTS.from_pretrained(device=_device)
            )
            _supported_languages_by_model[model_version] = {"en": "English"}
            print(f"✓ {model_version} initialized (English only)")

        elif model_version in ["chatterbox-multilingual-v1", "chatterbox-multilingual-v2"]:
            # Multilingual models
            model = await loop.run_in_executor(
                None,
                lambda: ChatterboxMultilingualTTS.from_pretrained(device=_device)
            )
            _supported_languages_by_model[model_version] = SUPPORTED_LANGUAGES.copy()
            print(f"✓ {model_version} initialized with {len(SUPPORTED_LANGUAGES)} languages")

        elif is_language_model(model_version):
            # Language-specific model from HuggingFace
            model_config = get_model_config(model_version)
            print(f"Loading language-specific model: {model_config.language} ({model_config.language_code})")
            print(f"  Repository: {model_config.repo_id}")
            print(f"  Variant: {model_config.variant}")

            # Load the model (with auto-download if needed)
            model = await loop.run_in_executor(
                None,
                lambda: load_language_model(
                    model_config,
                    cache_dir=Config.MODEL_CACHE_DIR,
                    device=_device,
                    auto_download=True
                )
            )

            # Set supported language for this model
            _supported_languages_by_model[model_version] = {
                model_config.language_code: model_config.language
            }
            print(f"✓ {model_version} initialized ({model_config.language})")

        else:
            raise ValueError(f"Unknown model version: {model_version}")

        _model_registry[model_version] = model
        return model

    except Exception as e:
        print(f"✗ Failed to load {model_version}: {e}")
        raise e


async def initialize_model():
    """Initialize the Chatterbox TTS model(s) based on configuration"""
    global _device, _initialization_state, _initialization_error, _initialization_progress
    global _default_model_version, _model_registry

    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()

        print(f"Initializing Chatterbox TTS models...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")

        _initialization_progress = "Creating model cache directory..."
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

        _initialization_progress = "Checking voice sample..."
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")

        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None

            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass

            def force_cpu_torch_load(f, map_location=None, **kwargs):
                return original_load(f, map_location='cpu', **kwargs)

            def force_cpu_load_file(filename, device=None):
                return original_load_file(filename, device='cpu')

            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file

        # Determine default model based on configuration
        use_multilingual = Config.USE_MULTILINGUAL_MODEL
        default_version = os.getenv("DEFAULT_MODEL_VERSION", "v2")

        if use_multilingual:
            _default_model_version = f"chatterbox-multilingual-{default_version}"
        else:
            _default_model_version = f"chatterbox-{default_version}"

        _initialization_progress = f"Loading default model ({_default_model_version})..."

        # Load the default model
        await load_model(_default_model_version)

        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Default model ({_default_model_version}) initialized successfully on {_device}")

        return _model_registry[_default_model_version]

    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model(model_version: Optional[str] = None):
    """Get a model instance by version, or the default model"""
    if model_version is None:
        model_version = _default_model_version
    return _model_registry.get(model_version)


async def get_or_load_model(model_version: Optional[str] = None):
    """Get a model, loading it if necessary"""
    if model_version is None:
        model_version = _default_model_version

    if model_version not in _model_registry:
        await load_model(model_version)

    return _model_registry[model_version]


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if at least one model is ready for use"""
    return _initialization_state == InitializationState.READY.value and len(_model_registry) > 0


def is_initializing():
    """Check if models are currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value


def is_multilingual(model_version: Optional[str] = None):
    """Check if the specified model supports multilingual generation"""
    if model_version is None:
        model_version = _default_model_version
    return "multilingual" in model_version


def get_supported_languages(model_version: Optional[str] = None):
    """Get the dictionary of supported languages for a model"""
    if model_version is None:
        model_version = _default_model_version
    return _supported_languages_by_model.get(model_version, {"en": "English"}).copy()


def supports_language(language_id: str, model_version: Optional[str] = None):
    """Check if the model supports a specific language"""
    if model_version is None:
        model_version = _default_model_version
    supported = _supported_languages_by_model.get(model_version, {"en": "English"})
    return language_id in supported


def get_loaded_models() -> list[str]:
    """Get list of currently loaded model versions"""
    return list(_model_registry.keys())


def get_available_models() -> list[Dict[str, Any]]:
    """Get list of all available model versions with their status"""
    # Base models
    base_models = [
        "chatterbox-v1",
        "chatterbox-v2",
        "chatterbox-multilingual-v1",
        "chatterbox-multilingual-v2"
    ]

    models_list = []

    # Add base models
    for model_id in base_models:
        models_list.append({
            "id": model_id,
            "name": model_id.replace("-", " ").title(),
            "is_multilingual": "multilingual" in model_id,
            "is_loaded": model_id in _model_registry,
            "is_default": model_id == _default_model_version,
            "supported_languages": _supported_languages_by_model.get(model_id, {"en": "English"}),
            "model_type": "base"
        })

    # Add language-specific models
    for lang_model in LANGUAGE_MODELS:
        model_id = lang_model.model_id
        model_info = get_language_model_info(lang_model, Config.MODEL_CACHE_DIR)

        models_list.append({
            "id": model_id,
            "name": f"{lang_model.language} ({lang_model.variant})" if lang_model.variant != "default" else lang_model.language,
            "is_multilingual": False,
            "is_loaded": model_id in _model_registry,
            "is_default": model_id == _default_model_version,
            "supported_languages": {lang_model.language_code: lang_model.language},
            "model_type": "language_specific",
            "language_code": lang_model.language_code,
            "repo_id": lang_model.repo_id,
            "format": lang_model.format,
            "variant": lang_model.variant,
            "is_cached": model_info["is_cached"],
            "cache_size_mb": model_info.get("cache_size_mb")
        })

    return models_list


def get_model_info(model_version: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive model information"""
    if model_version is None:
        model_version = _default_model_version

    is_loaded = model_version in _model_registry
    supported_langs = _supported_languages_by_model.get(model_version, {"en": "English"})

    info = {
        "model_version": model_version,
        "model_type": "multilingual" if "multilingual" in model_version else "standard",
        "is_multilingual": "multilingual" in model_version,
        "is_loaded": is_loaded,
        "is_default": model_version == _default_model_version,
        "supported_languages": supported_langs,
        "language_count": len(supported_langs),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state,
        "loaded_models": get_loaded_models(),
        "available_models": [m["id"] for m in get_available_models()]
    }

    # Add language-specific model information if applicable
    if is_language_model(model_version):
        model_config = get_model_config(model_version)
        lang_info = get_language_model_info(model_config, Config.MODEL_CACHE_DIR)
        info.update({
            "model_type": "language_specific",
            "language_code": model_config.language_code,
            "repo_id": model_config.repo_id,
            "format": model_config.format,
            "variant": model_config.variant,
            "is_cached": lang_info["is_cached"],
            "cache_size_mb": lang_info.get("cache_size_mb")
        })

    return info


def get_model_for_language(language_code: str, variant: str = "default") -> Optional[str]:
    """
    Get the model ID for a specific language

    Args:
        language_code: Language code (e.g., 'en', 'de', 'fr')
        variant: Model variant (e.g., 'default', 'havok2', 'SebastianBodza')

    Returns:
        Model ID or None if not found
    """
    from app.core.language_models import get_models_for_language

    models = get_models_for_language(language_code)

    if not models:
        return None

    # Find model with matching variant
    for model in models:
        if model.variant == variant:
            return model.model_id

    # If no exact variant match, return first available
    return models[0].model_id if models else None


async def load_model_for_language(language_code: str, variant: str = "default") -> Any:
    """
    Load a model for a specific language

    Args:
        language_code: Language code (e.g., 'en', 'de', 'fr')
        variant: Model variant (e.g., 'default', 'havok2', 'SebastianBodza')

    Returns:
        Loaded model instance

    Raises:
        ValueError: If no model is available for the language
    """
    model_id = get_model_for_language(language_code, variant)

    if model_id is None:
        raise ValueError(f"No model available for language: {language_code}")

    return await get_or_load_model(model_id)


def list_available_languages() -> Dict[str, List[str]]:
    """
    List all available languages with their variants

    Returns:
        Dictionary mapping language codes to lists of available variants
    """
    from app.core.language_models import MODELS_BY_LANGUAGE

    result = {}
    for lang_code, models in MODELS_BY_LANGUAGE.items():
        result[lang_code] = [model.variant for model in models]

    return result