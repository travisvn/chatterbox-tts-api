"""
TTS model initialization and management with multi-engine support

This module manages multiple TTS engines (Chatterbox, IndexTTS-2, Higgs Audio)
with lazy loading and a unified interface.
"""

import os
import asyncio
from enum import Enum
from typing import Optional, Dict, Any, Literal

from app.config import Config, detect_device
from app.core.tts_engines import BaseTTSEngine, ChatterboxEngine, IndexTTSEngine, HiggsAudioEngine

# Model version type - extended to include new models
ModelVersion = Literal[
    "chatterbox-v1",
    "chatterbox-v2",
    "chatterbox-multilingual-v1",
    "chatterbox-multilingual-v2",
    "indextts-2",
    "higgs-audio-v2"
]

# Global model registry - stores multiple loaded engines
_engine_registry: Dict[str, BaseTTSEngine] = {}
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_default_model_version = "chatterbox-multilingual-v2"


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


def _get_engine_class(model_version: str):
    """Get the appropriate engine class for a model version"""
    if model_version.startswith('chatterbox'):
        return ChatterboxEngine
    elif model_version == 'indextts-2':
        return IndexTTSEngine
    elif model_version == 'higgs-audio-v2':
        return HiggsAudioEngine
    else:
        raise ValueError(f"Unknown model version: {model_version}")


async def load_model(model_version: str) -> BaseTTSEngine:
    """Load a specific model version"""
    global _device, _engine_registry

    if model_version in _engine_registry:
        print(f"✓ Model {model_version} already loaded")
        return _engine_registry[model_version]

    print(f"Loading {model_version}...")

    try:
        # Get the appropriate engine class
        engine_class = _get_engine_class(model_version)

        # Create engine instance
        if model_version.startswith('chatterbox'):
            engine = engine_class(device=_device, model_version=model_version)
        elif model_version == 'indextts-2':
            engine = IndexTTSEngine(
                device=_device,
                model_cache_dir=os.path.join(Config.MODEL_CACHE_DIR, 'indextts')
            )
        elif model_version == 'higgs-audio-v2':
            engine = HiggsAudioEngine(
                device=_device,
                model_cache_dir=os.path.join(Config.MODEL_CACHE_DIR, 'higgs_audio')
            )
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        # Load the model
        await engine.load_model()

        # Register the engine
        _engine_registry[model_version] = engine

        # For backward compatibility, expose model and sr attributes
        engine.model = engine.model  # Already set by engine
        engine.sr = engine.sample_rate

        return engine

    except Exception as e:
        print(f"✗ Failed to load {model_version}: {e}")
        raise e


async def initialize_model():
    """Initialize the default TTS model based on configuration"""
    global _device, _initialization_state, _initialization_error, _initialization_progress
    global _default_model_version, _engine_registry

    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()

        print(f"Initializing TTS models...")
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

        # Check if user wants a different default engine
        default_engine = os.getenv("DEFAULT_TTS_ENGINE", "chatterbox")

        if default_engine == "indextts":
            _default_model_version = "indextts-2"
        elif default_engine == "higgs":
            _default_model_version = "higgs-audio-v2"
        else:
            # Default to Chatterbox
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

        return _engine_registry[_default_model_version]

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
    return _engine_registry.get(model_version)


async def get_or_load_model(model_version: Optional[str] = None):
    """Get a model, loading it if necessary"""
    if model_version is None:
        model_version = _default_model_version

    # Map OpenAI model names
    if model_version in ["tts-1", "tts-1-hd"]:
        model_version = _default_model_version

    if model_version not in _engine_registry:
        await load_model(model_version)

    return _engine_registry[model_version]


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
    return _initialization_state == InitializationState.READY.value and len(_engine_registry) > 0


def is_initializing():
    """Check if models are currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value


def is_multilingual(model_version: Optional[str] = None):
    """Check if the specified model supports multilingual generation"""
    if model_version is None:
        model_version = _default_model_version

    # Chatterbox models
    if "multilingual" in model_version:
        return True

    # Other engines
    engine = get_model(model_version)
    if engine:
        return len(engine.get_supported_languages()) > 1

    # Fallback
    return model_version in ["indextts-2", "higgs-audio-v2"]


def get_supported_languages(model_version: Optional[str] = None):
    """Get the dictionary of supported languages for a model"""
    if model_version is None:
        model_version = _default_model_version

    engine = get_model(model_version)
    if engine:
        return engine.get_supported_languages()

    # Fallback for not-yet-loaded models
    if "multilingual" in model_version:
        from app.core.mtl import SUPPORTED_LANGUAGES
        return SUPPORTED_LANGUAGES.copy()
    elif model_version == "indextts-2":
        return {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
    elif model_version == "higgs-audio-v2":
        return {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
    else:
        return {"en": "English"}


def supports_language(language_id: str, model_version: Optional[str] = None):
    """Check if the model supports a specific language"""
    if model_version is None:
        model_version = _default_model_version
    supported = get_supported_languages(model_version)
    return language_id in supported


def get_loaded_models() -> list[str]:
    """Get list of currently loaded model versions"""
    return list(_engine_registry.keys())


def get_available_models() -> list[Dict[str, Any]]:
    """Get list of all available model versions with their status"""
    all_models = [
        "chatterbox-v1",
        "chatterbox-v2",
        "chatterbox-multilingual-v1",
        "chatterbox-multilingual-v2",
        "indextts-2",
        "higgs-audio-v2"
    ]

    models_info = []
    for model_id in all_models:
        engine = get_model(model_id)
        if engine:
            # Get info from loaded engine
            info = engine.get_model_info()
            models_info.append({
                "id": model_id,
                "name": info.get("model_version", model_id).replace("-", " ").title(),
                "engine": info.get("engine", "unknown"),
                "is_multilingual": info.get("is_multilingual", False),
                "is_loaded": True,
                "is_default": model_id == _default_model_version,
                "supported_languages": engine.get_supported_languages(),
                "features": info.get("features", []),
                "model_size": info.get("model_size", "Unknown"),
                "vram_required": info.get("vram_required", "Unknown")
            })
        else:
            # Provide basic info for unloaded models
            is_multilingual = "multilingual" in model_id or model_id in ["indextts-2", "higgs-audio-v2"]
            models_info.append({
                "id": model_id,
                "name": model_id.replace("-", " ").title(),
                "engine": model_id.split("-")[0] if model_id != "higgs-audio-v2" else "higgs-audio",
                "is_multilingual": is_multilingual,
                "is_loaded": False,
                "is_default": model_id == _default_model_version,
                "supported_languages": get_supported_languages(model_id),
                "features": [],
                "model_size": "Not loaded",
                "vram_required": "Not loaded"
            })

    return models_info


def get_model_info(model_version: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive model information"""
    if model_version is None:
        model_version = _default_model_version

    engine = get_model(model_version)
    is_loaded = engine is not None

    if is_loaded:
        engine_info = engine.get_model_info()
        supported_langs = engine.get_supported_languages()
    else:
        engine_info = {}
        supported_langs = get_supported_languages(model_version)

    return {
        "model_version": model_version,
        "model_type": engine_info.get("engine", model_version.split("-")[0]),
        "is_multilingual": is_multilingual(model_version),
        "is_loaded": is_loaded,
        "is_default": model_version == _default_model_version,
        "supported_languages": supported_langs,
        "language_count": len(supported_langs),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state,
        "loaded_models": get_loaded_models(),
        "available_models": [m["id"] for m in get_available_models()],
        "features": engine_info.get("features", []),
        "model_size": engine_info.get("model_size", "Unknown"),
        "vram_required": engine_info.get("vram_required", "Unknown")
    }
