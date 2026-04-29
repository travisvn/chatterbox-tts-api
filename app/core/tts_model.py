"""
TTS model initialization and management
"""

import os
import asyncio
import torch
from enum import Enum
from typing import Optional, Dict, Any
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import perth

# Patch resemble-perth if it's broken (common on some platforms/Python versions)
if not getattr(perth, "PerthImplicitWatermarker", None):
    print("⚠️  PerthImplicitWatermarker not found or broken. Applying mock to prevent crash.")
    class MockWatermarker:
        def __init__(self, *args, **kwargs): pass
        def encode(self, audio, *args, **kwargs): return audio
        def decode(self, audio, *args, **kwargs): return None
    perth.PerthImplicitWatermarker = MockWatermarker

from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}

# Try to import safetensors
try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages
    
    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU/MPS compatibility
        # Chatterbox models are often saved with CUDA references which fail on non-CUDA systems
        if _device != 'cuda':
            print(f"Applying torch.load patch for {_device} compatibility...")
            original_load = torch.load
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # If map_location is explicitly provided (other than None), respect it.
                # Otherwise, default to 'cpu' for stability on non-CUDA systems.
                target_map = map_location if map_location is not None else 'cpu'
                return original_load(f, map_location=target_map, **kwargs)
            
            torch.load = force_cpu_torch_load
            
            # Also patch safetensors if available
            if HAS_SAFETENSORS:
                original_load_file = safetensors.torch.load_file
                def force_cpu_load_file(filename, device=None):
                    return original_load_file(filename, device='cpu')
                safetensors.torch.load_file = force_cpu_load_file
        
        # Determine if we should use multilingual model
        use_multilingual = Config.USE_MULTILINGUAL_MODEL
        
        _initialization_progress = "Loading TTS model (this may take a while)..."
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        def load_and_move_model(model_class, target_device):
            """Load model to CPU first, then move to target device for stability"""
            print(f"Loading {model_class.__name__} (CPU first)...")
            # We load to 'cpu' because our patch already forces cpu, 
            # but being explicit is better for clarity and future changes.
            model = model_class.from_pretrained(device='cpu')
            
            if target_device != 'cpu':
                print(f"Moving model components to {target_device}...")
                # chatterbox models usually have these components
                for attr in ['t3', 's3gen', 've']:
                    if hasattr(model, attr):
                        component = getattr(model, attr)
                        if hasattr(component, 'to'):
                            setattr(model, attr, component.to(target_device))
                
                # Clear MPS cache if needed as moving components can be memory intensive
                if target_device == 'mps' and hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()
                
                # Set the device attribute on the model
                model.device = target_device
            
            return model

        if use_multilingual:
            print(f"Loading Chatterbox Multilingual TTS model...")
            _model = await loop.run_in_executor(
                None, 
                lambda: load_and_move_model(ChatterboxMultilingualTTS, _device)
            )
            _is_multilingual = True
            _supported_languages = SUPPORTED_LANGUAGES.copy()
            print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
        else:
            print(f"Loading standard Chatterbox TTS model...")
            _model = await loop.run_in_executor(
                None, 
                lambda: load_and_move_model(ChatterboxTTS, _device)
            )
            _is_multilingual = False
            _supported_languages = {"en": "English"}  # Standard model only supports English
            print(f"✓ Standard model initialized (English only)")
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


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
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    return {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state
    }