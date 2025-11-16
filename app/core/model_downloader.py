"""
Model downloader utility for ChatterBox language-specific models

Handles downloading models from HuggingFace repositories with support
for both .pt and .safetensors formats.
"""

import os
import torch
from pathlib import Path
from typing import Optional
from app.core.language_models import LanguageModelConfig


def get_model_cache_path(model_config: LanguageModelConfig, cache_dir: str) -> Path:
    """Get the local cache path for a model"""
    cache_path = Path(cache_dir) / "language_models" / model_config.model_id
    return cache_path


def is_model_cached(model_config: LanguageModelConfig, cache_dir: str) -> bool:
    """Check if a model is already cached locally"""
    cache_path = get_model_cache_path(model_config, cache_dir)

    if not cache_path.exists():
        return False

    # Check for common model file patterns
    expected_files = [
        "pytorch_model.pt",
        "model.pt",
        "pytorch_model.bin",
        "model.safetensors",
        "chatterbox_model.pt",
        "chatterbox.pt"
    ]

    for file_name in expected_files:
        if (cache_path / file_name).exists():
            return True

    # Check if directory has any .pt or .safetensors files
    pt_files = list(cache_path.glob("*.pt")) + list(cache_path.glob("*.safetensors"))
    return len(pt_files) > 0


def download_model(
    model_config: LanguageModelConfig,
    cache_dir: str,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Download a language-specific model from HuggingFace

    Args:
        model_config: Model configuration
        cache_dir: Local cache directory
        progress_callback: Optional callback for progress updates

    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    cache_path = get_model_cache_path(model_config, cache_dir)

    if is_model_cached(model_config, cache_dir):
        if progress_callback:
            progress_callback(f"Model {model_config.model_id} already cached")
        return cache_path

    if progress_callback:
        progress_callback(f"Downloading {model_config.model_id} from {model_config.repo_id}...")

    print(f"Downloading language model: {model_config.model_id}")
    print(f"  Repository: {model_config.repo_id}")
    print(f"  Format: {model_config.format}")
    print(f"  Cache path: {cache_path}")

    try:
        # Download the entire repository
        # The ChatterBox models typically include config files and other necessary components
        snapshot_download(
            repo_id=model_config.repo_id,
            cache_dir=str(cache_path.parent),
            local_dir=str(cache_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        if progress_callback:
            progress_callback(f"✓ Downloaded {model_config.model_id}")

        print(f"✓ Model downloaded successfully to {cache_path}")
        return cache_path

    except Exception as e:
        error_msg = f"Failed to download model {model_config.model_id}: {e}"
        if progress_callback:
            progress_callback(f"✗ {error_msg}")
        print(f"✗ {error_msg}")
        raise RuntimeError(error_msg) from e


def load_language_model(
    model_config: LanguageModelConfig,
    cache_dir: str,
    device: str = "cpu",
    auto_download: bool = True
) -> torch.nn.Module:
    """
    Load a language-specific model, downloading if necessary

    Args:
        model_config: Model configuration
        cache_dir: Local cache directory
        device: Device to load the model on (cpu/cuda/mps)
        auto_download: Whether to automatically download if not cached

    Returns:
        Loaded model instance
    """
    # Check if model is cached
    if not is_model_cached(model_config, cache_dir):
        if not auto_download:
            raise FileNotFoundError(
                f"Model {model_config.model_id} not found in cache and auto_download is disabled"
            )
        download_model(model_config, cache_dir)

    cache_path = get_model_cache_path(model_config, cache_dir)

    print(f"Loading language model: {model_config.model_id}")
    print(f"  Path: {cache_path}")
    print(f"  Device: {device}")

    try:
        # Try to use ChatterboxTTS.from_pretrained with custom path
        from chatterbox.tts import ChatterboxTTS

        # The ChatterboxTTS library should support loading from local paths
        # We'll try loading from the cache path
        model = ChatterboxTTS.from_pretrained(
            model_dir=str(cache_path),
            device=device
        )

        return model

    except Exception as e:
        # If from_pretrained doesn't work with model_dir, try loading manually
        print(f"Note: from_pretrained failed, attempting manual load: {e}")

        try:
            from chatterbox.tts import ChatterboxTTS

            # Find the model file
            model_files = list(cache_path.glob("*.pt")) + list(cache_path.glob("*.safetensors"))

            if not model_files:
                raise FileNotFoundError(f"No model files found in {cache_path}")

            model_file = model_files[0]
            print(f"  Loading from file: {model_file}")

            # Load the model weights
            if model_file.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_file), device=device)
                except ImportError:
                    raise ImportError(
                        "safetensors is required for loading .safetensors models. "
                        "Install it with: pip install safetensors"
                    )
            else:
                state_dict = torch.load(str(model_file), map_location=device)

            # Create model and load weights
            model = ChatterboxTTS(device=device)
            model.load_state_dict(state_dict)

            return model

        except Exception as load_error:
            error_msg = f"Failed to load model {model_config.model_id}: {load_error}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg) from load_error


def get_model_info(model_config: LanguageModelConfig, cache_dir: str) -> dict:
    """Get information about a model's cache status"""
    cache_path = get_model_cache_path(model_config, cache_dir)
    is_cached = is_model_cached(model_config, cache_dir)

    info = {
        "id": model_config.model_id,
        "language": model_config.language,
        "language_code": model_config.language_code,
        "repo_id": model_config.repo_id,
        "format": model_config.format,
        "variant": model_config.variant,
        "is_cached": is_cached,
        "cache_path": str(cache_path) if is_cached else None
    }

    if is_cached:
        # Get cache size
        total_size = sum(
            f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
        )
        info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)

    return info
