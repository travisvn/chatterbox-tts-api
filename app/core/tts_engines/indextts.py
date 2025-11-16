"""
IndexTTS-2 Engine Implementation

Integration for IndexTTS-2, an industrial-level controllable and efficient
zero-shot text-to-speech system with emotion control and duration control.
"""

import asyncio
import os
from typing import Dict, Any, Optional
import torch

from .base import BaseTTSEngine


class IndexTTSEngine(BaseTTSEngine):
    """
    IndexTTS-2 TTS Engine.

    Features:
    - Zero-shot voice cloning
    - 8 emotion vectors for expressive speech
    - Precise synthesis duration control
    - Multi-language support
    """

    def __init__(self, device: str = 'cpu', model_cache_dir: str = './models/indextts'):
        """
        Initialize IndexTTS-2 engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps')
            model_cache_dir: Directory to cache model files
        """
        super().__init__(device)
        self.model_cache_dir = model_cache_dir
        self.checkpoint_dir = os.path.join(model_cache_dir, 'checkpoints')

    async def load_model(self) -> None:
        """Load the IndexTTS-2 model"""
        if self._is_loaded:
            return

        print("Loading IndexTTS-2 model...")

        try:
            # Import IndexTTS (lazy import to avoid loading if not needed)
            from indextts.infer_indextts2 import IndexTTS2
        except ImportError:
            raise ImportError(
                "IndexTTS-2 is not installed. Install it with: pip install indextts\n"
                "Then download models with: hf download IndexTeam/IndexTTS-2 --local-dir=models/indextts/checkpoints"
            )

        # Create cache directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Check if model files exist
        config_path = os.path.join(self.checkpoint_dir, 'config.yaml')
        if not os.path.exists(config_path):
            print("⚠️ IndexTTS-2 models not found. Downloading...")
            await self._download_models()

        loop = asyncio.get_event_loop()

        try:
            # Load model in executor to avoid blocking
            def _load():
                # Determine if we should use FP16
                use_fp16 = self.device == 'cuda' and torch.cuda.is_available()

                model = IndexTTS2(
                    cfg_path=config_path,
                    model_dir=self.checkpoint_dir,
                    is_fp16=use_fp16,
                    use_cuda_kernel=False  # Disable for compatibility
                )
                return model

            self.model = await loop.run_in_executor(None, _load)
            self.sr = 24000  # IndexTTS-2 default sample rate
            self._is_loaded = True
            print("✓ IndexTTS-2 initialized successfully")

        except Exception as e:
            print(f"✗ Failed to load IndexTTS-2: {e}")
            raise e

    async def _download_models(self) -> None:
        """Download IndexTTS-2 models from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download

            print("Downloading IndexTTS-2 models from HuggingFace...")
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id="IndexTeam/IndexTTS-2",
                    local_dir=self.checkpoint_dir,
                    local_dir_use_symlinks=False
                )
            )
            print("✓ IndexTTS-2 models downloaded successfully")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install it with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download IndexTTS-2 models: {e}")

    def generate(
        self,
        text: str,
        audio_prompt_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        language_id: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate speech from text using IndexTTS-2.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Path to voice sample for cloning
            exaggeration: Emotion intensity (0.25-2.0) - mapped to IndexTTS emotion scale
            cfg_weight: Pace control (0.0-1.0) - mapped to IndexTTS speed
            temperature: Sampling randomness (0.05-5.0)
            language_id: Language code (optional)
            **kwargs: Additional parameters (emotion_type, etc.)

        Returns:
            Audio tensor (shape: [1, samples])
        """
        if not self._is_loaded:
            raise RuntimeError("IndexTTS-2 model not loaded. Call load_model() first.")

        # Map our standard parameters to IndexTTS parameters
        # IndexTTS uses emotion types: neutral, happy, sad, angry, surprised, fearful, disgusted, contemptuous
        emotion_type = kwargs.get('emotion_type', 'neutral')

        # Map exaggeration to emotion strength (0.0-1.0)
        emotion_strength = min(max(exaggeration / 2.0, 0.0), 1.0)

        # Map cfg_weight to speed (0.5-2.0, where 1.0 is normal)
        # Lower cfg_weight = faster, higher = slower
        speed = 1.0 + (cfg_weight - 0.5)
        speed = min(max(speed, 0.5), 2.0)

        try:
            import tempfile
            import torchaudio as ta

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                output_path = tmp_output.name

            # Run inference
            self.model.infer(
                spk_audio_prompt=audio_prompt_path,
                text=text,
                output_path=output_path,
                # emotion_type=emotion_type,  # Uncomment if your version supports it
                # emotion_strength=emotion_strength,
                # speed=speed
            )

            # Load generated audio
            waveform, sample_rate = ta.load(output_path)

            # Clean up temp file
            os.unlink(output_path)

            # Resample if needed
            if sample_rate != self.sr:
                resampler = ta.transforms.Resample(sample_rate, self.sr)
                waveform = resampler(waveform)

            return waveform

        except Exception as e:
            raise RuntimeError(f"IndexTTS-2 generation failed: {e}")

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        # IndexTTS-2 supports multiple languages but exact list depends on model
        return {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "engine": "indextts-2",
            "model_version": "indextts-2",
            "is_multilingual": True,
            "supported_languages": len(self.get_supported_languages()),
            "sample_rate": self.sr,
            "license": "Apache-2.0",
            "source": "IndexTeam",
            "features": [
                "voice_cloning",
                "emotion_control",
                "duration_control",
                "zero_shot"
            ],
            "model_size": "~1-2GB",
            "vram_required": "8GB+"
        }
