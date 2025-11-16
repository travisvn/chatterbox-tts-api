"""
Chatterbox TTS Engine Implementation

Wrapper for the official Chatterbox TTS models from ResembleAI.
Supports both standard (English-only) and multilingual versions.
"""

import asyncio
from typing import Dict, Any, Optional
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from .base import BaseTTSEngine
from app.core.mtl import SUPPORTED_LANGUAGES


class ChatterboxEngine(BaseTTSEngine):
    """
    Chatterbox TTS Engine.

    Supports multiple versions:
    - chatterbox-v1: Standard English-only (experimental fork)
    - chatterbox-v2: Standard English-only (official)
    - chatterbox-multilingual-v1: Multilingual 23 languages (experimental fork)
    - chatterbox-multilingual-v2: Multilingual 23 languages (official)
    """

    def __init__(self, device: str = 'cpu', model_version: str = 'chatterbox-multilingual-v2'):
        """
        Initialize Chatterbox engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps')
            model_version: Model version to load
        """
        super().__init__(device)
        self.model_version = model_version
        self.is_multilingual = 'multilingual' in model_version

    async def load_model(self) -> None:
        """Load the Chatterbox TTS model"""
        if self._is_loaded:
            return

        print(f"Loading Chatterbox model: {self.model_version}...")
        loop = asyncio.get_event_loop()

        try:
            if self.is_multilingual:
                # Load multilingual model
                self.model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                )
                self.sr = self.model.sr
                print(f"✓ {self.model_version} initialized with {len(SUPPORTED_LANGUAGES)} languages")
            else:
                # Load standard English-only model
                self.model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxTTS.from_pretrained(device=self.device)
                )
                self.sr = self.model.sr
                print(f"✓ {self.model_version} initialized (English only)")

            self._is_loaded = True

        except Exception as e:
            print(f"✗ Failed to load {self.model_version}: {e}")
            raise e

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
        Generate speech from text using Chatterbox.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Path to voice sample for cloning
            exaggeration: Emotion intensity (0.25-2.0)
            cfg_weight: Pace control (0.0-1.0)
            temperature: Sampling randomness (0.05-5.0)
            language_id: Language code (for multilingual models)
            **kwargs: Additional parameters

        Returns:
            Audio tensor (shape: [channels, samples])
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.model_version} not loaded. Call load_model() first.")

        generate_kwargs = {
            "text": text,
            "audio_prompt_path": audio_prompt_path,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature
        }

        # Add language_id for multilingual models
        if self.is_multilingual and language_id:
            generate_kwargs["language_id"] = language_id

        return self.model.generate(**generate_kwargs)

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        if self.is_multilingual:
            return SUPPORTED_LANGUAGES.copy()
        else:
            return {"en": "English"}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "engine": "chatterbox",
            "model_version": self.model_version,
            "is_multilingual": self.is_multilingual,
            "supported_languages": len(self.get_supported_languages()),
            "sample_rate": self.sr if self.sr else 24000,
            "license": "MIT",
            "source": "ResembleAI",
            "features": [
                "voice_cloning",
                "emotion_control",
                "multilingual" if self.is_multilingual else "english_only"
            ]
        }
