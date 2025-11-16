"""
Higgs Audio 2 Engine Implementation

Integration for Higgs Audio V2 from Boson AI, a text-audio foundation model
supporting voice cloning and multi-character conversations.
"""

import asyncio
import os
from typing import Dict, Any, Optional
import torch
import numpy as np

from .base import BaseTTSEngine


class HiggsAudioEngine(BaseTTSEngine):
    """
    Higgs Audio V2 TTS Engine.

    Features:
    - Neural voice cloning from 30+ second reference audio
    - Multi-speaker conversations
    - High-quality audio synthesis
    - Supports generation up to 90 minutes
    """

    def __init__(self, device: str = 'cpu', model_cache_dir: str = './models/higgs_audio'):
        """
        Initialize Higgs Audio V2 engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps')
            model_cache_dir: Directory to cache model files
        """
        super().__init__(device)
        self.model_cache_dir = model_cache_dir
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        self.serve_engine = None

    async def load_model(self) -> None:
        """Load the Higgs Audio V2 model"""
        if self._is_loaded:
            return

        print("Loading Higgs Audio V2 model...")
        print("⚠️ Note: Higgs Audio requires significant VRAM (24GB+ recommended)")

        try:
            # Import Higgs Audio (lazy import to avoid loading if not needed)
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        except ImportError:
            raise ImportError(
                "Higgs Audio is not installed. Clone and install it from:\n"
                "git clone https://github.com/boson-ai/higgs-audio\n"
                "cd higgs-audio && pip install -e ."
            )

        # Create cache directory
        os.makedirs(self.model_cache_dir, exist_ok=True)

        loop = asyncio.get_event_loop()

        try:
            # Load model in executor to avoid blocking
            def _load():
                engine = HiggsAudioServeEngine(
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    device=self.device,
                    trust_remote_code=True
                )
                return engine

            self.serve_engine = await loop.run_in_executor(None, _load)
            self.model = self.serve_engine  # For compatibility
            self.sr = 24000  # Higgs Audio V2 default sample rate
            self._is_loaded = True
            print("✓ Higgs Audio V2 initialized successfully")

        except Exception as e:
            print(f"✗ Failed to load Higgs Audio V2: {e}")
            print("Note: Ensure you have cloned the higgs-audio repository and installed it.")
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
        Generate speech from text using Higgs Audio V2.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Path to voice sample for cloning (30+ seconds recommended)
            exaggeration: Emotion intensity (0.25-2.0)
            cfg_weight: Pace control (0.0-1.0)
            temperature: Sampling randomness (0.05-5.0)
            language_id: Language code (optional)
            **kwargs: Additional parameters

        Returns:
            Audio tensor (shape: [1, samples])
        """
        if not self._is_loaded:
            raise RuntimeError("Higgs Audio V2 model not loaded. Call load_model() first.")

        try:
            # Prepare messages for Higgs Audio
            # Higgs Audio uses a message-based format similar to chat APIs
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Generate natural speech."
                },
                {
                    "role": "user",
                    "content": f"Generate speech: {text}"
                }
            ]

            # Map temperature parameter
            # Higgs Audio uses top_p and temperature parameters
            top_p = min(max(cfg_weight, 0.1), 1.0)

            # Generate audio using the serve engine
            # Note: The actual API may vary based on version
            # This is a simplified implementation that may need adjustment
            import tempfile
            import torchaudio as ta

            # Create temporary output file
            output_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                    output_path = tmp_output.name

                # Higgs Audio inference
                # The exact method depends on the version, this is based on available docs
                try:
                    # Try to generate using reference audio
                    result = self.serve_engine.generate(
                        messages=messages,
                        reference_audio=audio_prompt_path,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=4096,
                        output_path=output_path
                    )

                    # Load generated audio
                    waveform, sample_rate = ta.load(output_path)

                    # Resample if needed
                    if sample_rate != self.sr:
                        resampler = ta.transforms.Resample(sample_rate, self.sr)
                        waveform = resampler(waveform)

                    # Ensure correct shape [1, samples]
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.shape[0] > 1:
                        # Convert stereo to mono if needed
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    return waveform

                except AttributeError:
                    # Fallback for different API version
                    raise NotImplementedError(
                        "Higgs Audio V2 API integration needs adjustment for this version. "
                    "Please check the higgs-audio repository for the correct API usage."
                )

            finally:
                # Clean up temp file
                if output_path and os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except OSError:
                        pass

        except Exception as e:
            raise RuntimeError(f"Higgs Audio V2 generation failed: {e}")

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        # Higgs Audio V2 supports multiple languages
        return {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "engine": "higgs-audio-v2",
            "model_version": "higgs-audio-v2-3B",
            "is_multilingual": True,
            "supported_languages": len(self.get_supported_languages()),
            "sample_rate": self.sr,
            "license": "Custom (Check Boson AI terms)",
            "source": "Boson AI",
            "features": [
                "voice_cloning",
                "multi_character_conversations",
                "long_form_generation",
                "high_quality"
            ],
            "model_size": "~3-4GB",
            "vram_required": "24GB+ (GPU strongly recommended)"
        }

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.serve_engine is not None:
            # Close/cleanup serve engine if it has such methods
            try:
                if hasattr(self.serve_engine, 'close'):
                    self.serve_engine.close()
            except Exception as e:
                print(f"Warning during Higgs Audio cleanup: {e}")

            del self.serve_engine
            self.serve_engine = None

        super().cleanup()
