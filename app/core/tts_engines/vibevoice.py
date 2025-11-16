"""
VibeVoice Engine Implementation

Integration for VibeVoice, an expressive, long-form conversational speech
synthesis system supporting multi-speaker podcasts and dialogues.
"""

import asyncio
import os
from typing import Dict, Any, Optional
import torch
import numpy as np

from .base import BaseTTSEngine


class VibeVoiceEngine(BaseTTSEngine):
    """
    VibeVoice TTS Engine.

    Features:
    - Expressive, long-form conversational speech synthesis
    - Multi-speaker support (up to 4 speakers)
    - Can synthesize speech up to 90 minutes (1.5B model) or 45 minutes (7B model)
    - Zero-shot voice cloning
    - Context-aware generation with background music
    """

    def __init__(
        self,
        device: str = 'cpu',
        model_cache_dir: str = './models/vibevoice',
        model_variant: str = '1.5b'
    ):
        """
        Initialize VibeVoice engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps')
            model_cache_dir: Directory to cache model files
            model_variant: Model variant ('1.5b' or '7b')
        """
        super().__init__(device)
        self.model_cache_dir = model_cache_dir
        self.model_variant = model_variant.lower()

        # Set model path based on variant
        if self.model_variant == '1.5b':
            self.model_path = "microsoft/VibeVoice-1.5B"
            self.context_length = 64000  # 64K tokens
            self.max_duration = 90  # minutes
        elif self.model_variant == '7b':
            self.model_path = "microsoft/VibeVoice-7B"
            self.context_length = 32000  # 32K tokens
            self.max_duration = 45  # minutes
        else:
            raise ValueError(f"Invalid model variant: {model_variant}. Choose '1.5b' or '7b'")

        self.inference_engine = None

    async def load_model(self) -> None:
        """Load the VibeVoice model"""
        if self._is_loaded:
            return

        print(f"Loading VibeVoice {self.model_variant.upper()} model...")
        print(f"⚠️ Note: VibeVoice is designed for long-form conversational speech")

        try:
            # Import VibeVoice (lazy import to avoid loading if not needed)
            try:
                from vibevoice import VibeVoice
            except ImportError:
                raise ImportError(
                    "VibeVoice is not installed. Install it with:\n"
                    "git clone https://github.com/vibevoice-community/VibeVoice.git\n"
                    "cd VibeVoice && pip install -e .\n"
                    "Or use: pip install git+https://github.com/vibevoice-community/VibeVoice.git"
                )

        except ImportError as e:
            raise e

        # Create cache directory
        os.makedirs(self.model_cache_dir, exist_ok=True)

        loop = asyncio.get_event_loop()

        try:
            # Load model in executor to avoid blocking
            def _load():
                # Check if model files exist locally
                local_model_path = os.path.join(self.model_cache_dir, self.model_variant)

                # Use local path if it exists, otherwise download from HuggingFace
                model_to_load = local_model_path if os.path.exists(local_model_path) else self.model_path

                # Initialize VibeVoice model
                from vibevoice import VibeVoice

                vibe_model = VibeVoice(
                    model_path=model_to_load,
                    device=self.device,
                    cache_dir=self.model_cache_dir
                )
                return vibe_model

            self.inference_engine = await loop.run_in_executor(None, _load)
            self.model = self.inference_engine  # For compatibility
            self.sr = 24000  # VibeVoice uses 24kHz sample rate
            self._is_loaded = True

            print(f"✓ VibeVoice {self.model_variant.upper()} initialized successfully")
            print(f"  Context: {self.context_length} tokens, Max duration: ~{self.max_duration} min")

        except Exception as e:
            print(f"✗ Failed to load VibeVoice: {e}")
            print("Note: Make sure you have installed VibeVoice from the repository.")
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
        Generate speech from text using VibeVoice.

        Args:
            text: Input text to synthesize (can be long-form conversation)
            audio_prompt_path: Path to voice sample for cloning
            exaggeration: Emotion intensity (0.25-2.0)
            cfg_weight: Pace control (0.0-1.0)
            temperature: Sampling randomness (0.05-5.0)
            language_id: Language code (optional)
            **kwargs: Additional parameters
                - speaker_names: List of speaker names for multi-speaker synthesis
                - top_p: Nucleus sampling parameter
                - seed: Random seed for reproducibility

        Returns:
            Audio tensor (shape: [1, samples])
        """
        if not self._is_loaded:
            raise RuntimeError("VibeVoice model not loaded. Call load_model() first.")

        try:
            import tempfile
            import torchaudio as ta

            # Extract additional parameters
            speaker_names = kwargs.get('speaker_names', ['Speaker'])
            top_p = kwargs.get('top_p', min(max(cfg_weight, 0.1), 1.0))
            seed = kwargs.get('seed', None)

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                output_path = tmp_output.name

            # Prepare generation parameters
            gen_params = {
                'text': text,
                'reference_audio': audio_prompt_path,
                'output_path': output_path,
                'temperature': temperature,
                'top_p': top_p,
            }

            # Add speaker names if provided
            if speaker_names and len(speaker_names) > 0:
                gen_params['speaker_names'] = speaker_names

            # Add seed if provided
            if seed is not None:
                gen_params['seed'] = seed

            # Try to use the inference engine's generate method
            try:
                # The actual API depends on the VibeVoice version
                # This is a best-effort implementation based on available documentation

                # Check if we have a direct generate method
                if hasattr(self.inference_engine, 'generate'):
                    self.inference_engine.generate(**gen_params)
                elif hasattr(self.inference_engine, 'infer'):
                    # Alternative method name
                    self.inference_engine.infer(**gen_params)
                else:
                    # Fallback: use the command-line interface approach
                    # This would require running the inference script
                    raise NotImplementedError(
                        "VibeVoice API not found. The integration may need adjustment "
                        "for your specific VibeVoice version. Please check the repository "
                        "for the correct API usage."
                    )

                # Load generated audio
                waveform, sample_rate = ta.load(output_path)

                # Clean up temp file
                os.unlink(output_path)

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

            except (AttributeError, TypeError) as api_error:
                # If the direct API approach doesn't work, provide helpful error
                raise NotImplementedError(
                    f"VibeVoice API integration error: {api_error}\n"
                    "The VibeVoice library API may have changed. Please check:\n"
                    "1. The VibeVoice repository for current API usage\n"
                    "2. Consider using the vibevoice-api package for OpenAI-compatible interface"
                )

        except Exception as e:
            # Clean up temp file if it exists
            if 'output_path' in locals() and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(f"VibeVoice generation failed: {e}")

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        # VibeVoice supports multiple languages
        # Based on the documentation, it primarily focuses on English and Chinese
        # but can handle other languages with varying quality
        return {
            "en": "English",
            "zh": "Chinese (中文)",
            "ja": "Japanese (日本語)",
            "ko": "Korean (한국어)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "ru": "Russian (Русский)",
            "ar": "Arabic (العربية)",
            "hi": "Hindi (हिन्दी)"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""

        # Model size depends on variant
        model_sizes = {
            '1.5b': "~3GB",
            '7b': "~14GB"
        }

        vram_requirements = {
            '1.5b': "8GB+ (16GB recommended)",
            '7b': "16GB+ (24GB recommended)"
        }

        return {
            "engine": "vibevoice",
            "model_version": f"vibevoice-{self.model_variant}",
            "model_variant": self.model_variant,
            "is_multilingual": True,
            "supported_languages": len(self.get_supported_languages()),
            "sample_rate": self.sr,
            "context_length": self.context_length,
            "max_generation_minutes": self.max_duration,
            "license": "MIT",
            "source": "Microsoft Research (Community Fork)",
            "features": [
                "voice_cloning",
                "multi_speaker",
                "long_form_generation",
                "conversational_speech",
                "context_aware",
                "expressive_synthesis"
            ],
            "model_size": model_sizes.get(self.model_variant, "Unknown"),
            "vram_required": vram_requirements.get(self.model_variant, "Unknown")
        }

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.inference_engine is not None:
            # Close/cleanup inference engine if it has such methods
            try:
                if hasattr(self.inference_engine, 'close'):
                    self.inference_engine.close()
                elif hasattr(self.inference_engine, 'cleanup'):
                    self.inference_engine.cleanup()
            except Exception as e:
                print(f"Warning during VibeVoice cleanup: {e}")

            del self.inference_engine
            self.inference_engine = None

        super().cleanup()
