"""
Base TTS Engine abstraction layer

This module provides a common interface for different TTS engines,
allowing the application to support multiple TTS models seamlessly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch


class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS engines.

    All TTS engine implementations must inherit from this class and
    implement the required abstract methods.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the TTS engine.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.model = None
        self.sr = None  # Sample rate
        self._is_loaded = False

    @abstractmethod
    async def load_model(self) -> None:
        """
        Load the TTS model.

        This method should download and initialize the model if needed.
        Should be called before generate().
        """
        pass

    @abstractmethod
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
        Generate speech from text.

        Args:
            text: Input text to synthesize
            audio_prompt_path: Path to voice sample for cloning
            exaggeration: Emotion intensity (0.25-2.0)
            cfg_weight: Pace control (0.0-1.0)
            temperature: Sampling randomness (0.05-5.0)
            language_id: Language code (e.g., 'en', 'es')
            **kwargs: Engine-specific parameters

        Returns:
            Audio tensor (shape: [channels, samples])
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._is_loaded

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of generated audio"""
        return self.sr if self.sr else 24000  # Default fallback

    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages.

        Returns:
            Dict mapping language codes to language names
            Example: {'en': 'English', 'es': 'Spanish'}
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict with model metadata (name, version, size, etc.)
        """
        pass

    def supports_language(self, language_id: str) -> bool:
        """
        Check if the engine supports a specific language.

        Args:
            language_id: Language code to check

        Returns:
            True if language is supported
        """
        return language_id in self.get_supported_languages()

    def cleanup(self) -> None:
        """
        Clean up resources.

        Override this method if your engine needs cleanup.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
