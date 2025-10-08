"""
Configuration management for Chatterbox TTS API
"""

import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class"""
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 4123))
    
    # TTS Model settings
    EXAGGERATION = float(os.getenv('EXAGGERATION', 0.5))
    CFG_WEIGHT = float(os.getenv('CFG_WEIGHT', 0.5))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
    
    # Text processing
    MAX_CHUNK_LENGTH = int(os.getenv('MAX_CHUNK_LENGTH', 280))
    MAX_TOTAL_LENGTH = int(os.getenv('MAX_TOTAL_LENGTH', 3000))
    
    # Voice and model settings
    VOICE_SAMPLE_PATH = os.getenv('VOICE_SAMPLE_PATH', './voice-sample.mp3')
    DEVICE_OVERRIDE = os.getenv('DEVICE', 'auto')
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './models')
    
    # Voice library settings
    VOICE_LIBRARY_DIR = os.getenv('VOICE_LIBRARY_DIR', './voices')

    # Long text processing settings
    LONG_TEXT_DATA_DIR = os.getenv('LONG_TEXT_DATA_DIR', './data/long_text_jobs')
    LONG_TEXT_MAX_LENGTH = int(os.getenv('LONG_TEXT_MAX_LENGTH', 100000))
    LONG_TEXT_CHUNK_SIZE = int(os.getenv('LONG_TEXT_CHUNK_SIZE', 2500))
    LONG_TEXT_SILENCE_PADDING_MS = int(os.getenv('LONG_TEXT_SILENCE_PADDING_MS', 200))
    LONG_TEXT_JOB_RETENTION_DAYS = int(os.getenv('LONG_TEXT_JOB_RETENTION_DAYS', 7))
    LONG_TEXT_MAX_CONCURRENT_JOBS = int(os.getenv('LONG_TEXT_MAX_CONCURRENT_JOBS', 3))

    # Multilingual model settings
    USE_MULTILINGUAL_MODEL = os.getenv('USE_MULTILINGUAL_MODEL', 'true').lower() == 'true'
    
    # Memory management settings
    MEMORY_CLEANUP_INTERVAL = int(os.getenv('MEMORY_CLEANUP_INTERVAL', 5))
    CUDA_CACHE_CLEAR_INTERVAL = int(os.getenv('CUDA_CACHE_CLEAR_INTERVAL', 3))
    ENABLE_MEMORY_MONITORING = os.getenv('ENABLE_MEMORY_MONITORING', 'true').lower() == 'true'
    MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT",60))
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    @classmethod
    def validate(cls):
        """Validate configuration values"""
        if not (0.25 <= cls.EXAGGERATION <= 2.0):
            raise ValueError(f"EXAGGERATION must be between 0.25 and 2.0, got {cls.EXAGGERATION}")
        if not (0.0 <= cls.CFG_WEIGHT <= 1.0):
            raise ValueError(f"CFG_WEIGHT must be between 0.0 and 1.0, got {cls.CFG_WEIGHT}")
        if not (0.05 <= cls.TEMPERATURE <= 5.0):
            raise ValueError(f"TEMPERATURE must be between 0.05 and 5.0, got {cls.TEMPERATURE}")
        if cls.MAX_CHUNK_LENGTH <= 0:
            raise ValueError(f"MAX_CHUNK_LENGTH must be positive, got {cls.MAX_CHUNK_LENGTH}")
        if cls.MAX_TOTAL_LENGTH <= 0:
            raise ValueError(f"MAX_TOTAL_LENGTH must be positive, got {cls.MAX_TOTAL_LENGTH}")
        if cls.MEMORY_CLEANUP_INTERVAL <= 0:
            raise ValueError(f"MEMORY_CLEANUP_INTERVAL must be positive, got {cls.MEMORY_CLEANUP_INTERVAL}")
        if cls.CUDA_CACHE_CLEAR_INTERVAL <= 0:
            raise ValueError(f"CUDA_CACHE_CLEAR_INTERVAL must be positive, got {cls.CUDA_CACHE_CLEAR_INTERVAL}")
        if cls.LONG_TEXT_MAX_LENGTH <= cls.MAX_TOTAL_LENGTH:
            raise ValueError(f"LONG_TEXT_MAX_LENGTH ({cls.LONG_TEXT_MAX_LENGTH}) must be greater than MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})")
        if cls.LONG_TEXT_CHUNK_SIZE <= 0:
            raise ValueError(f"LONG_TEXT_CHUNK_SIZE must be positive, got {cls.LONG_TEXT_CHUNK_SIZE}")
        if cls.LONG_TEXT_CHUNK_SIZE >= cls.MAX_TOTAL_LENGTH:
            raise ValueError(f"LONG_TEXT_CHUNK_SIZE ({cls.LONG_TEXT_CHUNK_SIZE}) must be less than MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})")
        if cls.LONG_TEXT_SILENCE_PADDING_MS < 0:
            raise ValueError(f"LONG_TEXT_SILENCE_PADDING_MS must be non-negative, got {cls.LONG_TEXT_SILENCE_PADDING_MS}")
        if cls.LONG_TEXT_JOB_RETENTION_DAYS <= 0:
            raise ValueError(f"LONG_TEXT_JOB_RETENTION_DAYS must be positive, got {cls.LONG_TEXT_JOB_RETENTION_DAYS}")
        if cls.LONG_TEXT_MAX_CONCURRENT_JOBS <= 0:
            raise ValueError(f"LONG_TEXT_MAX_CONCURRENT_JOBS must be positive, got {cls.LONG_TEXT_MAX_CONCURRENT_JOBS}")


def detect_device():
    """Detect the best available device"""
    if Config.DEVICE_OVERRIDE.lower() != 'auto':
        return Config.DEVICE_OVERRIDE.lower()
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu' 
    
