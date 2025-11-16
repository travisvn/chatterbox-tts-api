# VibeVoice Integration Guide

## Overview

VibeVoice has been successfully integrated into the Chatterbox TTS API as an additional TTS engine. VibeVoice is an expressive, long-form conversational speech synthesis system designed for creating podcast-like audio with multiple speakers.

## Available Models

Two VibeVoice models are now available:

### VibeVoice-1.5B
- **Model ID**: `vibevoice-1.5b`
- **HuggingFace Path**: `microsoft/VibeVoice-1.5B`
- **Context Length**: 64,000 tokens
- **Max Generation**: ~90 minutes
- **Model Size**: ~3GB
- **VRAM Required**: 8GB+ (16GB recommended)

### VibeVoice-7B
- **Model ID**: `vibevoice-7b`
- **HuggingFace Path**: `microsoft/VibeVoice-7B`
- **Context Length**: 32,000 tokens
- **Max Generation**: ~45 minutes
- **Model Size**: ~14GB
- **VRAM Required**: 16GB+ (24GB recommended)

## Features

Both VibeVoice models support:

- ✅ **Voice Cloning**: Zero-shot voice cloning from reference audio
- ✅ **Multi-Speaker**: Up to 4 distinct speakers in one generation
- ✅ **Long-Form**: Generate podcast-length audio (up to 90 minutes)
- ✅ **Conversational**: Natural dialogue and conversation synthesis
- ✅ **Context-Aware**: LLM-powered understanding of context
- ✅ **Expressive**: Natural prosody and emotion
- ✅ **Multilingual**: 12+ languages supported

## Supported Languages

Both models support:

- English (en)
- Chinese/中文 (zh)
- Japanese/日本語 (ja)
- Korean/한국어 (ko)
- Spanish/Español (es)
- French/Français (fr)
- German/Deutsch (de)
- Italian/Italiano (it)
- Portuguese/Português (pt)
- Russian/Русский (ru)
- Arabic/العربية (ar)
- Hindi/हिन्दी (hi)

## Installation

1. **Install VibeVoice library**:
   ```bash
   git clone https://github.com/vibevoice-community/VibeVoice.git
   cd VibeVoice
   pip install -e .
   ```

2. **Models will auto-download** from HuggingFace on first use, or you can pre-download:
   ```bash
   # For 1.5B model
   huggingface-cli download microsoft/VibeVoice-1.5B --local-dir ./models/vibevoice/1.5b

   # For 7B model
   huggingface-cli download microsoft/VibeVoice-7B --local-dir ./models/vibevoice/7b
   ```

## API Usage

### Basic Usage (OpenAI-compatible endpoint)

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-1.5b",
    "input": "Hello! This is a test of VibeVoice long-form speech synthesis.",
    "voice": "path/to/your/voice_sample.wav"
  }' \
  --output speech.wav
```

### Multi-Speaker Usage

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-7b",
    "input": "Speaker 1: Hello there! Speaker 2: Hi, how are you doing today?",
    "voice": "path/to/voice_sample.wav",
    "speaker_names": ["Alice", "Bob"]
  }' \
  --output conversation.wav
```

### With Parameters

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-1.5b",
    "input": "Your long-form text here...",
    "voice": "path/to/voice_sample.wav",
    "temperature": 0.8,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "language": "en"
  }' \
  --output speech.wav
```

### Python Client Example

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Generate speech
response = client.audio.speech.create(
    model="vibevoice-1.5b",
    voice="path/to/voice_sample.wav",
    input="This is a long-form conversational synthesis test."
)

# Save to file
response.stream_to_file("output.wav")
```

## API Endpoints

All standard Chatterbox TTS API endpoints support VibeVoice:

- `GET /v1/models` - List all models (includes vibevoice-1.5b and vibevoice-7b)
- `POST /v1/audio/speech` - Generate speech (supports both models)
- `GET /status` - Check model loading status
- `GET /config` - View current configuration

## Parameters

VibeVoice supports all standard Chatterbox TTS parameters:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `model` | string | - | - | Model ID: `vibevoice-1.5b` or `vibevoice-7b` |
| `input` | string | - | - | Text to synthesize (can be very long) |
| `voice` | string | - | - | Path to reference audio file |
| `temperature` | float | 0.05-5.0 | 0.8 | Sampling randomness |
| `exaggeration` | float | 0.25-2.0 | 0.5 | Emotion intensity |
| `cfg_weight` | float | 0.0-1.0 | 0.5 | Pace control |
| `language` | string | - | auto | Language code (en, zh, ja, etc.) |
| `speaker_names` | array | - | - | List of speaker names for multi-speaker |
| `top_p` | float | 0.0-1.0 | auto | Nucleus sampling parameter |
| `seed` | int | - | - | Random seed for reproducibility |

## Model Selection Guide

### Use VibeVoice-1.5B when:
- ✅ You need to generate very long audio (up to 90 minutes)
- ✅ You have limited VRAM (8-16GB)
- ✅ You want faster generation times
- ✅ You're creating podcasts or audiobooks

### Use VibeVoice-7B when:
- ✅ You need the highest quality output
- ✅ You have ample VRAM (16GB+)
- ✅ Quality is more important than speed
- ✅ You're working with complex multi-speaker scenarios

## Lazy Loading

VibeVoice models use **lazy loading** - they are only loaded into memory when first requested. This means:

1. No memory is used until you make your first request with a VibeVoice model
2. Models stay loaded for subsequent requests (faster response)
3. You can have multiple models loaded simultaneously
4. Models are automatically downloaded from HuggingFace on first use

## Verification

To verify the integration:

```bash
# Run verification script
python verify_vibevoice_integration.py

# Check available models via API
curl http://localhost:8000/v1/models

# Check model info
curl http://localhost:8000/status
```

## Best Practices

1. **Reference Audio**: Use clean, high-quality reference audio (30+ seconds recommended)
2. **Long-Form**: VibeVoice excels at long-form content - use it for podcasts, audiobooks, etc.
3. **Multi-Speaker**: Format text clearly with speaker labels: "Speaker 1: Hello. Speaker 2: Hi there."
4. **Context**: The models understand context, so provide coherent conversational text
5. **VRAM**: Monitor VRAM usage, especially with the 7B model

## Troubleshooting

### Model Not Loading

**Problem**: VibeVoice model fails to load

**Solutions**:
1. Ensure VibeVoice library is installed:
   ```bash
   pip show vibevoice
   ```
2. Check VRAM availability
3. Review logs for specific error messages

### Out of Memory

**Problem**: CUDA/VRAM out of memory errors

**Solutions**:
1. Use the 1.5B model instead of 7B
2. Close other applications using GPU
3. Reduce batch size or text length
4. Use CPU mode (slower but works with limited VRAM)

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'vibevoice'`

**Solution**:
```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice && pip install -e .
```

### API Not Recognizing Model

**Problem**: Model ID not recognized by API

**Solutions**:
1. Use exact model ID: `vibevoice-1.5b` or `vibevoice-7b`
2. Check spelling and capitalization
3. Verify integration with `python verify_vibevoice_integration.py`

## Technical Details

### Architecture

VibeVoice models use:
- Continuous speech tokenizers at 7.5 Hz frame rate
- Next-token diffusion with LLM for context understanding
- Zero-shot voice cloning capabilities
- Multi-speaker synthesis with up to 4 distinct voices

### File Structure

```
app/core/tts_engines/
├── vibevoice.py          # VibeVoice engine implementation
├── __init__.py           # Exports VibeVoiceEngine
└── base.py              # BaseTTSEngine interface

app/core/
└── tts_model.py         # Model registration and loading

models/vibevoice/        # Model cache directory (created on first use)
├── 1.5b/               # VibeVoice-1.5B files
└── 7b/                 # VibeVoice-7B files
```

### Model Info Response

```json
{
  "engine": "vibevoice",
  "model_version": "vibevoice-1.5b",
  "model_variant": "1.5b",
  "is_multilingual": true,
  "supported_languages": 12,
  "sample_rate": 24000,
  "context_length": 64000,
  "max_generation_minutes": 90,
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
  "model_size": "~3GB",
  "vram_required": "8GB+ (16GB recommended)"
}
```

## Resources

- **GitHub Repository**: https://github.com/vibevoice-community/VibeVoice
- **HuggingFace Models**:
  - https://huggingface.co/microsoft/VibeVoice-1.5B
  - https://huggingface.co/microsoft/VibeVoice-7B
- **Documentation**: Check the VibeVoice repository for detailed documentation
- **Community**: Discord server available for support

## License

VibeVoice is licensed under the MIT License. See the VibeVoice repository for full license details.

---

**Last Updated**: 2025-11-16
**Integration Version**: 1.0
**Compatible with**: Chatterbox TTS API v1.x
