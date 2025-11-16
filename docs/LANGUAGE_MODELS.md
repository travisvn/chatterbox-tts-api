# Language-Specific ChatterBox Models

This document describes the language-specific model support in the ChatterBox TTS API.

## Overview

The ChatterBox TTS API now supports language-specific models that are automatically downloaded from HuggingFace repositories. These models are trained specifically for individual languages and may provide better quality than the multilingual model for certain use cases.

## Available Language Models

| Language | Language Code | Model ID | HuggingFace Repository | Format | Variants |
|----------|---------------|----------|------------------------|--------|----------|
| English | `en` | `chatterbox-en` | ResembleAI/chatterbox | .pt | default |
| German | `de` | `chatterbox-de` | stlohrey/chatterbox_de | .safetensors | default |
| German | `de` | `chatterbox-de-havok2` | niobures/Chatterbox-TTS | .safetensors | havok2 |
| German | `de` | `chatterbox-de-SebastianBodza` | niobures/Chatterbox-TTS | .safetensors | SebastianBodza |
| French | `fr` | `chatterbox-fr` | Thomcles/ChatterBox-fr | .safetensors | default |
| Italian | `it` | `chatterbox-it` | niobures/Chatterbox-TTS | .pt | default |
| Russian | `ru` | `chatterbox-ru` | niobures/Chatterbox-TTS | .safetensors | default |
| Japanese | `ja` | `chatterbox-ja` | niobures/Chatterbox-TTS | .safetensors | default |
| Korean | `ko` | `chatterbox-ko` | niobures/Chatterbox-TTS | .safetensors | default |
| Norwegian | `no` | `chatterbox-no` | akhbar/chatterbox-tts-norwegian | .safetensors | default |
| Armenian | `hy` | `chatterbox-hy` | niobures/Chatterbox-TTS | .safetensors | default |
| Georgian | `ka` | `chatterbox-ka` | niobures/Chatterbox-TTS | .safetensors | default |

## Features

### Auto-Download on First Use

Language-specific models are automatically downloaded from HuggingFace when first requested. The models are cached locally in `MODEL_CACHE_DIR/language_models/` for subsequent use.

### Support for Multiple Formats

The system supports both `.pt` (PyTorch) and `.safetensors` formats. The appropriate loader is used automatically based on the model format.

### Multiple Variants

Some languages (e.g., German) have multiple model variants available. You can specify the variant when making TTS requests.

## API Endpoints

### List Available Languages

```bash
GET /languages
```

Returns a list of all supported languages with their available model variants.

**Response:**
```json
{
  "object": "list",
  "languages": [
    {
      "language_code": "de",
      "language_name": "German",
      "available_variants": ["default", "havok2", "SebastianBodza"],
      "default_model_id": "chatterbox-de"
    },
    {
      "language_code": "fr",
      "language_name": "French",
      "available_variants": ["default"],
      "default_model_id": "chatterbox-fr"
    }
    // ... more languages
  ]
}
```

### List Models for a Specific Language

```bash
GET /languages/{language_code}/models
```

Returns all available models for a specific language.

**Example:**
```bash
GET /languages/de/models
```

**Response:**
```json
{
  "object": "list",
  "models": [
    {
      "id": "chatterbox-de",
      "language": "German",
      "language_code": "de",
      "repo_id": "stlohrey/chatterbox_de",
      "format": "safetensors",
      "variant": "default",
      "is_cached": true,
      "is_loaded": false,
      "cache_size_mb": 245.6
    },
    {
      "id": "chatterbox-de-havok2",
      "language": "German",
      "language_code": "de",
      "repo_id": "niobures/Chatterbox-TTS",
      "format": "safetensors",
      "variant": "havok2",
      "is_cached": false,
      "is_loaded": false
    }
    // ... more variants
  ]
}
```

### List All Language-Specific Models

```bash
GET /language-models
```

Returns all available language-specific models across all languages.

### List All Models (Including Base Models)

```bash
GET /models
```

This endpoint now includes both base models (v1, v2, multilingual) and language-specific models.

## Using Language-Specific Models

### Basic TTS Request with Language Model

To use a language-specific model, specify the model ID in your TTS request:

```bash
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "Hallo, wie geht es dir?",
  "voice": "your-voice-id",
  "model": "chatterbox-de"
}
```

### Using a Specific Variant

For languages with multiple variants, specify the variant model ID:

```bash
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "Hallo, dies ist eine Test-Nachricht.",
  "voice": "your-voice-id",
  "model": "chatterbox-de-havok2"
}
```

### Python Example

```python
import requests

url = "http://localhost:4123/v1/audio/speech"
headers = {"Content-Type": "application/json"}

# Use German model
payload = {
    "input": "Guten Tag! Wie kann ich Ihnen helfen?",
    "voice": "alloy",
    "model": "chatterbox-de"
}

response = requests.post(url, json=payload, headers=headers)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### cURL Example

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bonjour! Comment allez-vous?",
    "voice": "alloy",
    "model": "chatterbox-fr"
  }' \
  --output french_speech.mp3
```

## Model Selection Strategy

### When to Use Language-Specific Models

- **Better Quality**: Language-specific models are trained on a single language and may provide better quality for that language compared to the multilingual model.
- **Specialized Variants**: Some languages have multiple variants optimized for different use cases.
- **Resource Constraints**: Language-specific models may have different resource requirements than the multilingual model.

### When to Use Multilingual Models

- **Multiple Languages**: If you need to generate speech in multiple languages, the multilingual model supports 23 languages without needing to switch models.
- **Consistent Voice**: The multilingual model can maintain consistent voice characteristics across languages.

## Model Caching

### Cache Location

Models are cached in: `MODEL_CACHE_DIR/language_models/{model_id}/`

By default: `./models/language_models/{model_id}/`

### Cache Management

- Models are automatically downloaded on first use
- Downloaded models are cached locally for subsequent requests
- Check cache status via the `/language-models` endpoint
- Each model's cache size is reported in MB

### Manual Cache Management

To pre-download a model, simply make a TTS request with that model. The download will happen automatically.

To clear the cache, delete the model directory:
```bash
rm -rf ./models/language_models/chatterbox-de
```

## Technical Details

### Model Loading Process

1. **Request Received**: TTS request specifies a language-specific model
2. **Model Check**: System checks if model is already loaded in memory
3. **Cache Check**: If not loaded, system checks local cache
4. **Auto-Download**: If not cached, model is downloaded from HuggingFace
5. **Model Load**: Model is loaded into memory on the configured device
6. **Generation**: TTS generation proceeds with the loaded model

### Supported Formats

- **PyTorch (.pt)**: Traditional PyTorch model format
- **SafeTensors (.safetensors)**: Safer, faster format for model weights

The system automatically detects and uses the appropriate loader based on the model format.

### Memory Management

- Models are loaded on-demand (lazy loading)
- Multiple models can be loaded simultaneously
- Models remain in memory until the service is restarted
- Memory is automatically managed based on your device capabilities

## Configuration

### Environment Variables

See `.env.example` for configuration options:

```bash
# Model cache directory
MODEL_CACHE_DIR=./models

# Device for model inference
DEVICE=auto  # auto/cuda/mps/cpu
```

### HuggingFace Configuration

Optional HuggingFace settings:

```bash
# HuggingFace cache directory
HF_HOME=/cache/huggingface

# Disable HuggingFace telemetry
HF_HUB_DISABLE_TELEMETRY=true
```

## Troubleshooting

### Model Download Fails

If a model fails to download:
1. Check your internet connection
2. Verify the HuggingFace repository is accessible
3. Check disk space in MODEL_CACHE_DIR
4. Review server logs for detailed error messages

### Model Load Fails

If a model fails to load:
1. Verify the model was downloaded completely
2. Check available memory/VRAM
3. Try using CPU device if GPU fails
4. Clear cache and re-download

### Slow Downloads

Large models may take time to download:
- German models: ~200-300 MB
- Other models: varies by language
- First request will be slower due to download
- Subsequent requests use cached model

## Performance Considerations

### First Request

The first request to a language-specific model will be slower due to:
1. Model download from HuggingFace (one-time)
2. Model loading into memory

### Subsequent Requests

After initial load:
- Models remain in memory
- Requests are processed at normal speed
- No additional download needed

### Resource Usage

- Each loaded model consumes memory/VRAM
- Consider available resources when loading multiple models
- Use CPU device for lower memory usage (slower generation)

## Credits

Thanks to the community contributors who created these language-specific models:
- stlohrey (German)
- Thomcles (French)
- niobures (Multiple languages)
- akhbar (Norwegian)
- ResembleAI (Original ChatterBox)
