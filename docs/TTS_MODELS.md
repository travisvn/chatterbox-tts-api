# TTS Models Guide

This guide covers all available TTS models in the Chatterbox TTS API, including the newly added IndexTTS-2 and Higgs Audio V2 engines.

## Available Models

### Chatterbox Models (Default)

| Model ID | Type | Languages | Description |
|----------|------|-----------|-------------|
| `chatterbox-multilingual-v2` | Multilingual | 23 | Official ResembleAI package with v2 updates (default) |
| `chatterbox-multilingual-v1` | Multilingual | 23 | Experimental fork with additional features |
| `chatterbox-v2` | Standard | English | Official ResembleAI package, English-only |
| `chatterbox-v1` | Standard | English | Experimental fork, English-only |

**Features:**
- ✅ Voice cloning from audio samples
- ✅ Multilingual support (v2: 23 languages)
- ✅ Emotion/exaggeration control
- ✅ OpenAI-compatible API
- ✅ Production-ready, well-tested
- ✅ MIT licensed

**Requirements:**
- Model Size: ~1GB
- VRAM: 4-8GB recommended
- Installation: Included by default

### IndexTTS-2 (New!)

| Model ID | Type | Languages | Description |
|----------|------|-----------|-------------|
| `indextts-2` | Multilingual | 4+ | Industrial-level controllable zero-shot TTS with emotion control |

**Features:**
- ✅ Zero-shot voice cloning
- ✅ **Advanced emotion control** (8 emotion vectors)
- ✅ Precise synthesis duration control
- ✅ Highly expressive speech synthesis
- ✅ Apache-2.0 licensed

**Requirements:**
- Model Size: ~1-2GB
- VRAM: 8GB+ recommended
- Installation: See [Installation](#installing-indextts-2)

**Supported Languages:**
- English (en)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)

### Higgs Audio V2 (New!)

| Model ID | Type | Languages | Description |
|----------|------|-----------|-------------|
| `higgs-audio-v2` | Multilingual | 10+ | Text-audio foundation model from Boson AI |

**Features:**
- ✅ Neural voice cloning from 30+ second reference audio
- ✅ **Multi-speaker conversations**
- ✅ **Long-form generation** (up to 90 minutes)
- ✅ Exceptional audio fidelity
- ⚠️ **Requires significant GPU VRAM (24GB+)**

**Requirements:**
- Model Size: ~3-4GB
- VRAM: **24GB+ strongly recommended**
- Installation: See [Installation](#installing-higgs-audio-v2)

**Supported Languages:**
- English (en), Chinese (zh), Japanese (ja), Korean (ko)
- Spanish (es), French (fr), German (de), Italian (it)
- Portuguese (pt), Russian (ru)

## Installation

### Chatterbox Models

Chatterbox models are included by default and will be downloaded automatically on first use.

```bash
# Already installed with base requirements
pip install chatterbox-tts==0.1.4
```

### Installing IndexTTS-2

#### Option 1: Using pip (Recommended)

```bash
# Install IndexTTS-2
pip install indextts

# Download models (first time only)
pip install huggingface-hub
hf download IndexTeam/IndexTTS-2 --local-dir=models/indextts/checkpoints
```

#### Option 2: Automatic Download

Simply set `DEFAULT_TTS_ENGINE=indextts` in your `.env` file. The models will download automatically on first request.

```env
DEFAULT_TTS_ENGINE=indextts
```

### Installing Higgs Audio V2

Higgs Audio requires installation from source:

```bash
# Clone the repository
git clone https://github.com/boson-ai/higgs-audio
cd higgs-audio

# Install
pip install -e .

# Return to project directory
cd /path/to/chatterbox-tts-api
```

Then set in `.env`:

```env
DEFAULT_TTS_ENGINE=higgs
```

**Note:** Higgs Audio models (~3-4GB) will download automatically on first use.

## Configuration

### Setting the Default Engine

Edit your `.env` file:

```env
# Choose your default engine
DEFAULT_TTS_ENGINE=chatterbox  # Default
# DEFAULT_TTS_ENGINE=indextts  # IndexTTS-2
# DEFAULT_TTS_ENGINE=higgs     # Higgs Audio V2

# For Chatterbox, also configure:
USE_MULTILINGUAL_MODEL=true
DEFAULT_MODEL_VERSION=v2
```

### Model Selection Per Request

You can override the default model in each API request:

```bash
# Use IndexTTS-2
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "indextts-2"}' \
  --output speech.wav

# Use Higgs Audio V2
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "higgs-audio-v2"}' \
  --output speech.wav

# Use Chatterbox multilingual v2
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "chatterbox-multilingual-v2"}' \
  --output speech.wav
```

## API Usage Examples

### Python Example

```python
import requests

# Generate speech with IndexTTS-2
response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "This uses IndexTTS-2 with emotion control!",
        "model": "indextts-2",
        "exaggeration": 0.8,  # Higher emotion intensity
        "temperature": 0.9
    }
)

with open("indextts_output.wav", "wb") as f:
    f.write(response.content)

# Generate speech with Higgs Audio V2
response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "This uses Higgs Audio V2 for high-quality synthesis!",
        "model": "higgs-audio-v2",
        "voice": "my-custom-voice",  # Voice from library
        "exaggeration": 0.7
    }
)

with open("higgs_output.wav", "wb") as f:
    f.write(response.content)
```

### JavaScript Example

```javascript
async function generateSpeech(text, model) {
  const response = await fetch('http://localhost:4123/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: text,
      model: model,
      exaggeration: 0.7,
      temperature: 0.8
    })
  });

  const blob = await response.blob();
  return blob;
}

// Use IndexTTS-2
const audio1 = await generateSpeech("Hello from IndexTTS-2!", "indextts-2");

// Use Higgs Audio V2
const audio2 = await generateSpeech("Hello from Higgs Audio!", "higgs-audio-v2");
```

## Model Comparison

| Feature | Chatterbox | IndexTTS-2 | Higgs Audio V2 |
|---------|------------|------------|----------------|
| Voice Cloning | ✅ | ✅ | ✅ |
| Multilingual | ✅ (23 langs) | ✅ (4 langs) | ✅ (10+ langs) |
| Emotion Control | Basic | ✅ Advanced | ✅ |
| Multi-Speaker | ❌ | ❌ | ✅ |
| Long Form (90min+) | ❌ | ❌ | ✅ |
| Model Size | ~1GB | ~1-2GB | ~3-4GB |
| VRAM Required | 4-8GB | 8GB+ | 24GB+ |
| License | MIT | Apache-2.0 | Custom |
| OpenAI Compatible | ✅ | ✅ | ✅ |

## Lazy Loading

All models support **lazy loading**:

- Models are downloaded only when first requested
- Models stay in memory after loading for fast access
- You can load multiple models simultaneously
- Check loaded models with `GET /models`

```bash
# Check which models are loaded
curl http://localhost:4123/models
```

## Performance Tips

### For Limited VRAM (< 8GB)

Use Chatterbox models:

```env
DEFAULT_TTS_ENGINE=chatterbox
USE_MULTILINGUAL_MODEL=false  # English-only uses less VRAM
```

### For Standard VRAM (8-16GB)

Use IndexTTS-2 for emotion control:

```env
DEFAULT_TTS_ENGINE=indextts
```

### For High VRAM (24GB+)

Use Higgs Audio V2 for best quality and multi-speaker:

```env
DEFAULT_TTS_ENGINE=higgs
```

## Troubleshooting

### IndexTTS-2 Not Found

```bash
# Install IndexTTS-2
pip install indextts

# Download models manually
pip install huggingface-hub
hf download IndexTeam/IndexTTS-2 --local-dir=models/indextts/checkpoints
```

### Higgs Audio Import Error

```bash
# Clone and install from source
git clone https://github.com/boson-ai/higgs-audio
cd higgs-audio
pip install -e .
```

### Out of Memory Error

- Use a smaller model (Chatterbox instead of Higgs Audio)
- Reduce batch size: `MAX_CHUNK_LENGTH=200`
- Enable CPU mode: `DEVICE=cpu` (slower but uses less VRAM)
- Close other GPU applications

### Model Download Fails

```bash
# Set HuggingFace token if needed
export HF_TOKEN=your_token_here

# Or download manually
git lfs install
git clone https://huggingface.co/IndexTeam/IndexTTS-2 models/indextts/checkpoints
```

## Advanced Configuration

### Memory Management

```env
# Cleanup memory more frequently with multiple models
MEMORY_CLEANUP_INTERVAL=3
CUDA_CACHE_CLEAR_INTERVAL=2
```

### Model-Specific Cache Directories

Models are cached in separate directories:

```
models/
├── chatterbox/       # Chatterbox models
├── indextts/         # IndexTTS-2 models
│   └── checkpoints/
└── higgs_audio/      # Higgs Audio V2 models
```

## FAQ

**Q: Can I use multiple models at once?**

A: Yes! Models are loaded lazily. You can switch between models using the `model` parameter in each request.

**Q: Which model should I use?**

A:
- **General use**: Chatterbox multilingual v2 (default, best balance)
- **Emotion control**: IndexTTS-2
- **Multi-speaker/long-form**: Higgs Audio V2 (requires 24GB+ VRAM)

**Q: Do models download automatically?**

A: Yes, models download on first use. IndexTTS-2 and Chatterbox download automatically. Higgs Audio requires manual installation from source.

**Q: How much disk space do I need?**

A:
- Chatterbox: ~1GB
- IndexTTS-2: ~1-2GB
- Higgs Audio V2: ~3-4GB
- **Total (all models)**: ~5-7GB

**Q: Can I use CPU instead of GPU?**

A: Yes, set `DEVICE=cpu` in `.env`. However, generation will be significantly slower.

## Support

For issues or questions:
- GitHub Issues: https://github.com/travisvn/chatterbox-tts-api/issues
- Discord: http://chatterboxtts.com/discord

## References

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [IndexTTS-2](https://github.com/index-tts/index-tts)
- [Higgs Audio](https://github.com/boson-ai/higgs-audio)
