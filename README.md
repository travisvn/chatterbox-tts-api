<p align="center">
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOU5bjcd6giJaPhnlpTZysr24u6k9WGqwIjNgQo" alt="Chatterbox API TTS header">
</p>

# Chatterbox TTS API

<p align="center">
	<a href="https://github.com/travisvn/chatterbox-tts-api">
		<img src="https://img.shields.io/github/stars/travisvn/chatterbox-tts-api?style=social" alt="GitHub stars"></a>
	<a href="https://github.com/travisvn/chatterbox-tts-api">
		<img alt="GitHub forks" src="https://img.shields.io/github/forks/travisvn/chatterbox-tts-api"></a>
	<a href="https://github.com/travisvn/chatterbox-tts-api/issues">
	  <img src="https://img.shields.io/github/issues/travisvn/chatterbox-tts-api" alt="GitHub issues"></a>
	<img src="https://img.shields.io/github/last-commit/travisvn/chatterbox-tts-api?color=red" alt="GitHub last commit">
  <a href="http://chatterboxtts.com/discord">
    <img src="https://img.shields.io/badge/Discord-Voice_AI_%26_TTS_Tools-blue?logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

**FastAPI**-powered REST API for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox), providing OpenAI-compatible text-to-speech endpoints with voice cloning capabilities and additional features on top of the `chatterbox-tts` base package.

## Features

🚀 **OpenAI-Compatible API** - Drop-in replacement for OpenAI's TTS API  
⚡ **FastAPI Performance** - High-performance async API with automatic documentation  
🌍 **Multilingual Support** - Generate speech in 22 languages with language-aware voice cloning  
🎨 **React Frontend** - Includes an optional, ready-to-use web interface  
🎭 **Voice Cloning** - Use your own voice samples for personalized speech  
🎤 **Voice Library Management** - Upload, manage, and use custom voices by name  
📝 **Smart Text Processing** - Automatic chunking for long texts  
📊 **Real-time Status** - Monitor TTS progress, statistics, and request history  
🐳 **Docker Ready** - Full containerization with persistent voice storage  
⚙️ **Configurable** - Extensive environment variable configuration  
🎛️ **Parameter Control** - Real-time adjustment of speech characteristics  
📚 **Auto Documentation** - Interactive API docs at `/docs` and `/redoc`  
🔧 **Type Safety** - Full Pydantic validation for requests and responses  
🧠 **Memory Management** - Advanced memory monitoring and automatic cleanup

> [!NOTE]
> _Support for Chatterbox Turbo coming soon_


> [!IMPORTANT]
> `resemble-ai/chatterbox` is currently broken for non-CUDA setups (see [chatterbox issues](https://github.com/resemble-ai/chatterbox/issues))
> 
> Revert to non-multilingual by using the `stable` branch of this repo
>
> [View more instructions](#issues-with-multilingual)

## ⚡️ Quick Start

```bash
git clone https://github.com/travisvn/chatterbox-tts-api
cd chatterbox-tts-api
uv sync
uv run main.py
```

> [!TIP]  
> [uv](https://docs.astral.sh/uv/) installed with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Local Installation with Python 🐍

#### Option A: Using uv (Recommended - Faster & Better Dependencies)

```bash
# Clone the repository
git clone https://github.com/travisvn/chatterbox-tts-api
cd chatterbox-tts-api

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv (automatically creates venv)
uv sync

# Copy and customize environment variables
cp .env.example .env

# Start the API with FastAPI
uv run uvicorn app.main:app --host 0.0.0.0 --port 4123
# Or use the main script
uv run main.py
```

> 💡 **Why uv?** Users report better compatibility with `chatterbox-tts`, 25-40% faster installs, and superior dependency resolution. [See migration guide →](docs/UV_MIGRATION.md)

#### Option B: Using pip (Traditional)

```bash
# Clone the repository
git clone https://github.com/travisvn/chatterbox-tts-api
cd chatterbox-tts-api

# Setup environment — using Python 3.11
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and customize environment variables
cp .env.example .env

# Add your voice sample (or use the provided one)
# cp your-voice.mp3 voice-sample.mp3

# Start the API with FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 4123
# Or use the main script
python main.py
```

> Ran into issues? Check the [troubleshooting section](https://github.com/travisvn/chatterbox-tts-api?tab=readme-ov-file#common-issues)

### 🐳 Docker (Recommended)

```bash
# Clone and start with Docker Compose
git clone https://github.com/travisvn/chatterbox-tts-api
cd chatterbox-tts-api

# Use Docker-optimized environment variables
cp .env.example.docker .env  # Docker-specific paths, ready to use
# Or: cp .env.example .env    # Local development paths, needs customization

# Choose your deployment method:

# API Only (default)
docker compose -f docker/docker-compose.yml up -d             # Standard (pip-based)
docker compose -f docker/docker-compose.uv.yml up -d          # uv-optimized (faster builds)
docker compose -f docker/docker-compose.gpu.yml up -d         # Standard + GPU
docker compose -f docker/docker-compose.uv.gpu.yml up -d      # uv + GPU (recommended for GPU users)
docker compose -f docker/docker-compose.cpu.yml up -d         # CPU-only
docker compose -f docker/docker-compose.blackwell.yml up -d   # Blackwell (50XX) NVIDIA GPUs

# API + Frontend (add --profile frontend to any of the above)
docker compose -f docker/docker-compose.yml --profile frontend up -d             # Standard + Frontend
docker compose -f docker/docker-compose.gpu.yml --profile frontend up -d         # GPU + Frontend
docker compose -f docker/docker-compose.uv.gpu.yml --profile frontend up -d      # uv + GPU + Frontend
docker compose -f docker/docker-compose.blackwell.yml --profile frontend up -d   # (Blackwell) uv + GPU + Frontend

# Watch the logs as it initializes (the first use of TTS takes the longest)
docker logs chatterbox-tts-api -f

# Test the API
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Chatterbox TTS!"}' \
  --output test.wav
```

<details>
<summary><strong>🚀 Running with the Web UI (Full Stack)</strong></summary>

This project includes an optional React-based web UI. Use Docker Compose profiles to easily opt in or out of the frontend:

### With Docker Compose Profiles

```bash
# API only (default behavior)
docker compose -f docker/docker-compose.yml up -d

# API + Frontend + Web UI (with --profile frontend)
docker compose -f docker/docker-compose.yml --profile frontend up -d

# Or use the convenient helper script for fullstack:
python start.py fullstack

# Same pattern works with all deployment variants:
docker compose -f docker/docker-compose.gpu.yml --profile frontend up -d    # GPU + Frontend
docker compose -f docker/docker-compose.uv.yml --profile frontend up -d     # uv + Frontend
docker compose -f docker/docker-compose.cpu.yml --profile frontend up -d    # CPU + Frontend
```

### Local Development

For local development, you can run the API and frontend separately:

```bash
# Start the API first (follow earlier instructions)
# Then run the frontend:
cd frontend && npm install && npm run dev
```

Click the link provided from Vite to access the web UI.

### Build for Production

Build the frontend for production deployment:

```bash
cd frontend && npm install && npm run build
```

You can then access it directly from your local file system at `/dist/index.html`.

### Port Configuration

- **API Only**: Accessible at `http://localhost:4123` (direct API access)
- **With Frontend**: Web UI at `http://localhost:4321`, API requests routed via proxy

The frontend uses a reverse proxy to route requests, so when running with `--profile frontend`, the web interface will be available at `http://localhost:4321` while the API runs behind the proxy.

</details>

## Screenshots of Frontend (Web UI)

<div align="center">
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOUS62gM9PGyDAvTxnjVKQO0Zz5uI6Jg4UodHEa" alt="Chatterbox TTS API - Frontend - Dark Mode" width="33%" />
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOUXYXF1ekWhMaPnZ3rSTRIEkDzvKwGU05qjAol" alt="Chatterbox TTS API - Frontend - Light Mode" width="33%" />
</div>

<div align="center">
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOUt4kJ0goXPb09QmDchfSoNxgB3KLETFyvnsU5" alt="Chatterbox TTS API - Frontend Processing - Dark Mode" width="33%" />
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOU0v7EUEwi1efdOvQm6TrWKoPuX7xEl4pc8RVw" alt="Chatterbox TTS API - Frontend Processing - Light Mode" width="33%" />
</div>

> 🖼️ View screenshot of full frontend web UI — [light mode](https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOUoONOy6UZv2m8CUjqGrBbDy4aXzNV9Rl1ZAgQ) / [dark mode](https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOU7RmQRTFVcR8ntzKQs0IxJ6ibFrq2hjCSadUG)

## API Usage

### Basic Text-to-Speech (Default Voice)

This endpoint works for both the API-only and full-stack setups.

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Your text here"}' \
  --output speech.wav
```

### Using Custom Parameters (JSON)

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Dramatic speech!", "exaggeration": 1.2, "cfg_weight": 0.3, "temperature": 0.9}' \
  --output dramatic.wav
```

### Custom Voice Upload

Upload your own voice sample for personalized speech:

```bash
curl -X POST http://localhost:4123/v1/audio/speech/upload \
  -F "input=Hello with my custom voice!" \
  -F "exaggeration=0.8" \
  -F "voice_file=@my_voice.mp3" \
  --output custom_voice_speech.wav
```

### With Custom Parameters and Voice Upload

```bash
curl -X POST http://localhost:4123/v1/audio/speech/upload \
  -F "input=Dramatic speech!" \
  -F "exaggeration=1.2" \
  -F "cfg_weight=0.3" \
  -F "temperature=0.9" \
  -F "voice_file=@dramatic_voice.wav" \
  --output dramatic.wav
```

### Voice Library Management

Store and manage custom voices by name for reuse across requests:

```bash
# Upload a voice to the library
curl -X POST http://localhost:4123/voices \
  -F "voice_file=@my_voice.wav" \
  -F "voice_name=my-custom-voice"

# Upload a voice with language (multilingual support)
curl -X POST http://localhost:4123/voices \
  -F "voice_file=@french_voice.wav" \
  -F "voice_name=french-speaker" \
  -F "language=fr"

# Use the voice by name in speech generation
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello with my custom voice!", "voice": "my-custom-voice"}' \
  --output custom_voice_output.wav

# Generate French speech (language auto-detected from voice)
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bonjour, comment allez-vous?", "voice": "french-speaker"}' \
  --output french_speech.wav

# List all available voices (includes language metadata)
curl http://localhost:4123/voices

# Get supported languages
curl http://localhost:4123/languages
```

**🔧 [Complete Voice Library Documentation →](docs/VOICE_LIBRARY_MANAGEMENT.md)**

## 🌍 Multilingual Support

Generate speech in **22 languages** with language-aware voice cloning and automatic language detection.

### Supported Languages

Arabic (ar) • Danish (da) • German (de) • Greek (el) • **English (en)** • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr)

### Quick Start

```bash
# Get supported languages
curl http://localhost:4123/languages

# Upload voice with language
curl -X POST http://localhost:4123/voices \
  -F "voice_name=spanish_speaker" \
  -F "language=es" \
  -F "voice_file=@spanish_voice.wav"

# Generate multilingual speech
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "¡Hola! ¿Cómo estás hoy?", "voice": "spanish_speaker"}' \
  --output spanish_speech.wav
```

### Key Features

- 🎯 **Language Auto-Detection** - Voices store language metadata, automatically used in generation
- 🌐 **No API Changes** - Maintains OpenAI compatibility, language determined from voice metadata
- 🔄 **Configurable** - Enable/disable with `USE_MULTILINGUAL_MODEL` environment variable
- 📚 **Voice Library Integration** - Language badges and filtering in web UI
- 🧠 **Smart Fallback** - Defaults to English for backward compatibility

**📚 [Complete Multilingual Documentation →](docs/MULTILINGUAL.md)**

## 🎵 Real-time Audio Streaming

The API supports multiple streaming formats for lower latency and better user experience:

- **Raw Audio Streaming**: Traditional audio chunks (WAV format)
- **Server-Side Events (SSE)**: OpenAI-compatible format with base64-encoded audio chunks

### Quick Start

```bash
# Basic audio streaming
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "This streams in real-time!"}' \
  --output streaming.wav

# SSE streaming (OpenAI compatible)
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"input": "This streams as Server-Side Events!", "stream_format": "sse"}' \
  --no-buffer

# Real-time playback
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Play as it generates!"}' \
  | ffplay -f wav -i pipe:0 -autoexit -nodisp
```

### 🚀 **[Complete Streaming Documentation →](docs/STREAMING_API.md)**

For comprehensive streaming features including:

- **Advanced chunking strategies** (sentence, paragraph, word, fixed)
- **Quality presets** (fast, balanced, high)
- **Configurable parameters** and performance tuning
- **Real-time progress monitoring**
- **Python, JavaScript, and cURL examples**
- **Integration patterns** for different use cases

**Key Benefits:**

- ⚡ **Lower latency** - Start hearing audio in 1-2 seconds
- 🎯 **Better UX** - No waiting for complete generation
- 💾 **Memory efficient** - Process chunks individually
- 🎛️ **Configurable** - Choose speed vs quality trade-offs

<details>
<summary><strong>🐍 Python Examples</strong></summary>

### Default Voice (JSON)

```python
import requests

response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "Hello world!",
        "exaggeration": 0.8
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Upload Voice with Language (Multilingual)

```python
import requests

# Upload a multilingual voice
with open("german_voice.wav", "rb") as voice_file:
    response = requests.post(
        "http://localhost:4123/voices",
        data={
            "voice_name": "german_speaker",
            "language": "de"
        },
        files={
            "voice_file": ("german_voice.wav", voice_file, "audio/wav")
        }
    )

print(f"Upload status: {response.status_code}")

# Generate German speech
response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "Guten Tag! Wie geht es Ihnen?",
        "voice": "german_speaker",
        "exaggeration": 0.8
    }
)

with open("german_output.wav", "wb") as f:
    f.write(response.content)
```

### Upload Endpoint (Default Voice)

```python
import requests

response = requests.post(
    "http://localhost:4123/v1/audio/speech/upload",
    data={
        "input": "Hello world!",
        "exaggeration": 0.8
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Custom Voice Upload

```python
import requests

with open("my_voice.mp3", "rb") as voice_file:
    response = requests.post(
        "http://localhost:4123/v1/audio/speech/upload",
        data={
            "input": "Hello with my custom voice!",
            "exaggeration": 0.8,
            "temperature": 1.0
        },
        files={
            "voice_file": ("my_voice.mp3", voice_file, "audio/mpeg")
        }
    )

with open("custom_output.wav", "wb") as f:
    f.write(response.content)
```

### Basic Streaming Example

```python
import requests

# Stream audio generation in real-time
response = requests.post(
    "http://localhost:4123/v1/audio/speech/stream",
    json={
        "input": "This will stream as it's generated!",
        "exaggeration": 0.8
    },
    stream=True  # Enable streaming mode
)

with open("streaming_output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            print(f"Received chunk: {len(chunk)} bytes")
```

### SSE Streaming Example (OpenAI Compatible)

```python
import requests
import json
import base64

# Stream audio using Server-Side Events format
response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "This streams as Server-Side Events!",
        "stream_format": "sse",
        "exaggeration": 0.8
    },
    stream=True,
    headers={'Accept': 'text/event-stream'}
)

audio_chunks = []

for line in response.iter_lines(decode_unicode=True):
    if line.startswith('data: '):
        event_data = line[6:]  # Remove 'data: ' prefix

        try:
            event = json.loads(event_data)

            if event.get('type') == 'speech.audio.delta':
                # Decode base64 audio chunk
                audio_data = base64.b64decode(event['audio'])
                audio_chunks.append(audio_data)
                print(f"Received audio chunk: {len(audio_data)} bytes")

            elif event.get('type') == 'speech.audio.done':
                usage = event.get('usage', {})
                print(f"Complete! Tokens: {usage.get('total_tokens', 0)}")
                break
        except:
            continue

print(f"Received {len(audio_chunks)} audio chunks")
```

**📚 [Complete Streaming Examples & Documentation →](docs/STREAMING_API.md)**

Including real-time playback, progress monitoring, custom voice uploads, and advanced integration patterns.

</details>

### Voice File Requirements

**Supported Formats:**

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)

**Requirements:**

- Maximum file size: 10MB
- Recommended duration: 10-30 seconds of clear speech
- Avoid background noise for best results
- Higher quality audio produces better voice cloning

## 🎛️ Configuration

The project provides two environment example files:

- **`.env.example`** - For local development (uses `./models`, `./voice-sample.mp3`)
- **`.env.example.docker`** - For Docker deployment (uses `/cache`, `/app/voice-sample.mp3`)

Choose the appropriate one for your setup:

```bash
# For local development
cp .env.example .env

# For Docker deployment
cp .env.example.docker .env
```

Key environment variables (see the example files for full list):

| Variable                 | Default              | Description                    |
| ------------------------ | -------------------- | ------------------------------ |
| `PORT`                   | `4123`               | API server port                |
| `USE_MULTILINGUAL_MODEL` | `true`               | Enable 23-language support     |
| `EXAGGERATION`           | `0.5`                | Emotion intensity (0.25-2.0)   |
| `CFG_WEIGHT`             | `0.5`                | Pace control (0.0-1.0)         |
| `TEMPERATURE`            | `0.8`                | Sampling randomness (0.05-5.0) |
| `VOICE_SAMPLE_PATH`      | `./voice-sample.mp3` | Voice sample for cloning       |
| `DEVICE`                 | `auto`               | Device (auto/cuda/mps/cpu)     |

<details>
<summary><strong>🎭 Voice Cloning</strong></summary>

Replace the default voice sample:

```bash
# Replace the default voice sample
cp your-voice.mp3 voice-sample.mp3

# Or set a custom path
echo "VOICE_SAMPLE_PATH=/path/to/your/voice.mp3" >> .env
```

For best results:

- Use 10-30 seconds of clear speech
- Avoid background noise
- Prefer WAV or high-quality MP3

</details>

<details>
<summary><strong>🐳 Docker Deployment</strong></summary>

### Development

```bash
docker compose -f docker/docker-compose.yml up
```

### Production

```bash
# Create production environment
cp .env.example.docker .env
nano .env  # Set production values

# Deploy
docker compose -f docker/docker-compose.yml up -d
```

### With GPU Support

```bash
# Use GPU-enabled compose file
# Ensure NVIDIA Container Toolkit is installed
docker compose -f docker/docker-compose.gpu.yml up -d
```

</details>

<details>
<summary><strong>📚 API Reference</strong></summary>

## API Endpoints

| Endpoint                      | Method | Description                                                         |
| ----------------------------- | ------ | ------------------------------------------------------------------- |
| `/audio/speech`               | POST   | Generate speech from text (complete)                                |
| `/audio/speech/upload`        | POST   | Generate speech with voice upload                                   |
| `/audio/speech/stream`        | POST   | **Stream** speech generation ([docs](docs/STREAMING_API.md))        |
| `/audio/speech/stream/upload` | POST   | **Stream** speech with voice upload ([docs](docs/STREAMING_API.md)) |
| `/voices`                     | GET    | List voices in library (with language metadata)                     |
| `/voices`                     | POST   | Upload voice to library (with language support)                     |
| `/languages`                  | GET    | **Get supported languages** ([docs](docs/MULTILINGUAL.md))          |
| `/health`                     | GET    | Health check and status                                             |
| `/config`                     | GET    | Current configuration                                               |
| `/v1/models`                  | GET    | Available models (OpenAI compat)                                    |
| `/status`                     | GET    | TTS processing status & progress                                    |
| `/status/progress`            | GET    | Real-time progress (lightweight)                                    |
| `/status/statistics`          | GET    | Processing statistics                                               |
| `/status/history`             | GET    | Recent request history                                              |
| `/info`                       | GET    | Complete API information                                            |
| `/docs`                       | GET    | Interactive API documentation                                       |
| `/redoc`                      | GET    | Alternative API documentation                                       |

## Parameters Reference

### Speech Generation Parameters

**Exaggeration (0.25-2.0)**

- `0.3-0.4`: Professional, neutral
- `0.5`: Default balanced
- `0.7-0.8`: More expressive
- `1.0+`: Very dramatic

**CFG Weight (0.0-1.0)**

- `0.2-0.3`: Faster speech
- `0.5`: Default pace
- `0.7-0.8`: Slower, deliberate

**Temperature (0.05-5.0)**

- `0.4-0.6`: More consistent
- `0.8`: Default balance
- `1.0+`: More creative/random

**Stream Format**

- `audio`: Raw audio streaming (default)
- `sse`: Server-Side Events with base64-encoded audio chunks (OpenAI compatible)

</details>

<details>
<summary><strong>🧠 Memory Management</strong></summary>

The API includes advanced memory management to prevent memory leaks and optimize performance:

### Memory Management Features

- **Automatic Cleanup**: Periodic garbage collection and tensor cleanup
- **CUDA Memory Management**: Automatic GPU cache clearing
- **Memory Monitoring**: Real-time memory usage tracking
- **Manual Controls**: API endpoints for manual cleanup operations

### Memory Configuration

| Variable                    | Default | Description                       |
| --------------------------- | ------- | --------------------------------- |
| `MEMORY_CLEANUP_INTERVAL`   | `5`     | Cleanup memory every N requests   |
| `CUDA_CACHE_CLEAR_INTERVAL` | `3`     | Clear CUDA cache every N requests |
| `ENABLE_MEMORY_MONITORING`  | `true`  | Enable detailed memory logging    |

### Memory Monitoring Endpoints

```bash
# Get memory status
curl http://localhost:4123/memory

# Trigger manual cleanup
curl "http://localhost:4123/memory?cleanup=true&force_cuda_clear=true"

# Reset memory tracking (with confirmation)
curl -X POST "http://localhost:4123/memory/reset?confirm=true"
```

### Real-time Status Tracking

Monitor TTS processing in real-time:

```bash
# Check current processing status
curl "http://localhost:4123/v1/status/progress"

# Get detailed status with memory and stats
curl "http://localhost:4123/v1/status?include_memory=true&include_stats=true"

# View processing statistics
curl "http://localhost:4123/v1/status/statistics"

# Check request history
curl "http://localhost:4123/v1/status/history?limit=5"

# Get comprehensive API information
curl "http://localhost:4123/info"
```

**Status Response Example:**

```json
{
  "is_processing": true,
  "status": "generating_audio",
  "current_step": "Generating audio for chunk 2/4",
  "current_chunk": 2,
  "total_chunks": 4,
  "progress_percentage": 50.0,
  "duration_seconds": 2.5,
  "text_preview": "Your text being processed..."
}
```

See [Status API Documentation](docs/STATUS_API.md) for complete details.

### Memory Testing

Run the memory management test suite:

```bash
# Test memory patterns and cleanup
python tests/test_memory.py  # or: uv run tests/test_memory.py

# Monitor memory during testing
watch -n 1 'curl -s http://localhost:4123/memory | jq .memory_info'
```

### Memory Optimization Tips

**For High-Volume Production:**

```env
MEMORY_CLEANUP_INTERVAL=3
CUDA_CACHE_CLEAR_INTERVAL=2
ENABLE_MEMORY_MONITORING=false  # Reduce logging overhead
MAX_CHUNK_LENGTH=200             # Smaller chunks for less memory usage
```

**For Development/Debugging:**

```env
MEMORY_CLEANUP_INTERVAL=1
CUDA_CACHE_CLEAR_INTERVAL=1
ENABLE_MEMORY_MONITORING=true
```

**Memory Leak Prevention:**

- Tensors are automatically moved to CPU before deletion
- Gradient tracking is disabled during inference
- Audio chunks are cleaned up after concatenation
- CUDA cache is periodically cleared
- Python garbage collection is triggered regularly

</details>

<details>
<summary><strong>🧪 Testing</strong></summary>

Run the test script to verify the API functionality:

```bash
python tests/test_api.py
```

The test script will:

- Test health check endpoint
- Test models endpoint
- Test API documentation endpoints (new!)
- Generate speech for various text lengths
- Test custom parameter validation
- Test error handling with validation
- Save generated audio files as `test_output_*.wav`

</details>

<details>
<summary><strong>⚡ Performance</strong></summary>

**FastAPI Benefits:**

- **Async support**: Better concurrent request handling
- **Faster serialization**: JSON responses ~25% faster than Flask
- **Type validation**: Pydantic models prevent invalid requests
- **Auto documentation**: No manual API doc maintenance

**Hardware Recommendations:**

- **CPU**: Works but slower, reduce chunk size for better memory usage
- **GPU**: Recommended for production, significantly faster
- **Memory**: 4GB minimum, 8GB+ recommended
- **Concurrency**: Async support allows better multi-request handling

</details>

<details>
<summary><strong>🔧 Troubleshooting</strong></summary>

### Common Issues

**CUDA/CPU Compatibility Error**

```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

This happens because `chatterbox-tts` models require PyTorch with CUDA support, even when running on CPU. Solutions:

```bash
# Option 1: Use default setup (now includes CUDA-enabled PyTorch)
docker compose -f docker/docker-compose.yml up -d

# Option 2: Use explicit CUDA setup (traditional)
docker compose -f docker/docker-compose.gpu.yml up -d

# Option 3: Use uv + GPU setup (recommended for GPU users)
docker compose -f docker/docker-compose.uv.gpu.yml up -d

# Option 4: Use CPU-only setup (may have compatibility issues)
docker compose -f docker/docker-compose.cpu.yml up -d

# Option 5: Clear model cache and retry with CUDA-enabled setup
docker volume rm chatterbox-tts-api_chatterbox-models
docker compose -f docker/docker-compose.yml up -d --build

# Option 6: Try uv for better dependency resolution
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 4123
```

**For local development**, install PyTorch with CUDA support:

```bash
# With pip
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/travisvn/chatterbox-multilingual.git@exp

# With uv (handles this automatically)
uv sync
```

**Windows Users**, using pip & having issues:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
pip install --force-reinstall typing_extensions
```

**Port conflicts**

```bash
# Change port
echo "PORT=4124" >> .env
```

**GPU not detected**

```bash
# Force CPU mode
echo "DEVICE=cpu" >> .env
```

**Out of memory**

```bash
# Reduce chunk size
echo "MAX_CHUNK_LENGTH=200" >> .env
```

**Model download fails**

```bash
# Clear cache and retry
rm -rf models/
uvicorn app.main:app --host 0.0.0.0 --port 4123  # or: uv run main.py
```

**FastAPI startup issues**

```bash
# Check if uvicorn is installed
uvicorn --version

# Run with verbose logging
uvicorn app.main:app --host 0.0.0.0 --port 4123 --log-level debug

# Alternative startup method
python main.py
```

</details>

<details>
<summary><strong>💻 Development</strong></summary>

### Project Structure

This project follows a clean, modular architecture for maintainability:

```
app/                     # FastAPI backend application
├── __init__.py           # Main package
├── config.py            # Configuration management
├── main.py              # FastAPI application
├── models/              # Pydantic models
│   ├── requests.py      # Request models
│   └── responses.py     # Response models
├── core/                # Core functionality
│   ├── memory.py        # Memory management
│   ├── text_processing.py # Text processing utilities
│   └── tts_model.py     # TTS model management
└── api/                 # API endpoints
    ├── router.py        # Main router
    └── endpoints/       # Individual endpoint modules
        ├── speech.py    # TTS endpoint
        ├── health.py    # Health check
        ├── models.py    # Model listing
        ├── memory.py    # Memory management
        └── config.py    # Configuration

frontend/                # React frontend application
├── src/
├── Dockerfile
├── nginx.conf          # Integrated proxy configuration
└── package.json

docker/                  # Docker files consolidated
├── Dockerfile
├── Dockerfile.uv       # uv-optimized image
├── Dockerfile.gpu      # GPU-enabled image
├── Dockerfile.cpu      # CPU-only image
├── Dockerfile.uv.gpu   # uv + GPU image
├── docker-compose.yml  # Standard deployment
├── docker-compose.uv.yml # uv deployment
├── docker-compose.gpu.yml # GPU deployment
├── docker-compose.uv.gpu.yml # uv + GPU deployment
└── docker-compose.cpu.yml # CPU-only deployment

tests/                   # Test suite
├── test_api.py         # API tests
└── test_memory.py      # Memory tests

main.py                  # Main entry point
start.py                 # Development helper script
```

### Quick Start Scripts

```bash
# Development mode with auto-reload
python start.py dev

# Production mode
python start.py prod

# Full Stack mode with UI (using Docker)
python start.py fullstack

# Run tests
python start.py test

# View project structure
python start.py info
```

### Local Development

```bash
# Install in development mode (pip)
pip install -e .

# Or with uv (basic development tools)
uv sync

# Or with test dependencies (for contributors)
uv sync --group test

# Start with auto-reload (FastAPI development)
uvicorn app.main:app --host 0.0.0.0 --port 4123 --reload

# Or use the main script
python main.py

# Or use the development helper
python start.py dev
```

### Testing

```bash
# Run API tests
python tests/test_api.py  # or: uv run tests/test_api.py

# Run memory tests
python tests/test_memory.py

# Test specific endpoint
curl http://localhost:4123/health

# Check API documentation
curl http://localhost:4123/openapi.json
```

### FastAPI Development Features

- **Auto-reload**: Use `--reload` flag for development
- **Interactive docs**: Visit `/docs` for live API testing
- **Type hints**: Full IDE support with Pydantic models
- **Validation**: Automatic request/response validation
- **Modular structure**: Easy to extend and maintain

</details>

<details>
<summary><strong>🤝 Contributing</strong></summary>

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure FastAPI docs are updated
6. Submit a pull request

</details>

## Issues with Multilingual?

Fallback to the LKG (last known good) for the pre-multilingual release

```bash
git clone --branch stable https://github.com/travisvn/chatterbox-tts-api
```

[View `stable` branch](https://github.com/travisvn/chatterbox-tts-api/tree/stable) to see proper install / troubleshooting documentation

## Support

- 📖 **Documentation**: See [API Documentation](docs/API_README.md) and [Docker Guide](docs/DOCKER_README.md)
- 🐛 **Issues**: Report bugs via GitHub issues or on [our Discord](http://chatterboxtts.com/discord)
- 💬 **Discord**: [Join the Discord for this project](http://chatterboxtts.com/discord)

---

## 🔗 Integrations

### Open WebUI

> [!TIP]
> Customize available voices first by using the frontend at `http://localhost:4321`

To use Chatterbox TTS API with Open WebUI, follow these steps:

- Open the Admin Panel and go to `Settings` -> `Audio`
- Set your TTS Settings to match the following:
  - Text-to-Speech Engine: _OpenAI_
  - API Base URL: `http://localhost:4123/v1` # alternatively, try `http://host.docker.internal:4123/v1`
  - API Key: `none`
  - TTS Model: `tts-1` or `tts-1-hd`
  - TTS Voice: _Name of the voice you've cloned_ (can also include aliases, defined in the frontend)
  - Response splitting: `Paragraphs`

<p align="center">
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOUjUe3QjHytHQ0xqn2CishmXgGfeJ4o983TUMO" alt="Settings to integrate Chatterbox TTS API with Open WebUI" />
</p>

### ➡️ View the [Open WebUI docs for installing Chatterbox TTS API](https://docs.openwebui.com/tutorials/text-to-speech/chatterbox-tts-api-integration)
