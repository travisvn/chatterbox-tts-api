# Chatterbox TTS API - Quick Reference Guide

## Base URL
```
http://localhost:4123
```

## Core TTS Endpoints (4)

### 1. Generate Speech (JSON)
```bash
POST /v1/audio/speech
Content-Type: application/json

{
  "input": "Hello, world!",
  "model": "chatterbox-multilingual-v2",
  "voice": "alloy",
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "temperature": 0.8,
  "stream_format": "audio",  # or "sse" for Server-Sent Events
  "enable_pauses": true,
  "custom_pauses": {"...": 800}
}
```

### 2. Generate Speech with Voice Upload
```bash
POST /v1/audio/speech/upload
Content-Type: multipart/form-data

input=Hello world
model=chatterbox-multilingual-v2
voice=alloy
voice_file=@voice-sample.mp3
exaggeration=0.5
stream_format=audio
```

### 3. Stream Speech
```bash
POST /v1/audio/speech/stream
Content-Type: application/json

{
  "input": "This is a long text...",
  "model": "chatterbox-multilingual-v2",
  "streaming_chunk_size": 200,
  "streaming_strategy": "sentence",  # sentence|paragraph|fixed|word
  "streaming_quality": "balanced"     # fast|balanced|high
}
```

### 4. Stream Speech with Voice Upload
```bash
POST /v1/audio/speech/stream/upload
Content-Type: multipart/form-data

input=Long text here...
streaming_strategy=paragraph
streaming_quality=high
voice_file=@my-voice.mp3
```

---

## Long Text Async Endpoints (13)

### Create Job
```bash
POST /v1/audio/speech/long
Content-Type: application/json

{
  "input": "Very long text (3000+ chars)...",
  "model": "chatterbox-multilingual-v2",
  "voice": "alloy",
  "response_format": "mp3"
}
```

### List Jobs
```bash
GET /v1/audio/speech/long
GET /v1/audio/speech/long?status=processing&session_id=my-session
```

### Get Job Status
```bash
GET /v1/audio/speech/long/{job_id}
```

### Get Job Details
```bash
GET /v1/audio/speech/long/{job_id}/details
```

### Download Audio
```bash
GET /v1/audio/speech/long/{job_id}/download
# Returns audio file (mp3 or wav)
```

### Stream Job Progress (SSE)
```bash
GET /v1/audio/speech/long/{job_id}/sse
# Real-time progress updates
```

### Control Job
```bash
PATCH /v1/audio/speech/long/{job_id}
Content-Type: application/json
{ "action": "pause" }  # or "cancel", "resume"

PUT /v1/audio/speech/long/{job_id}/pause
PUT /v1/audio/speech/long/{job_id}/resume
```

### Retry Failed Job
```bash
POST /v1/audio/speech/long/{job_id}/retry
```

### Bulk Operations
```bash
POST /v1/audio/speech/long/bulk
Content-Type: application/json

[
  { "input": "Text 1", "voice": "alloy", "model": "..." },
  { "input": "Text 2", "voice": "nova", "model": "..." }
]
```

### Job History & Stats
```bash
GET /v1/audio/speech/long-history
GET /v1/audio/speech/long-history/stats
DELETE /v1/audio/speech/long/history
```

### Delete Job
```bash
DELETE /v1/audio/speech/long/{job_id}
```

---

## Voice Library Management (14)

### List Voices
```bash
GET /v1/voices
```

### Upload Voice
```bash
POST /v1/voices
Content-Type: multipart/form-data

voice_name=alice
voice_file=@voice.mp3
language=en
```

### Voice Info
```bash
GET /v1/voices/{voice_name}
PUT /v1/voices/{voice_name}?new_name=bob
DELETE /v1/voices/{voice_name}
GET /v1/voices/{voice_name}/download
```

### Voice Aliases
```bash
POST /v1/voices/{voice_name}/aliases?alias=alice-clone
GET /v1/voices/{voice_name}/aliases
DELETE /v1/voices/{voice_name}/aliases/{alias}
```

### Default Voice
```bash
GET /v1/voices/default
POST /v1/voices/default?voice_name=alice
DELETE /v1/voices/default
```

### Other
```bash
GET /v1/voices/all-names          # All voice names and aliases
POST /v1/voices/cleanup           # Clean up missing files
GET /v1/languages                 # Supported languages
```

---

## Models & Language Models (4)

### List Models
```bash
GET /v1/models
```

### Language Models
```bash
GET /v1/languages                              # All languages
GET /v1/language-models                        # All language models
GET /v1/languages/{language_code}/models       # Models for language
```

---

## Health & Status (7)

### Health Check
```bash
GET /health
GET /ping
```

### Status & Progress
```bash
GET /v1/status
GET /v1/status/progress
GET /v1/status/history
GET /v1/status/statistics
GET /v1/info
POST /v1/status/history/clear?confirm=true
```

---

## Configuration & Memory (6)

### Configuration
```bash
GET /v1/config
GET /v1/endpoints
```

### Memory Management
```bash
GET /memory
POST /memory/reset
GET /memory/config
POST /memory/config
```

---

## Available Models

### Chatterbox (Default Engine)
- `chatterbox-v1` - English only, experimental
- `chatterbox-v2` - English only, official
- `chatterbox-multilingual-v1` - 23 languages, experimental
- `chatterbox-multilingual-v2` - 23 languages, official (DEFAULT)

### Advanced Engines
- `indextts-2` - IndexTTS-2 (8 emotion control vectors)
- `higgs-audio-v2` - Higgs Audio V2 (multi-speaker)
- `vibevoice-1.5b` - VibeVoice 1.5B (long-form, 8GB+ VRAM)
- `vibevoice-7b` - VibeVoice 7B (long-form, 16GB+ VRAM)

### OpenAI Compatibility
- `tts-1` - Maps to default model
- `tts-1-hd` - Maps to default model

---

## Common Parameter Ranges

| Parameter | Min | Default | Max | Notes |
|-----------|-----|---------|-----|-------|
| exaggeration | 0.25 | 0.5 | 2.0 | Emotion intensity |
| cfg_weight | 0.0 | 0.5 | 1.0 | Pace control (lower=faster) |
| temperature | 0.05 | 0.8 | 5.0 | Randomness (lower=deterministic) |
| input length | 1 | - | 3000 | Characters (short texts) |
| long_text | 3000 | - | 100000 | Characters (async) |
| streaming_chunk_size | 50 | 200 | 500 | Characters per chunk |

---

## Response Examples

### Success (TTS)
```
200 OK
Content-Type: audio/wav
[WAV binary data]
```

### Success (Long Text Job)
```json
{
  "job_id": "uuid-here",
  "status": "queued",
  "created_at": "2025-11-16T...",
  "estimated_completion": "2025-11-16T..."
}
```

### Error
```json
{
  "error": {
    "message": "Error description",
    "type": "error_type"
  }
}
```

---

## Useful Curl Examples

### Basic TTS
```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}' \
  --output audio.wav
```

### With Custom Parameters
```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "vibevoice-1.5b",
    "exaggeration": 0.8,
    "cfg_weight": 0.3,
    "temperature": 1.0
  }' \
  --output audio.wav
```

### Upload Custom Voice
```bash
curl -X POST http://localhost:4123/v1/voices \
  -F "voice_name=my_voice" \
  -F "voice_file=@my_voice.mp3" \
  -F "language=en"
```

### Stream with Custom Voice
```bash
curl -X POST http://localhost:4123/v1/audio/speech/stream/upload \
  -F "input=This is a test" \
  -F "voice_file=@voice.mp3"
```

### SSE Streaming
```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello",
    "stream_format": "sse"
  }' \
  --no-buffer
```

### Check Status
```bash
curl http://localhost:4123/v1/status
curl http://localhost:4123/health
```

---

## Web UI Access
```
Frontend:   http://localhost:4321
API Docs:   http://localhost:4123/docs
ReDoc:      http://localhost:4123/redoc
OpenAPI:    http://localhost:4123/openapi.json
```
