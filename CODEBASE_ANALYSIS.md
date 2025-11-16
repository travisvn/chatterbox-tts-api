# Chatterbox TTS API - Complete Codebase Analysis

## 1. OVERALL PROJECT STRUCTURE

### Directory Layout
```
chatterbox-tts-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── router.py           # Main router combining all endpoints
│   │   └── endpoints/          # All API endpoint implementations
│   │       ├── speech.py       # Core TTS speech generation endpoints (4 endpoints)
│   │       ├── long_text.py    # Long-form TTS async endpoints (13 endpoints)
│   │       ├── voices.py       # Voice library management (14 endpoints)
│   │       ├── models.py       # Model listing endpoint (1 endpoint)
│   │       ├── health.py       # Health checks (2 endpoints)
│   │       ├── status.py       # TTS status tracking (5 endpoints)
│   │       ├── memory.py       # Memory management (4 endpoints)
│   │       ├── config.py       # Configuration endpoints (2 endpoints)
│   │       └── language_models.py  # Language-specific models (3 endpoints)
│   ├── models/                 # Pydantic request/response models
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── long_text.py        # Long-text specific models
│   └── core/                   # Core functionality
│       ├── tts_model.py        # TTS model management and loading
│       ├── tts_engines/        # Multiple TTS engine implementations
│       │   ├── base.py         # Abstract base class
│       │   ├── chatterbox.py   # Chatterbox engine (primary)
│       │   ├── indextts.py     # IndexTTS-2 engine
│       │   ├── higgs_audio.py  # Higgs Audio V2 engine
│       │   └── vibevoice.py    # VibeVoice engine
│       ├── voice_library.py    # Voice management
│       ├── pause_handler.py    # Punctuation pause processing
│       ├── text_processing.py  # Text chunking and processing
│       ├── long_text_jobs.py   # Async long-text job management
│       ├── background_tasks.py # Background job processor
│       ├── memory.py           # Memory monitoring
│       ├── status.py           # Request status tracking
│       ├── audio_processing.py # Audio concatenation
│       ├── language_models.py  # Language-specific model management
│       └── mtl.py              # Multilingual language support
├── tests/                      # Comprehensive test suite (14 test files, 3142 lines)
├── frontend/                   # React web UI (optional)
├── docker/                     # Docker deployment configurations
├── docs/                       # Extensive documentation
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
└── BUGFIX_REPORT.md           # Documentation of recent bug fixes
```

### Key Statistics
- **Total Python Files**: 40+
- **Test Files**: 14 (comprehensive coverage)
- **Test Lines**: 3,142 lines
- **Total API Endpoints**: 48+ endpoints across 8 endpoint modules
- **TTS Engines Supported**: 4 engines with 8 model variants

---

## 2. SUPPORTED TTS MODELS AND ENGINES

### A. Chatterbox (Primary Engine) - 4 Variants
**Repository**: ResembleAI/chatterbox  
**File**: `app/core/tts_engines/chatterbox.py`

| Model ID | Type | Languages | Features | Notes |
|----------|------|-----------|----------|-------|
| `chatterbox-v1` | Standard | English | Voice cloning, emotion control | Experimental fork |
| `chatterbox-v2` | Standard | English | Voice cloning, emotion control | Official release |
| `chatterbox-multilingual-v1` | Multilingual | 23 languages | Voice cloning, emotion, multilingual | Experimental fork |
| `chatterbox-multilingual-v2` | Multilingual | 23 languages | Voice cloning, emotion, multilingual | Official release (DEFAULT) |

**Sample Rate**: 24,000 Hz  
**Device Support**: CPU, CUDA, MPS  
**Voice Cloning**: Yes (zero-shot via audio prompt)

### B. IndexTTS-2 Engine
**Repository**: IndexTeam/IndexTTS-2  
**File**: `app/core/tts_engines/indextts.py`

| Model ID | Type | Languages | Features | Hardware |
|----------|------|-----------|----------|----------|
| `indextts-2` | Advanced | Multi-language | 8 emotion vectors, duration control, zero-shot voice cloning | 8GB+ VRAM |

**Key Features**:
- Industrial-level controllable TTS
- Emotion control through emotion vectors
- Precise synthesis duration control
- Zero-shot voice cloning
- Installation: `pip install indextts` + model download from HuggingFace

### C. Higgs Audio V2 Engine
**Repository**: boson-ai/higgs-audio  
**File**: `app/core/tts_engines/higgs_audio.py`

| Model ID | Type | Features | Hardware |
|----------|------|----------|----------|
| `higgs-audio-v2` | Advanced | Multi-speaker, neural voice cloning, 90 min generation | 24GB+ VRAM (GPU essential) |

**Key Features**:
- Neural voice cloning from 30+ second reference audio
- Multi-speaker conversations
- High-quality synthesis (up to 90 minutes)
- Text-audio foundation model
- Installation: Clone from GitHub + `pip install -e .`

### D. VibeVoice Engine
**Repository**: Microsoft Research  
**File**: `app/core/tts_engines/vibevoice.py`

| Model ID | Type | Context | Duration | VRAM | Features |
|----------|------|---------|----------|------|----------|
| `vibevoice-1.5b` | Large | 64K tokens | 90 min | 8GB+ | Long-form, multi-speaker, expressive |
| `vibevoice-7b` | Extra Large | 32K tokens | 45 min | 16GB+ | Long-form, multi-speaker, expressive |

**Key Features**:
- Expressive, long-form conversational speech synthesis
- Multi-speaker support (up to 4 speakers)
- Context-aware generation
- Background music support
- Zero-shot voice cloning
- Installation: Clone from community repository + `pip install -e .`

### Language Support
- **Chatterbox Multilingual**: 23 languages (English, German, French, Spanish, Italian, etc.)
- **IndexTTS-2**: Multi-language (specific languages TBD from implementation)
- **VibeVoice**: Long-form conversational (language support via context)
- **Higgs Audio**: Multilingual capable

### OpenAI Compatibility
- `tts-1` - Maps to default model
- `tts-1-hd` - Maps to default model

---

## 3. COMPLETE API ENDPOINTS (48+)

### A. Speech Generation Endpoints (4 main, 2 streaming = 4 routes)
**Module**: `app/api/endpoints/speech.py`

#### Core Endpoints
| Method | Path | Aliases | Purpose | Features |
|--------|------|---------|---------|----------|
| POST | `/audio/speech` | `/v1/audio/speech` | Generate speech (JSON) | Standard TTS, streaming/SSE formats |
| POST | `/audio/speech/upload` | `/v1/audio/speech/upload` | Generate with voice file upload | Library OR custom voice file |
| POST | `/audio/speech/stream` | `/v1/audio/speech/stream` | Stream speech generation | Real-time chunked audio |
| POST | `/audio/speech/stream/upload` | `/v1/audio/speech/stream/upload` | Stream with voice upload | Streaming + custom voice |

**Request Parameters**:
- `input` (required): Text to synthesize (1-3000 chars)
- `model` (optional): Model version (chatterbox-*, indextts-2, higgs-audio-v2, vibevoice-*)
- `voice` (optional): Voice name or alias (default: alloy)
- `response_format`: Audio format (always WAV)
- `speed`: Speech speed (placeholder, ignored)
- `exaggeration` (0.25-2.0): Emotion intensity (default: 0.5)
- `cfg_weight` (0.0-1.0): Pace control (default: 0.5)
- `temperature` (0.05-5.0): Sampling randomness (default: 0.8)
- `stream_format`: 'audio' or 'sse' for Server-Sent Events
- `streaming_chunk_size` (50-500): Chars per chunk
- `streaming_strategy`: 'sentence', 'paragraph', 'fixed', 'word'
- `streaming_quality`: 'fast', 'balanced', 'high'
- `enable_pauses`: Enable punctuation pauses
- `custom_pauses`: Override pause durations

---

### B. Long Text TTS Endpoints (13 endpoints)
**Module**: `app/api/endpoints/long_text.py`

For texts exceeding configured minimum (default: 3000 chars, max: 100,000 chars)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/audio/speech/long` | Create async long-text job |
| GET | `/audio/speech/long` | List all jobs |
| GET | `/audio/speech/long/{job_id}` | Get job status |
| GET | `/audio/speech/long/{job_id}/details` | Detailed job info |
| GET | `/audio/speech/long/{job_id}/download` | Download completed audio |
| GET | `/audio/speech/long/{job_id}/sse` | Server-Sent Events progress stream |
| PATCH | `/audio/speech/long/{job_id}` | Update job (pause/cancel) |
| PUT | `/audio/speech/long/{job_id}/pause` | Pause job |
| PUT | `/audio/speech/long/{job_id}/resume` | Resume job |
| POST | `/audio/speech/long/{job_id}/retry` | Retry failed job |
| POST | `/audio/speech/long/bulk` | Submit multiple jobs |
| GET | `/audio/speech/long-history` | Get job history |
| GET | `/audio/speech/long-history/stats` | Job statistics |
| DELETE | `/audio/speech/long/{job_id}` | Delete job |
| DELETE | `/audio/speech/long/history` | Clear history |

---

### C. Voice Library Management (14 endpoints)
**Module**: `app/api/endpoints/voices.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/voices` | List all voices in library |
| POST | `/voices` | Upload new voice |
| GET | `/voices/{voice_name}` | Get voice info |
| PUT | `/voices/{voice_name}` | Rename voice |
| DELETE | `/voices/{voice_name}` | Delete voice |
| GET | `/voices/{voice_name}/download` | Download voice file |
| GET | `/voices/default` | Get default voice |
| POST | `/voices/default` | Set default voice |
| DELETE | `/voices/default` | Reset to system default |
| POST | `/voices/{voice_name}/aliases` | Add alias to voice |
| DELETE | `/voices/{voice_name}/aliases/{alias}` | Remove alias |
| GET | `/voices/{voice_name}/aliases` | List voice aliases |
| GET | `/voices/all-names` | List all voice names and aliases |
| POST | `/voices/cleanup` | Clean up missing voice files |
| GET | `/languages` | Get supported languages for voices |

---

### D. Model Management (1 endpoint)
**Module**: `app/api/endpoints/models.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/models` | List all available models |

Returns: OpenAI-compatible model list (includes tts-1, tts-1-hd, and all engine-specific models)

---

### E. Health & Status Endpoints (7 endpoints)
**Module**: `app/api/endpoints/health.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check with model/device status |
| GET | `/ping` | Simple connectivity check |

**Module**: `app/api/endpoints/status.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/status` | Comprehensive TTS status (memory, history, stats optional) |
| GET | `/status/progress` | Current TTS progress (lightweight) |
| GET | `/status/history` | Request history |
| GET | `/status/statistics` | Processing statistics |
| POST | `/status/history/clear` | Clear history (requires confirmation) |
| GET | `/info` | Complete API info (version, status, memory, stats) |

---

### F. Configuration Endpoints (2 endpoints)
**Module**: `app/api/endpoints/config.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/config` | Get current configuration |
| GET | `/endpoints` | List all available endpoints and aliases |

---

### G. Memory Management (4 endpoints)
**Module**: `app/api/endpoints/memory.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/memory` | Get memory info with alerts |
| POST | `/memory/reset` | Force memory cleanup |
| GET | `/memory/config` | Get memory config |
| POST | `/memory/config` | Update memory config |

---

### H. Language Models (3 endpoints)
**Module**: `app/api/endpoints/language_models.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/languages` | List supported languages |
| GET | `/language-models` | List language-specific models |
| GET | `/languages/{language_code}/models` | Get models for specific language |

---

## 4. EXISTING TEST FILES AND COVERAGE

### Test Files Overview (14 files, 3,142 lines)

| File | Lines | Focus Area | Status |
|------|-------|-----------|--------|
| test_api.py | 358 | Core TTS JSON/streaming endpoints | Active |
| test_memory.py | 392 | Memory management and cleanup | Active |
| test_pause_handler.py | 129 | Punctuation pause handling | Active |
| test_regression.py | 414 | Regression testing, edge cases | Active |
| test_sse_streaming.py | 230 | Server-Sent Events streaming | Active |
| test_status.py | 245 | Status tracking API | Active |
| test_streaming_frontend.py | 256 | Streaming performance/frontend | Active |
| test_summary.py | 146 | Quick test summary | Active |
| test_vibevoice_integration.py | 366 | NEW - VibeVoice engine testing | NEW! |
| test_voice_library.py | 260 | Voice upload/management | Active |
| test_voice_upload.py | 346 | Voice file upload handling | Active |
| conftest.py | - | Test configuration and fixtures | - |
| run_tests.py | - | Test runner script | - |

### Test Coverage Areas
1. **Health Checks**: `/health`, `/ping` endpoints
2. **TTS Generation**: JSON and streaming endpoints with various text lengths
3. **Voice Library**: Upload, rename, delete, alias management
4. **Parameters**: Exaggeration, CFG weight, temperature variations
5. **Memory Management**: Cleanup, monitoring, alerts
6. **Status Tracking**: Progress, history, statistics
7. **Streaming**: Both raw audio and SSE formats
8. **Long Text**: Async job management
9. **Pause Handling**: Punctuation-based pauses
10. **Regression**: Edge cases, error handling
11. **VibeVoice Integration**: NEW model testing

### Test Utilities
- `conftest.py`: Fixtures for API client, test output directory, test parameters
- `run_tests.py`: Automated test runner with options
- Parametrized tests for multiple text lengths and parameter combinations
- Concurrent request testing
- Mock health checks

---

## 5. CONFIGURATION FILES AND SETUP

### Configuration Files
1. **`.env.example`** (189 lines) - Primary configuration template
2. **`.env.example.docker`** - Docker-optimized paths
3. **`app/config.py`** - Configuration class with validation
4. **`requirements.txt`** - Dependency specification
5. **`pyproject.toml`** - Project metadata
6. **`setup.py`** - Installation script

### Key Configuration Parameters

#### Server
- `HOST` (default: 0.0.0.0)
- `PORT` (default: 4123)
- `CORS_ORIGINS` (default: *)

#### TTS Model Settings
- `DEVICE` (auto/cuda/mps/cpu) - Device selection
- `USE_MULTILINGUAL_MODEL` (default: true)
- `DEFAULT_MODEL_VERSION` (v1/v2, default: v2)
- `DEFAULT_TTS_ENGINE` (chatterbox/indextts/higgs, default: chatterbox)

#### TTS Parameters
- `EXAGGERATION` (0.25-2.0, default: 0.5) - Emotion intensity
- `CFG_WEIGHT` (0.0-1.0, default: 0.5) - Pace control
- `TEMPERATURE` (0.05-5.0, default: 0.8) - Randomness

#### Text Processing
- `MAX_CHUNK_LENGTH` (default: 280 chars)
- `MAX_TOTAL_LENGTH` (default: 3000 chars)

#### Long Text Configuration
- `LONG_TEXT_MIN_LENGTH` (default: 3000 chars)
- `LONG_TEXT_MAX_LENGTH` (default: 100,000 chars)
- `LONG_TEXT_CHUNK_SIZE` (default: 2500 chars)
- `LONG_TEXT_BATCH_SIZE` (default: 4)
- `LONG_TEXT_MAX_CONCURRENT_JOBS` (default: 3)
- `LONG_TEXT_SILENCE_PADDING_MS` (default: 200ms)

#### Pause Handling
- `ENABLE_PUNCTUATION_PAUSES` (default: true)
- `ELLIPSIS_PAUSE_MS` (default: 800ms)
- `EM_DASH_PAUSE_MS` (default: 550ms)
- `EN_DASH_PAUSE_MS` (default: 375ms)
- `PERIOD_PAUSE_MS` (default: 500ms)
- `PARAGRAPH_PAUSE_MS` (default: 800ms)
- `LINE_BREAK_PAUSE_MS` (default: 350ms)
- `MIN_PAUSE_MS` (default: 200ms)
- `MAX_PAUSE_MS` (default: 2000ms)

#### Memory Management
- `MEMORY_CLEANUP_INTERVAL` (default: 5 requests)
- `CUDA_CACHE_CLEAR_INTERVAL` (default: 3 requests)
- `ENABLE_MEMORY_MONITORING` (default: true)

#### Quality Presets
Three preset configurations (fast, balanced, high):
- `QUALITY_FAST_CHUNK_SIZE`: 1500 chars
- `QUALITY_BALANCED_CHUNK_SIZE`: 2500 chars
- `QUALITY_HIGH_CHUNK_SIZE`: 2800 chars
- CFG weight and temperature adjustments per preset

---

## 6. POTENTIAL ISSUES AND INCONSISTENCIES FOUND

### CRITICAL BUGS - RECENTLY FIXED (BUGFIX_REPORT.md)
✅ **ALL FIXED** - The following critical issues were identified and fixed:

1. **Missing Return Statement** (FIXED)
   - **File**: `app/core/tts_model.py:410`
   - **Issue**: `get_model_info()` function returned None instead of model info
   - **Impact**: Model info API would return null
   - **Status**: FIXED

2. **Wrong Function Call in SSE Streaming** (FIXED)
   - **File**: `app/api/endpoints/speech.py:737`
   - **Issue**: Called `get_model()` instead of `get_or_load_model()`
   - **Impact**: SSE streaming would fail
   - **Status**: FIXED

3. **Missing `model_version` Parameter** (FIXED)
   - **File**: `app/api/endpoints/speech.py:702`
   - **Issue**: Function signature missing parameter used in function
   - **Impact**: NameError at runtime
   - **Status**: FIXED

4. **Missing `model` Parameter** (FIXED)
   - **File**: `app/api/endpoints/speech.py:1241`
   - **Issue**: Parameter missing from stream_text_to_speech_with_upload()
   - **Impact**: NameError at runtime
   - **Status**: FIXED

---

### KNOWN LIMITATIONS & ISSUES

#### 1. **Optional Dependencies for New Engines**
- **IndexTTS-2**: Requires `pip install indextts` + manual model download
- **Higgs Audio V2**: Manual GitHub clone + installation required
- **VibeVoice**: Manual GitHub clone + installation required
- **Impact**: Users won't have access to these models unless explicitly installed
- **Recommendation**: Document required steps clearly

#### 2. **Hardware Requirements Misalignment**
- **VibeVoice-1.5B**: Needs 8GB+ VRAM but 16GB recommended
- **VibeVoice-7B**: Needs 16GB+ but 24GB recommended
- **Higgs Audio V2**: Requires 24GB+ VRAM (GPU essential)
- **Impact**: Silent failures on insufficient hardware
- **Recommendation**: Add pre-flight hardware checks

#### 3. **Model Loading Race Conditions**
- **Issue**: Multiple concurrent requests might trigger simultaneous model loads
- **Current Mitigation**: Lazy loading with registry
- **Recommendation**: Add locking mechanism for thread-safe model loading

#### 4. **Memory Cleanup Heuristics**
- Current: Cleanup every N requests (configurable)
- **Issue**: Doesn't account for actual memory pressure
- **Recommendation**: Implement threshold-based cleanup instead

#### 5. **Error Handling Gaps**
- **SSE Endpoints**: Limited error recovery in streaming context
- **Long Text Jobs**: Job state transitions not fully validated
- **Voice Library**: No transaction support for multi-step operations
- **Recommendation**: Enhance error handling in streaming/async contexts

#### 6. **Multilingual Support Inconsistencies**
- Chatterbox: 23 languages supported
- IndexTTS-2: Language support unclear from code
- VibeVoice: Language support unclear
- **Impact**: Language parameter may be ignored by some engines
- **Recommendation**: Document language support per engine

#### 7. **Streaming Implementation Concerns**
- **SSE Streaming**: Base64 encoding of audio adds overhead
- **WAV Header**: Streaming WAV header uses max size placeholder
- **Player Compatibility**: Some players may have issues with streaming WAV
- **Recommendation**: Add MP3 streaming option

#### 8. **Voice Library Persistence**
- **Issue**: Voice library persisted to disk but no backup/recovery mechanism
- **Risk**: Metadata corruption could lose voice references
- **Recommendation**: Implement atomic operations and backup/recovery

#### 9. **Long Text Job Cleanup**
- **Issue**: Jobs cleaned up after N days without checking if downloaded
- **Impact**: Users might lose access to completed jobs
- **Recommendation**: Implement explicit cleanup request before auto-delete

#### 10. **Configuration Validation Incomplete**
- **Issue**: Some parameter combinations not validated (e.g., streaming_strategy)
- **Impact**: Invalid parameters silently ignored
- **Recommendation**: Fail fast with validation errors

---

### ARCHITECTURAL INCONSISTENCIES

1. **Endpoint Aliasing Complexity**
   - Both `/audio/speech` and `/v1/audio/speech` supported
   - Multiple layers of aliasing (route_aliases, endpoint_info)
   - **Risk**: Maintenance burden for multiple paths
   - **Recommendation**: Standardize on single path per resource

2. **Model Version Selection**
   - Model version specified via config + request parameter
   - Unclear precedence: which takes priority?
   - **Recommendation**: Document clearly or use explicit hierarchy

3. **Voice Resolution Logic**
   - Multiple fallback chains (library → aliases → OpenAI names → default)
   - Complex resolution in `resolve_voice_path_and_language()`
   - **Risk**: Unexpected voice selection
   - **Recommendation**: Explicit error vs. silent fallback

4. **Status Tracking Granularity**
   - Fine-grained status tracking but no aggregation for batch operations
   - **Issue**: Long text bulk operations lack consolidated status
   - **Recommendation**: Add batch status tracking

5. **Pause Handling vs. Text Processing**
   - Pause handling happens in speech.py, not text_processing.py
   - **Risk**: Code organization inconsistency
   - **Recommendation**: Move pause logic to unified text processing module

---

### TESTING GAPS

1. **New VibeVoice Integration**
   - Added recently but integration not fully validated
   - Test file exists but may need expansion
   - **Recommendation**: Add hardware-aware skip tests for unavailable models

2. **Multi-Engine Switching**
   - No tests for switching between different engines
   - **Recommendation**: Add engine switching integration tests

3. **Error Recovery**
   - Limited testing of failure scenarios
   - **Recommendation**: Add chaos testing for network/resource failures

4. **Performance Testing**
   - No load testing or concurrent request limits
   - **Recommendation**: Add k6 or locust performance tests

5. **Streaming Edge Cases**
   - Large file streaming not tested
   - Connection drop recovery not tested
   - **Recommendation**: Add streaming robustness tests

---

## 7. DEPENDENCIES AND COMPATIBILITY

### Required Dependencies
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pydantic >= 2.0.0
- torch >= 2.0.0, < 2.7.0
- torchaudio >= 2.0.0, < 2.7.0
- chatterbox-tts == 0.1.4
- huggingface-hub >= 0.20.0
- sse-starlette >= 3.0.2
- psutil >= 5.9.0
- python-dotenv >= 1.0.0

### Optional Dependencies (Lazy-Loaded)
- indextts (for IndexTTS-2)
- boson-multimodal (for Higgs Audio V2)
- vibevoice (for VibeVoice)

### Python Version
- Required: Python 3.10+
- Tested: Python 3.11

### OS Support
- Linux (primary)
- macOS (with MPS support)
- Windows (CPU/CUDA only)

---

## SUMMARY & RECOMMENDATIONS

### Strengths
1. ✅ Well-organized modular architecture
2. ✅ OpenAI-compatible API design
3. ✅ Comprehensive documentation and test coverage
4. ✅ Multiple TTS engine support (4 engines, 8 models)
5. ✅ Advanced features (streaming, SSE, async long-text, voice library)
6. ✅ Recent bug fixes applied (see BUGFIX_REPORT.md)
7. ✅ Extensive configuration options
8. ✅ Memory management and monitoring

### Priority Improvements Needed
1. **Hardware Pre-flight Checks**: Validate VRAM before loading large models
2. **Thread-Safe Model Loading**: Prevent race conditions with concurrent requests
3. **Comprehensive Error Handling**: Better error messages for missing dependencies
4. **Performance Testing**: Add load testing and concurrent request limits
5. **Streaming Robustness**: Better error recovery for streaming operations
6. **Configuration Validation**: Fail fast for invalid parameter combinations
7. **Documentation**: Clearer troubleshooting guides for new model engines

### For Testing Comprehensive Coverage
Focus on these areas:
1. **All 4 TTS Engines** with basic generation test
2. **All 8 Model Variants** (Chatterbox × 4, IndexTTS, Higgs, VibeVoice × 2)
3. **All 48+ API Endpoints** with various parameter combinations
4. **Streaming Formats**: Raw audio, WAV streaming, SSE
5. **Voice Operations**: Upload, rename, delete, alias, default voice
6. **Long Text**: Async job lifecycle (create, monitor, download, retry, delete)
7. **Error Handling**: Invalid inputs, missing models, resource exhaustion
8. **Concurrent Requests**: Multiple simultaneous TTS generations
9. **Parameter Ranges**: Boundary values for exaggeration, cfg_weight, temperature
10. **Language Support**: Multilingual generation and language-specific models

