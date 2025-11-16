# Comprehensive TTS API Testing Strategy

## Overview
This document provides a structured approach to comprehensively test all 48+ endpoints across 4 TTS engines with 8 model variants.

## Test Organization

### 1. Core Speech Generation (4 endpoints)
Each endpoint should be tested with:
- All 8 model variants
- Different input lengths (short, medium, long at limit)
- All streaming formats (audio, sse)
- Various parameter combinations
- Custom voices (uploaded)
- Error conditions

**Tests to Create**:
```
test_audio_speech_basic_generation.py
test_audio_speech_with_parameters.py
test_audio_speech_streaming.py
test_audio_speech_upload_voice.py
test_audio_speech_error_handling.py
```

### 2. Long Text Async (13 endpoints)
Test the complete job lifecycle:

**Tests to Create**:
```
test_long_text_job_creation.py
test_long_text_job_status.py
test_long_text_job_download.py
test_long_text_job_control.py          # pause/resume/cancel
test_long_text_job_retry.py
test_long_text_bulk_operations.py
test_long_text_history_stats.py
test_long_text_sse_streaming.py
test_long_text_job_cleanup.py
```

### 3. Voice Library Management (14 endpoints)
**Tests to Create**:
```
test_voice_library_upload.py            # Create
test_voice_library_get_info.py          # Read
test_voice_library_rename.py            # Update
test_voice_library_delete.py            # Delete
test_voice_library_aliases.py
test_voice_library_default_voice.py
test_voice_library_list_all.py
test_voice_library_download.py
test_voice_library_cleanup.py
test_voice_library_languages.py
```

### 4. Models & Configuration (4 endpoints)
**Tests to Create**:
```
test_models_listing.py
test_language_models.py
test_config_endpoints.py
test_supported_languages.py
```

### 5. Health & Status (7 endpoints)
**Tests to Create**:
```
test_health_endpoints.py
test_status_tracking.py
test_request_history.py
test_statistics.py
test_api_info.py
```

### 6. Memory Management (4 endpoints)
**Tests to Create**:
```
test_memory_monitoring.py
test_memory_cleanup.py
test_memory_alerts.py
test_memory_config.py
```

---

## Test Matrix

### Models to Test (8 variants)
- [ ] chatterbox-v1
- [ ] chatterbox-v2
- [ ] chatterbox-multilingual-v1
- [ ] chatterbox-multilingual-v2 (PRIMARY)
- [ ] indextts-2
- [ ] higgs-audio-v2
- [ ] vibevoice-1.5b
- [ ] vibevoice-7b

### Text Input Variations
```python
TEST_INPUTS = {
    "empty": "",                              # Should fail
    "minimal": "Hi",                          # 2 chars
    "short": "Hello, world!",                 # ~15 chars
    "medium": "The quick brown fox...",       # ~100 chars
    "long": "Lorem ipsum..." * 20,            # ~2000 chars
    "at_limit": "Text..." * 100,              # ~3000 chars (max for short endpoint)
    "over_limit": "Text..." * 150,            # >3000 chars (requires long endpoint)
    "max_long": "Text..." * 1000,             # ~30000 chars (within 100K long limit)
    "special_chars": "Hello... — test",       # Test pause handling
    "multilingual": "Bonjour, hola, hello"    # Test multilingual
}
```

### Parameter Combinations
```python
PARAMETER_MATRIX = [
    # Emotion/Expression variations
    {"exaggeration": 0.25, "cfg_weight": 0.5, "temperature": 0.8},  # Low emotion
    {"exaggeration": 1.5, "cfg_weight": 0.5, "temperature": 0.8},   # High emotion
    
    # Pace variations
    {"exaggeration": 0.5, "cfg_weight": 0.1, "temperature": 0.8},   # Fast
    {"exaggeration": 0.5, "cfg_weight": 0.9, "temperature": 0.8},   # Slow
    
    # Randomness variations
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.05},  # Deterministic
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 5.0},   # Creative
    
    # Edge cases
    {"exaggeration": 0.25, "cfg_weight": 0.0, "temperature": 0.05}, # Min values
    {"exaggeration": 2.0, "cfg_weight": 1.0, "temperature": 5.0},   # Max values
]
```

### Streaming Parameters
```python
STREAMING_CONFIGS = [
    {"streaming_strategy": "sentence", "streaming_quality": "fast"},
    {"streaming_strategy": "sentence", "streaming_quality": "balanced"},
    {"streaming_strategy": "sentence", "streaming_quality": "high"},
    
    {"streaming_strategy": "paragraph", "streaming_quality": "balanced"},
    {"streaming_strategy": "word", "streaming_quality": "fast"},
    {"streaming_strategy": "fixed", "streaming_chunk_size": 200},
]
```

---

## Critical Test Scenarios

### Scenario 1: Basic TTS Generation
**Goal**: Verify each engine generates valid audio

```python
def test_tts_generation_all_models():
    models = [
        "chatterbox-multilingual-v2",
        "indextts-2",
        "higgs-audio-v2",
        "vibevoice-1.5b"
    ]
    
    for model in models:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "model": model}
        )
        assert response.status_code == 200
        assert len(response.content) > 0
        assert response.headers["content-type"] == "audio/wav"
```

### Scenario 2: Voice Upload and Use
**Goal**: Verify voice upload and subsequent use

```python
def test_voice_upload_and_use():
    # 1. Upload voice
    voice_response = client.post(
        "/v1/voices",
        data={"voice_name": "test-voice", "language": "en"},
        files={"voice_file": open("test.mp3", "rb")}
    )
    assert voice_response.status_code == 201
    
    # 2. Use uploaded voice
    tts_response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello", "voice": "test-voice"}
    )
    assert tts_response.status_code == 200
    
    # 3. Clean up
    assert client.delete(f"/v1/voices/test-voice").status_code == 200
```

### Scenario 3: Long Text Processing
**Goal**: Verify async job lifecycle

```python
def test_long_text_complete_lifecycle():
    # 1. Create job
    job_response = client.post(
        "/v1/audio/speech/long",
        json={"input": "Very long text..." * 100}
    )
    assert job_response.status_code == 201
    job_id = job_response.json()["job_id"]
    
    # 2. Monitor status
    max_retries = 60
    for i in range(max_retries):
        status = client.get(f"/v1/audio/speech/long/{job_id}").json()
        if status["status"] == "completed":
            break
        time.sleep(1)
    
    # 3. Download audio
    download = client.get(f"/v1/audio/speech/long/{job_id}/download")
    assert download.status_code == 200
    assert len(download.content) > 0
    
    # 4. Clean up
    assert client.delete(f"/v1/audio/speech/long/{job_id}").status_code == 200
```

### Scenario 4: Streaming Generation
**Goal**: Verify real-time streaming

```python
def test_sse_streaming():
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello world",
            "stream_format": "sse"
        },
        stream=True
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Parse SSE events
    events = []
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            events.append(json.loads(line[6:]))
    
    assert any(e.get("type") == "done" for e in events)
```

### Scenario 5: Concurrent Requests
**Goal**: Verify thread-safety and concurrent handling

```python
def test_concurrent_tts_requests():
    import concurrent.futures
    
    def make_request(i):
        return client.post(
            "/v1/audio/speech",
            json={"input": f"Request {i}"}
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All requests should succeed
    assert all(r.status_code == 200 for r in results)
    assert all(len(r.content) > 0 for r in results)
```

### Scenario 6: Error Handling
**Goal**: Verify proper error responses

```python
def test_error_responses():
    # Missing input
    assert client.post("/v1/audio/speech", json={}).status_code == 422
    
    # Invalid model
    assert client.post(
        "/v1/audio/speech",
        json={"input": "test", "model": "nonexistent"}
    ).status_code in [400, 500]  # Depends on implementation
    
    # Text too long
    assert client.post(
        "/v1/audio/speech",
        json={"input": "x" * 10000}
    ).status_code == 400
    
    # Invalid parameters
    assert client.post(
        "/v1/audio/speech",
        json={
            "input": "test",
            "exaggeration": 10.0  # Out of range
        }
    ).status_code == 422
```

### Scenario 7: Pause Handling
**Goal**: Verify punctuation pause insertion

```python
def test_pause_handling():
    # With pauses enabled
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello... World—Test. Done!\n\nMore text.",
            "enable_pauses": True,
            "custom_pauses": {"...": 1000, "—": 600}
        }
    )
    assert response.status_code == 200
    
    # Verify audio duration increased (rough check)
    # Audio with pauses should be longer than without
```

### Scenario 8: Voice Aliases
**Goal**: Verify alias resolution

```python
def test_voice_aliases():
    # Upload voice
    client.post(
        "/v1/voices",
        data={"voice_name": "alice", "language": "en"},
        files={"voice_file": open("test.mp3", "rb")}
    )
    
    # Add alias
    alias_response = client.post(
        "/v1/voices/alice/aliases",
        data={"alias": "alice-v1"}
    )
    assert alias_response.status_code == 201
    
    # Use alias in TTS
    tts_response = client.post(
        "/v1/audio/speech",
        json={"input": "Test", "voice": "alice-v1"}
    )
    assert tts_response.status_code == 200
```

### Scenario 9: Multilingual Generation
**Goal**: Verify multilingual support

```python
def test_multilingual_generation():
    texts = {
        "en": "Hello world",
        "es": "Hola mundo",
        "fr": "Bonjour le monde",
        "de": "Hallo Welt",
    }
    
    for lang, text in texts.items():
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": text,
                "model": "chatterbox-multilingual-v2",
                "language_id": lang
            }
        )
        assert response.status_code == 200
```

### Scenario 10: Job Control (Pause/Resume)
**Goal**: Verify job lifecycle control

```python
def test_long_text_job_control():
    # Create job
    job = client.post(
        "/v1/audio/speech/long",
        json={"input": "x" * 50000}
    ).json()
    
    # Pause immediately
    pause = client.put(f"/v1/audio/speech/long/{job['job_id']}/pause")
    assert pause.status_code == 200
    
    # Resume
    resume = client.put(f"/v1/audio/speech/long/{job['job_id']}/resume")
    assert resume.status_code == 200
    
    # Cancel
    cancel = client.patch(
        f"/v1/audio/speech/long/{job['job_id']}",
        json={"action": "cancel"}
    )
    assert cancel.status_code == 200
```

---

## Performance Testing Checklist

- [ ] Load test with 50 concurrent TTS requests
- [ ] Load test with 10 concurrent long-text jobs
- [ ] Measure average response time per endpoint
- [ ] Verify memory cleanup under load
- [ ] Test with various text lengths (100, 1000, 3000+ chars)
- [ ] Test streaming chunk delivery latency
- [ ] Monitor GPU/CPU utilization
- [ ] Test sustained load for 1 hour
- [ ] Verify no memory leaks after extended use
- [ ] Test resource limits and graceful degradation

---

## Data Validation Tests

- [ ] WAV header validation for audio responses
- [ ] Audio duration matches expected text length
- [ ] Sample rate is 24kHz (or engine-specific)
- [ ] Streaming events are valid JSON
- [ ] Job status transitions are valid
- [ ] Timestamps are in ISO 8601 format
- [ ] File sizes are reasonable
- [ ] Voice library metadata is consistent
- [ ] Job history doesn't exceed limits
- [ ] Configuration values are within ranges

---

## Integration Tests

- [ ] Test with optional engines disabled
- [ ] Test model hot-loading (lazy loading)
- [ ] Test voice library persistence
- [ ] Test configuration reload
- [ ] Test across different devices (CPU/CUDA/MPS)
- [ ] Test database persistence of job data
- [ ] Test cleanup of orphaned files
- [ ] Test endpoint aliasing (`/audio/speech` vs `/v1/audio/speech`)
- [ ] Test error recovery and retry logic
- [ ] Test documentation accuracy (`/docs`, `/redoc`)

---

## Regression Tests

- [ ] Verify all previously passing tests still pass
- [ ] Check for memory regressions
- [ ] Check for performance regressions
- [ ] Verify backward compatibility
- [ ] Test with same test data as previous runs
- [ ] Compare audio output consistency

---

## Test Execution Plan

### Phase 1: Unit Tests (1-2 hours)
```bash
pytest tests/ -v --tb=short
```

### Phase 2: Integration Tests (2-3 hours)
```bash
pytest tests/ -v -m integration
```

### Phase 3: Performance Tests (1-2 hours)
```bash
pytest tests/ -v -m performance
```

### Phase 4: Load Tests (2-3 hours)
```bash
k6 run load_tests.js
```

---

## Documentation for Test Results

Save results to:
```
test_results/
├── test_execution_YYYY-MM-DD.json
├── coverage_report.html
├── performance_results.csv
├── load_test_results.json
└── summary.md
```

---

## Continuous Integration Checklist

- [ ] Run all tests on every commit
- [ ] Run load tests nightly
- [ ] Generate coverage reports
- [ ] Alert on test failures
- [ ] Track performance trends
- [ ] Archive test results
- [ ] Update test documentation
- [ ] Maintain test data freshness
