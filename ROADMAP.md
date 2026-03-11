# Roadmap — Chatterbox HTTPS Server

## API Contract Decision

### Candidates reviewed

| | OpenAI | ElevenLabs | Google Cloud | Azure |
|---|---|---|---|---|
| **Request format** | JSON (flat) | JSON + path param | JSON (nested) | SSML (XML) |
| **Response format** | Raw binary audio | Raw binary audio | Base64 in JSON | Raw binary audio |
| **Auth** | `Authorization: Bearer` | `xi-api-key` header | OAuth / API key | Subscription key or Bearer |
| **Required fields** | 3 (`model`, `input`, `voice`) | 2 (`voice_id` path + `text`) | 3 nested objects | SSML body + 3 headers |
| **Output format control** | Body field (`response_format`) | Query param | Body field (`audioEncoding`) | Header |
| **HTTP streaming** | Yes (chunked, default) | Yes (dedicated `/stream` endpoint) | No (REST; gRPC only) | Yes (format-dependent) |
| **Ecosystem adoption** | De facto standard for AI devs | Leading for voice cloning | Enterprise / IVR | Enterprise / call centers |

### Decision: **OpenAI-compatible** (`POST /v1/audio/speech`)

- **Simplest contract** — flat JSON, 3 required fields, binary audio response.
- **Widest AI-ecosystem adoption** — any client built for OpenAI TTS works out of the box.
- **Clean mapping** to chatterbox: `model` → turbo vs full, `voice` → reference audio preset, `input` → text.
- **Streaming** is implicit (chunked transfer), no extra endpoints needed.

### OpenAI TTS contract reference

```
POST /v1/audio/speech
Authorization: Bearer <key>
Content-Type: application/json

{
  "model": "tts-1",              // required — map to "turbo" / "full"
  "input": "Hello world",        // required — text to synthesize (max 4096 chars)
  "voice": "alloy",              // required — voice preset or reference ID
  "response_format": "wav",      // optional — mp3, opus, aac, flac, wav, pcm (default: mp3)
  "speed": 1.0                   // optional — 0.25 to 4.0
}

→ 200 OK
Content-Type: audio/wav (matches response_format)
Body: raw audio bytes
```

We will extend with a custom `x-audio-prompt` field (or multipart upload) for voice cloning with reference audio — not part of OpenAI's spec but additive and non-breaking.

---

## Stage 1 — Basic local setup with Turbo model

Goal: working HTTPS server on localhost, OpenAI-compatible TTS endpoint, basic tests.

### 1.1 Project scaffolding
- [ ] Add `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx` (test client) to dependencies
- [ ] Create `app/` package with `main.py`, `models.py`, `config.py`
- [ ] Create `tests/` package

### 1.2 Core server (`app/main.py`)
- [ ] FastAPI app with lifespan hook — load `ChatterboxTurboTTS` onto CUDA at startup
- [ ] `GET /health` — model status, device, VRAM usage
- [ ] `POST /v1/audio/speech` — OpenAI-compatible endpoint:
  - Accept JSON body: `model`, `input`, `voice`, `response_format`, `speed`
  - Map `model: "tts-1"` → Turbo (only option in stage 1)
  - Map `voice` → predefined reference audio presets (ship 2-3 bundled WAV clips) or `"default"` for no prompt
  - Return audio as raw binary stream with correct `Content-Type`
  - Support `wav` and `mp3` output formats at minimum
- [ ] Error handling: 422 for bad input, 503 if model not ready, 400 for unsupported formats

### 1.3 Voice cloning extension
- [ ] `POST /v1/audio/speech` with `multipart/form-data` alternative — allows uploading `audio_prompt` file alongside JSON fields
- [ ] Or: accept `voice` as an ID that maps to a previously uploaded reference clip
- [ ] `POST /v1/voices` — upload and register a reference WAV, returns a `voice_id`
- [ ] `GET /v1/voices` — list registered voices
- [ ] Store voice references on disk (volume-mounted path)

### 1.4 Self-signed HTTPS for local dev
- [ ] Generate self-signed cert in the dev container on first start
- [ ] Configure uvicorn with `--ssl-keyfile` / `--ssl-certfile`
- [ ] Document how to trust the cert locally or bypass in curl (`-k`)

### 1.5 Tests
- [ ] `tests/test_health.py` — health endpoint returns expected shape
- [ ] `tests/test_tts.py` — basic synthesis returns valid WAV bytes, correct Content-Type
- [ ] `tests/test_validation.py` — missing fields → 422, bad model → 400, empty input → 422
- [ ] `tests/conftest.py` — shared fixtures (test client, mock model for fast CI)
- [ ] Add `pytest`, `httpx` to dev dependencies
- [ ] Verify tests run inside the dev container

### 1.6 Dev container updates
- [ ] Forward port 8000 (HTTPS server) in addition to 7860 (Gradio)
- [ ] Add startup script option: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile ...`

---

## Stage 2 — Authenticated cloud setup with full model

Goal: production-ready deployment with API key auth, the larger 500M model, and cloud deployment.

### 2.1 Multi-model support
- [ ] Load `ChatterboxTTS` (500M full model) alongside or instead of Turbo
- [ ] Map `model: "tts-1"` → Turbo (fast, lower quality), `model: "tts-1-hd"` → Full (slower, higher quality)
- [ ] Full model exposes extra params: `exaggeration`, `cfg_weight` — accept via extension fields
- [ ] Configurable via env vars which models to load (save VRAM on smaller GPUs)

### 2.2 Authentication & API keys
- [ ] Bearer token auth middleware (`Authorization: Bearer <key>`)
- [ ] API key storage — env var for single-key setup, or a keys file/DB for multi-tenant
- [ ] Rate limiting per key (optional, via `slowapi` or similar)
- [ ] Reject unauthenticated requests with 401

### 2.3 TLS with real certificates
- [ ] Let's Encrypt / ACME integration (e.g., via caddy reverse proxy or certbot)
- [ ] Or: document cloud load balancer TLS termination (the more common path)

### 2.4 Production container
- [ ] Separate `Dockerfile.prod` — multi-stage build, no dev tools, smaller image
- [ ] `docker-compose.yml` with:
  - TTS service (uvicorn with gunicorn worker manager)
  - Reverse proxy (caddy or nginx) for TLS termination
  - Volume for model cache and voice references
- [ ] Health check in compose for orchestrator readiness probes
- [ ] Configurable worker count (default 1 for GPU — single model instance)

### 2.5 Cloud deployment
- [ ] Document deployment on a GPU cloud instance (RunPod, Lambda, AWS g5, etc.)
- [ ] Environment variables: `API_KEY`, `MODELS_TO_LOAD`, `LOG_LEVEL`, `WORKERS`
- [ ] Persistent volume for HuggingFace cache so cold starts only happen once

### 2.6 Observability
- [ ] Structured JSON logging
- [ ] Request ID tracking (`X-Request-ID` header)
- [ ] Basic metrics: requests/sec, latency p50/p95, VRAM usage
- [ ] Optional: Prometheus `/metrics` endpoint

### 2.7 Tests (extended)
- [ ] Auth tests: missing key → 401, invalid key → 401, valid key → 200
- [ ] Model selection tests: `tts-1` → turbo, `tts-1-hd` → full
- [ ] Load test: concurrent requests don't crash (GPU serialization)
- [ ] Output quality smoke test: generated audio is non-silent, correct sample rate
