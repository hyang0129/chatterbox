# Roadmap — Chatterbox HTTPS Server

---

## Bug Fixes & Client Integration Requirements

These items were surfaced during integration with the video-agent pipeline and represent
correctness bugs or missing contract features that block reliable client use. They should
be resolved before Stage 1 work is considered stable.

---

### BUG-1 — `/tts` endpoint is sync: `asyncio.Lock` cannot be used, concurrent requests race on the GPU

**Severity:** High — silent correctness bug; can cause CUDA OOM or corrupted audio under concurrency.

**Root cause:**
`app/main.py::synthesize` is declared `def synthesize` (synchronous). FastAPI dispatches
sync routes to a `ThreadPoolExecutor`, so two concurrent HTTP requests can both call
`model.generate()` at the same time on the same model instance. `ChatterboxTurboTTS` is
not thread-safe; concurrent `generate()` calls on a shared GPU model risk CUDA OOM and
non-deterministic output.

Additionally, because the endpoint is sync, an `asyncio.Lock` placed on `app.state` is
unreachable — you cannot `await` inside a sync function, so the lock is silently never
acquired.

**Required fix:**
Convert the endpoint to `async def` and serialise inference with an `asyncio.Lock` held
for the duration of each `generate()` call. Move the blocking CPU/GPU work off the event
loop with `run_in_executor` so uvicorn can still accept new connections while inference runs:

```python
# In lifespan:
app.state.lock = asyncio.Lock()

# Endpoint:
@app.post("/tts", ...)
async def synthesize(req: TTSRequest) -> Response:
    async with app.state.lock:
        wav = await asyncio.get_running_loop().run_in_executor(
            None, lambda: app.state.model.generate(req.text, ...)
        )
    return Response(content=_encode_wav(wav, app.state.model.sr), media_type="audio/wav")
```

Concurrent HTTP requests will queue behind the lock; they do not fail unless a client
timeout fires. This is the correct behaviour for a single-GPU inference server.

---

### BUG-2 — `asyncio.get_event_loop()` is deprecated in Python 3.10+ and will break

**Severity:** Medium — currently emits DeprecationWarnings; will raise `RuntimeError` in a future Python version.

**Root cause:**
Any code path that calls `asyncio.get_event_loop()` inside a running coroutine (e.g. after
BUG-1 is fixed and the endpoint becomes `async def`) uses the deprecated form. Python 3.10
deprecated `get_event_loop()` when there is no current event loop in the calling thread;
Python 3.12 makes this an error inside coroutines.

**Required fix:**
Replace all `asyncio.get_event_loop()` calls inside coroutines with
`asyncio.get_running_loop()`. This is always safe inside an `async def` function and is
the forward-compatible form:

```python
# Before (deprecated):
loop = asyncio.get_event_loop()
wav = await loop.run_in_executor(None, ...)

# After:
wav = await asyncio.get_running_loop().run_in_executor(None, ...)
```

---

### BUG-3 — Direct (serverless) mode thread-safety is undocumented; callers may use it unsafely

**Severity:** Medium — silent data corruption or CUDA OOM if callers share a model instance across threads.

**Root cause:**
The serverless usage pattern (`ChatterboxTurboTTS.from_pretrained(device="cuda")` called
once, then `model.generate()` called per segment) is not thread-safe. Nothing in the
README, docstrings, or API reference communicates this constraint.

Callers that share a single model instance across threads (e.g. via a
`ThreadPoolExecutor`) will hit race conditions on internal model state, CUDA stream
conflicts, and potential OOM from overlapping allocations.

**Required fix:**
Add an explicit thread-safety warning to the README serverless section and to the
`ChatterboxTurboTTS` class docstring:

> **Thread safety:** A single `ChatterboxTurboTTS` instance must not be used from multiple
> threads concurrently. If you need concurrent synthesis, create one model instance per
> thread, or serialise all `generate()` calls behind a `threading.Lock`. On a single GPU
> with limited VRAM, serial single-instance use is strongly recommended.

Pipeline callers using `chatterbox_direct` backend mode must run the audio stage with
`serial=True` (no `ThreadPoolExecutor`). Violating this causes non-deterministic output
and potential CUDA OOM.

---

### FEAT-4 — Export a stdlib `TTSRequest` dataclass so clients don't depend on FastAPI/Pydantic

**Severity:** Low — currently forces clients to either duplicate the request schema or take a FastAPI dependency just for the type.

**Context:**
`TTSRequest` is currently a Pydantic `BaseModel` defined inside `app/main.py`. Clients
that want to construct a typed request object (e.g. the video-agent's
`src/tools/chatterbox_backend.py`) cannot import it without depending on `fastapi` and
`pydantic`, which are server-side dependencies.

**Requested addition:**
Add a pure-stdlib dataclass to `app/models.py` (or a standalone `chatterbox_client.py`)
that clients can import with zero heavy dependencies:

```python
from dataclasses import dataclass, field

@dataclass
class TTSRequest:
    text: str
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
```

The server-side Pydantic model can remain as-is for validation; this dataclass is an
additional client-facing type.

---

### FEAT-5 — `/tts` should return synthesis metadata in response headers

**Severity:** Low — callers currently must run `ffprobe` or parse the WAV header to obtain audio duration.

**Context:**
The `/tts` endpoint returns raw WAV bytes only. Clients that need the audio duration for
timeline alignment (e.g. to check whether the synthesised segment fits a planned scene
window) must shell out to `ffprobe` or parse the RIFF header themselves.

**Requested addition:**
Include the following HTTP response headers alongside the audio body:

| Header | Example | Notes |
|--------|---------|-------|
| `X-Audio-Duration-S` | `4.23` | Duration in seconds, two decimal places |
| `X-Sample-Rate` | `24000` | Native model sample rate (`model.sr`) |
| `X-Audio-Frames` | `101520` | Total PCM frame count |

These are zero-cost to produce (all values are known before the response is sent) and
eliminate the need for a second tool invocation on the client side.

---

### FEAT-6 — Document the native output sample rate (`model.sr`) in the API reference and README

**Severity:** Low — callers piping audio into tools that assume a different rate will get pitch/speed errors without warning.

**Context:**
`ChatterboxTurboTTS` outputs audio at its native sample rate, accessible as `model.sr`.
This value is not documented anywhere in the README, `docs/api.md`, or the `/health`
endpoint. Callers that pass the WAV to tools expecting a specific rate (e.g. `ffmpeg`
pipelines configured for 44100 Hz, or Rhubarb lip-sync configured for 22050 Hz) will
silently get audio at the wrong playback speed unless they explicitly resample.

**Requested additions:**
1. Document `model.sr` (e.g. `24000 Hz`) in `docs/api.md` under response format.
2. Include `"sample_rate": model.sr` in the `/health` response body.
3. (Optional) Accept an optional `sample_rate` request parameter; if provided and
   different from `model.sr`, resample before returning. This lets clients avoid a
   separate `ffmpeg` resampling step.

---

## Implementation Order

The following sequence is logically constrained by dependencies and risk:

| Step | Issue | Why this order |
|------|-------|---------------|
| 1 | **BUG-1 + BUG-2** | Highest severity. BUG-2 is addressed as part of BUG-1 — `get_running_loop()` is used when adding `run_in_executor`. These two are done together in a single commit. |
| 2 | **FEAT-4** | No dependencies. Pure addition of `app/models.py`. Unblocks video-agent client code immediately. |
| 3 | **FEAT-5** | Depends on BUG-1 (async endpoint must exist to add headers cleanly). Adds zero-cost metadata to the response. |
| 4 | **FEAT-6** | Depends on FEAT-5 (health endpoint already touched; `/health` and `docs/api.md` updated in the same pass). Mostly documentation. |
| 5 | **BUG-3** | Documentation-only. Done last since the thread-safety constraint is already captured in the issue doc and integration plan. Low risk of regression. |

### Detailed issue docs

Each issue has a dedicated design doc under `docs/issues/`:

- [`docs/issues/BUG-1-async-endpoint-lock.md`](issues/BUG-1-async-endpoint-lock.md)
- [`docs/issues/BUG-2-asyncio-get-running-loop.md`](issues/BUG-2-asyncio-get-running-loop.md)
- [`docs/issues/BUG-3-thread-safety-docs.md`](issues/BUG-3-thread-safety-docs.md)
- [`docs/issues/FEAT-4-tts-request-dataclass.md`](issues/FEAT-4-tts-request-dataclass.md)
- [`docs/issues/FEAT-5-response-headers.md`](issues/FEAT-5-response-headers.md)
- [`docs/issues/FEAT-6-sample-rate-docs.md`](issues/FEAT-6-sample-rate-docs.md)

Each doc contains: problem description, implementation plan with code, and a testing plan
with fixture texts drawn from the video-agent WW2 tanks script (`tests/fixtures/ww2_tanks_segments.json`).

---

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
