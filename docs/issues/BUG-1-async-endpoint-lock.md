---
id: BUG-1
title: "/tts endpoint is sync — asyncio.Lock cannot be used, concurrent requests race on the GPU"
severity: High
status: open
---

## Problem

`app/main.py::synthesize` is declared `def synthesize` (synchronous). FastAPI dispatches
sync routes to a `ThreadPoolExecutor`, so two concurrent HTTP requests can both call
`model.generate()` at the same time on the same GPU model instance.

`ChatterboxTurboTTS` is not thread-safe. Concurrent `generate()` calls on a shared GPU
model risk:

- **CUDA OOM** — overlapping allocations exhaust VRAM
- **Corrupted/non-deterministic audio** — shared internal state mutated across threads
- **Silent failure** — no exception is raised in the common case; the caller receives bad audio

Additionally, because the endpoint is `def` (sync), an `asyncio.Lock` placed on `app.state`
is unreachable — you cannot `await` inside a sync function, so the lock is silently never
acquired even if one is added to the lifespan.

### Current code (broken)

```python
# app/main.py — current (sync, no serialisation)
@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
def synthesize(req: TTSRequest) -> Response:
    model: ChatterboxTurboTTS = app.state.model
    wav = model.generate(req.text, ...)
    return Response(content=_encode_wav(wav, model.sr), media_type="audio/wav")
```

## Implementation Plan

### Step 1 — Add `asyncio.Lock` to app lifespan

```python
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    app.state.lock = asyncio.Lock()   # <-- add this
    yield
```

### Step 2 — Convert `synthesize` to `async def`

```python
@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize(req: TTSRequest) -> Response:
    model: ChatterboxTurboTTS = app.state.model
    async with app.state.lock:
        wav = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: model.generate(
                req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            ),
        )
    return Response(content=_encode_wav(wav, model.sr), media_type="audio/wav")
```

`run_in_executor` moves the blocking GPU work off the event loop thread so uvicorn
can still accept and queue new connections while inference runs. The `asyncio.Lock`
ensures only one `generate()` is active at a time.

### Step 3 — Apply the same pattern to `/tts/clone`

`/tts/clone` is already `async def` (for the `UploadFile` await) but has no lock.
Wrap its `model.generate()` call in the same `async with app.state.lock` block and
move the generate call into `run_in_executor`.

```python
@app.post("/tts/clone", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize_clone(...) -> Response:
    ...
    async with app.state.lock:
        wav = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: model.generate(text, audio_prompt_path=tmp_path, ...),
        )
    ...
```

## Testing Plan

### Unit tests (no GPU, mocked model)

Add `tests/test_concurrency.py`:

1. **Lock is acquired during synthesis** — mock `app.state.lock` as a real `asyncio.Lock`,
   fire two concurrent `TestClient` requests, assert only one `generate()` runs at a time
   (second call is queued).
2. **`generate()` is called via executor** — patch `asyncio.get_running_loop` and assert
   `run_in_executor` is invoked.
3. **Response still valid after async conversion** — POST `/tts` with the standard fixture
   and assert 200, `audio/wav` content-type, valid WAV bytes.

### Fixture texts (from video-agent WW2 tanks script)

```json
[
  "Did you know Allied soldiers called any enemy tank a Tiger — regardless of its actual type — because the Tiger I was so feared?",
  "The American M4 Sherman was produced over 49,000 times — winning through sheer quantity over German armor quality.",
  "The Soviet T-34 shocked German engineers so much they considered copying its sloped armor for their own future tank designs."
]
```

These represent the exact segment texts the video-agent pipeline sends to `/tts`. All
three should return 200 with valid WAV bytes after the fix.

### Integration test (GPU required)

See `tests/integration/test_gpu_server.py`. After BUG-1 is fixed, add a concurrent
request test that fires 3 requests in rapid succession and asserts all return 200 with
non-empty WAV (no CUDA OOM). Use `asyncio.gather` or `concurrent.futures.ThreadPoolExecutor`
from the test side.
