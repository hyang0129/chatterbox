---
id: FEAT-5
title: "/tts should return synthesis metadata in response headers"
severity: Low
status: open
depends_on: BUG-1
---

## Problem

The `/tts` endpoint returns raw WAV bytes only. Clients that need the audio duration for
timeline alignment (e.g. to check whether a synthesised segment fits a planned scene
window) must shell out to `ffprobe` or parse the RIFF header themselves.

The video-agent pipeline (`_resolve_segment_duration_seconds`) uses the ElevenLabs
metadata dict's `estimated_duration_s` field. When switching to Chatterbox, the WAV is
available immediately and its duration is exact (not estimated). Including it in response
headers is zero-cost — all values are known before the response is sent — and eliminates
the need for a second tool invocation.

## Implementation Plan

### Response headers to add

| Header | Example | Notes |
|--------|---------|-------|
| `X-Audio-Duration-S` | `4.23` | Duration in seconds, two decimal places |
| `X-Sample-Rate` | `24000` | Native model sample rate (`model.sr`) |
| `X-Audio-Frames` | `101520` | Total PCM frame count (frames = duration * sr) |

### Changes to `_encode_wav` (or return alongside bytes)

Option A — return a tuple from `_encode_wav`:

```python
def _encode_wav(wav: torch.Tensor, sample_rate: int) -> tuple[bytes, int]:
    """Returns (wav_bytes, frame_count)."""
    buf = io.BytesIO()
    sf.write(buf, wav[0].cpu().numpy(), sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    data = buf.read()
    frame_count = wav.shape[-1]
    return data, frame_count
```

Option B — compute frame count from the tensor before encoding (simpler):

```python
frame_count = wav.shape[-1]
wav_bytes = _encode_wav(wav, model.sr)
duration_s = frame_count / model.sr
```

### Updated `synthesize` endpoint

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
    frame_count = wav.shape[-1]
    duration_s = frame_count / model.sr
    headers = {
        "X-Audio-Duration-S": f"{duration_s:.2f}",
        "X-Sample-Rate": str(model.sr),
        "X-Audio-Frames": str(frame_count),
    }
    return Response(
        content=_encode_wav(wav, model.sr),
        media_type="audio/wav",
        headers=headers,
    )
```

Apply the same headers to `/tts/clone` for consistency.

## Testing Plan

### Unit tests — add to `tests/test_api.py`

```python
class TestTTSResponseHeaders:
    def test_x_audio_duration_s_present(self, client, mock_model):
        r = client.post("/tts", json={"text": "Header check."})
        assert "x-audio-duration-s" in r.headers

    def test_x_audio_duration_s_is_float(self, client, mock_model):
        r = client.post("/tts", json={"text": "Duration float check."})
        val = float(r.headers["x-audio-duration-s"])
        assert val > 0.0

    def test_x_sample_rate_matches_model_sr(self, client, mock_model):
        r = client.post("/tts", json={"text": "Sample rate check."})
        assert r.headers["x-sample-rate"] == str(mock_model.sr)  # 24000

    def test_x_audio_frames_is_integer(self, client, mock_model):
        r = client.post("/tts", json={"text": "Frame count check."})
        frames = int(r.headers["x-audio-frames"])
        assert frames > 0

    def test_duration_consistency(self, client, mock_model):
        """frames / sample_rate must equal reported duration."""
        # mock returns 0.5s of silence at 24000 Hz = 12000 frames
        r = client.post("/tts", json={"text": "Consistency check."})
        sr = int(r.headers["x-sample-rate"])
        frames = int(r.headers["x-audio-frames"])
        duration = float(r.headers["x-audio-duration-s"])
        assert abs(frames / sr - duration) < 0.01

    def test_clone_also_returns_headers(self, client, mock_model):
        """POST /tts/clone should return the same metadata headers."""
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 24000)
        buf.seek(0)
        r = client.post(
            "/tts/clone",
            data={"text": "Clone header check."},
            files={"reference_audio": ("ref.wav", buf.read(), "audio/wav")},
        )
        assert "x-audio-duration-s" in r.headers
        assert "x-sample-rate" in r.headers
        assert "x-audio-frames" in r.headers
```

### Video-agent contract test

The video-agent `ChatterboxServerBackend.synthesize()` should read `X-Audio-Duration-S`
and include it in the returned metadata dict under `estimated_duration_s`. Add a
test in `tests/integration/test_gpu_server.py` that:

1. Posts a WW2 tanks segment text to `/tts`
2. Asserts `X-Audio-Duration-S` is present and > 0
3. Asserts the reported duration is within ±0.5s of `len(text)/20` (rough chars-per-second)
