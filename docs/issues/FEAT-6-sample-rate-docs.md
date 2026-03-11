---
id: FEAT-6
title: "Document the native output sample rate (model.sr) in API reference, README, and /health"
severity: Low
status: open
---

## Problem

`ChatterboxTurboTTS` outputs audio at its native sample rate, accessible as `model.sr`
(expected: 24000 Hz). This value is not documented anywhere in the README, `docs/api.md`,
or the `/health` endpoint.

Callers that pass the WAV to tools expecting a specific rate get silent, hard-to-debug
failures:

- `ffmpeg` pipelines configured for 44100 Hz → audio plays at wrong speed
- Rhubarb lip-sync (`RHUBARB_WAV_SAMPLE_RATE = 22050` in video-agent config) reads segment
  WAVs directly → pitch and timing errors without any error message
- `AUDIO_SAMPLE_RATE = 44100` in `src/config.py` → the video-agent's FFmpeg master-build
  step resamples silently, but Rhubarb receives the original 24 kHz WAV segment

The only way callers currently discover the sample rate is by parsing the WAV header or
reading the source code.

## Implementation Plan

### 1. Update `/health` to include `sample_rate`

```python
@app.get("/health")
async def health() -> dict:
    model = getattr(app.state, "model", None)
    return {
        "status": "ok",
        "model": "chatterbox-turbo",
        "sample_rate": model.sr if model else None,
    }
```

### 2. Update `docs/api.md` — response format section

Under the `POST /tts` response format, add:

```
**Response format:** `audio/wav` — PCM 16-bit, mono, 24000 Hz

The WAV is encoded at the model's native sample rate (24000 Hz). If your downstream
tool expects a different rate (e.g. 44100 Hz for broadcast, 22050 Hz for Rhubarb
lip-sync), resample with ffmpeg:

    ffmpeg -i input.wav -ar 22050 output_22050.wav

The `X-Sample-Rate` response header (added in FEAT-5) also carries this value so
clients can adapt without hardcoding 24000.
```

### 3. Update README serverless section

After the code example, add:

```markdown
The model outputs at its native sample rate (`model.sr`, typically 24000 Hz).
If your downstream tools expect a different rate, resample before writing:

    import librosa
    wav_resampled = librosa.resample(wav.squeeze(0).cpu().numpy(), orig_sr=model.sr, target_sr=22050)
    sf.write("output.wav", wav_resampled, 22050)
```

### 4. (Optional) Accept `sample_rate` request parameter

Accept an optional `sample_rate` field in `TTSRequest`. If provided and different from
`model.sr`, resample the output tensor with `torchaudio.functional.resample` before
encoding. This avoids a client-side `ffmpeg` step.

```python
class TTSRequest(BaseModel):
    ...
    sample_rate: int | None = Field(None, ge=8000, le=48000)
```

```python
if req.sample_rate and req.sample_rate != model.sr:
    import torchaudio
    wav = torchaudio.functional.resample(wav, model.sr, req.sample_rate)
    output_sr = req.sample_rate
else:
    output_sr = model.sr
```

This is marked optional — the video-agent currently handles resampling via FFmpeg in
the master-build step.

## Testing Plan

### Unit tests — add to `tests/test_api.py`

```python
class TestHealthSampleRate:
    def test_health_includes_sample_rate(self, client):
        r = client.get("/health")
        body = r.json()
        assert "sample_rate" in body
        assert body["sample_rate"] == 24000  # MOCK_SR from conftest

    def test_health_sample_rate_is_integer(self, client):
        r = client.get("/health")
        assert isinstance(r.json()["sample_rate"], int)
```

### Documentation review

1. `docs/api.md` mentions 24000 Hz under the `/tts` response format section
2. `README.md` serverless section mentions `model.sr`
3. `/health` JSON body includes `"sample_rate": 24000`

### Video-agent integration note

The video-agent `RHUBARB_WAV_SAMPLE_RATE = 22050` means Rhubarb reads segment WAVs at
22050 Hz. The chatterbox server outputs at 24000 Hz. The integration plan addresses this
by resampling in `ChatterboxServerBackend.synthesize()` before writing the segment WAV —
not by relying on the server. FEAT-6 simply ensures the sample rate is discoverable so
future callers don't silently misconfigure.
