# API Reference

Base URL: `http://localhost:8000`

---

## GET /health

Returns server status.

**Response**
```json
{"status": "ok", "model": "chatterbox-turbo", "sample_rate": 24000, "voices": 3}
```

---

## POST /tts

Synthesize speech from text. Returns a WAV audio file.

**Request body** (JSON)

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize. 1–5000 characters, non-whitespace. |
| `voice` | string | `null` | Registered voice ID. When set, clones that voice. See `GET /v1/voices`. |
| `temperature` | float | `0.8` | Sampling temperature. Range: `0.1`–`2.0`. |
| `top_p` | float | `0.95` | Nucleus sampling probability. Range: `0.0`–`1.0`. |
| `top_k` | int | `1000` | Top-k sampling. Range: `1`–`5000`. |
| `repetition_penalty` | float | `1.2` | Repetition penalty. Range: `1.0`–`3.0`. |

**Response**

`Content-Type: audio/wav` — mono, PCM 16-bit, **24000 Hz** (the model's native sample rate).

> **Sample rate:** The output is always at 24000 Hz (`model.sr`). If your downstream tool
> expects a different rate (e.g. 44100 Hz for broadcast, 22050 Hz for Rhubarb lip-sync),
> resample with ffmpeg:
>
>     ffmpeg -i output.wav -ar 22050 output_22050.wav

**Response headers**

| Header | Example | Description |
|--------|---------|-------------|
| `X-Audio-Duration-S` | `4.23` | Audio duration in seconds (two decimal places) |
| `X-Sample-Rate` | `24000` | Sample rate of the returned WAV |
| `X-Audio-Frames` | `101520` | Total PCM frame count |

**Example**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello [chuckle], how are you today?",
    "temperature": 0.8,
    "top_p": 0.95
  }' \
  --output output.wav
```

```python
import httpx

r = httpx.post("http://localhost:8000/tts", json={
    "text": "Hello, how are you?",
    "temperature": 0.8,
})
r.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(r.content)
```

---

## POST /tts/clone

Synthesize speech using a reference audio file for voice cloning.
Accepts `multipart/form-data`.

**Form fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize. 1–5000 characters. |
| `reference_audio` | file | required | Reference audio file (WAV, MP3, FLAC, OGG). ≥5 seconds recommended. Non-WAV formats are auto-converted via ffmpeg. |
| `temperature` | float | `0.8` | Sampling temperature. Range: `0.1`–`2.0`. |
| `top_p` | float | `0.95` | Nucleus sampling probability. Range: `0.0`–`1.0`. |
| `top_k` | int | `1000` | Top-k sampling. Range: `1`–`5000`. |
| `repetition_penalty` | float | `1.2` | Repetition penalty. Range: `1.0`–`3.0`. |

**Response**

`Content-Type: audio/wav` — mono, PCM 16-bit, 24000 Hz. Same `X-Audio-*` headers as
`POST /tts` (see above).

**Example**

```bash
curl -X POST http://localhost:8000/tts/clone \
  -F "text=Hello, this is a cloned voice." \
  -F "reference_audio=@reference.wav" \
  -F "temperature=0.8" \
  --output cloned.wav
```

```python
import httpx

with open("reference.wav", "rb") as ref:
    r = httpx.post("http://localhost:8000/tts/clone", data={
        "text": "Hello, this is a cloned voice.",
        "temperature": 0.8,
    }, files={
        "reference_audio": ("reference.wav", ref, "audio/wav"),
    })
r.raise_for_status()
with open("cloned.wav", "wb") as f:
    f.write(r.content)
```

---

## Voice Management

Register, list, and delete persistent voice references. Registered voices can be used
in `POST /tts` via the `voice` field, eliminating the need to upload reference audio
on every request.

Voice references are stored on disk in the directory configured by the
`CHATTERBOX_VOICES_DIR` environment variable (default: `./voices/`).

### POST /v1/voices

Register a new voice from a reference audio file. Accepts `multipart/form-data`.

**Form fields**

| Field | Type | Description |
|---|---|---|
| `name` | string | Human-readable name (1–200 chars). Auto-slugified into a `voice_id`. |
| `reference_audio` | file | Reference audio file (WAV, MP3, FLAC, OGG). 3–30 seconds, ≥16 kHz, <10 MB. Non-WAV formats are auto-converted via ffmpeg. |

**Response** — `201 Created`

```json
{
  "voice_id": "sarah-warm",
  "name": "Sarah Warm",
  "created_at": "2026-03-13T12:00:00Z"
}
```

**Errors:** `409` if a voice with the same slug already exists, `422` if audio is invalid.

**Example**

```bash
curl -X POST http://localhost:8000/v1/voices \
  -F "name=Sarah Warm" \
  -F "reference_audio=@sarah_sample.wav"
```

---

### GET /v1/voices

List all registered voices.

**Response** — `200 OK`

```json
{
  "voices": [
    {
      "voice_id": "sarah-warm",
      "name": "Sarah Warm",
      "original_filename": "sarah_sample.wav",
      "created_at": "2026-03-13T12:00:00Z",
      "duration_s": 8.5,
      "sample_rate": 24000
    }
  ]
}
```

---

### GET /v1/voices/{voice_id}

Get metadata for a single registered voice.

**Response** — `200 OK` (same shape as one element of the list above), or `404`.

---

### DELETE /v1/voices/{voice_id}

Remove a registered voice and its reference audio from disk.

**Response** — `204 No Content` on success, `404` if not found.

**Example**

```bash
curl -X DELETE http://localhost:8000/v1/voices/sarah-warm
```

---

## Paralinguistic tags

Both `/tts` and `/tts/clone` support inline paralinguistic tags embedded in the `text`
field. These insert non-speech vocalisations into the generated audio at the position
where they appear in the text.

**Supported tags**

| Tag | Sound |
|-----|-------|
| `[laugh]` | Laughter |
| `[chuckle]` | Soft, brief laugh |
| `[cough]` | Cough |
| `[sigh]` | Sigh |
| `[gasp]` | Sharp intake of breath |
| `[groan]` | Groan |
| `[sniff]` | Sniff |
| `[shush]` | Shushing sound |
| `[clear throat]` | Throat clearing |

**Usage**

Place tags inline where you want the sound to occur:

```
Hello [chuckle], how are you? [sigh] I'm doing fine.
```

Multiple tags can appear in a single request. Tags are case-sensitive and must use the
exact spelling shown above (lowercase, square brackets).

**Tips**

- Place tags between sentences or clauses for the most natural results.
- Tags at the very start or end of the text work but may sound less natural than
  mid-sentence placement.
- Surrounding punctuation (commas, periods) near a tag can help the model produce
  more natural pacing.

---

## Error responses

All validation errors return HTTP `422 Unprocessable Entity` with a JSON body describing the failed field(s):

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```
