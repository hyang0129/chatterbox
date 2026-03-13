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

Synthesize speech from text using a registered voice. Returns a WAV audio file.

**Request body** (JSON)

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize. 1–5000 characters, non-whitespace. |
| `voice` | string | `"kronimi7030"` | Voice ID. See `GET /voices` for available IDs. |
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
| `X-Voice-WPM` | `142.5` | Estimated words-per-minute for this voice (only present if the voice has been calibrated) |

**Example**

```bash
# Use the default voice (kronimi7030)
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello [chuckle], how are you today?"}' \
  --output output.wav

# Use a specific registered voice
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice": "sarah-warm"}' \
  --output output.wav
```

---

## POST /voices/clone

Register a new voice by cloning from a reference audio file.
Accepts `multipart/form-data`.

**Form fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Human-readable name (1–200 chars). Auto-slugified into a `voice_id`. |
| `reference_audio` | file | required | Reference audio file (WAV, MP3, FLAC, OGG). 3–300 seconds, ≥16 kHz, <50 MB. Non-WAV formats are auto-converted via ffmpeg. |
| `max_duration_s` | float | `300` | Override max reference duration (3–7200s). |

**Response** — `201 Created`

```json
{
  "voice_id": "sarah-warm",
  "name": "Sarah Warm",
  "wpm": 142.5
}
```

After saving the reference audio, the server synthesizes a calibration passage
with the new voice and measures the output duration to estimate words-per-minute.

**Errors:** `409` if a voice with the same slug already exists, `422` if audio is invalid.

**Example**

```bash
curl -X POST http://localhost:8000/voices/clone \
  -F "name=Sarah Warm" \
  -F "reference_audio=@sarah_sample.wav"
```

---

## POST /voices/blend

Create a new voice by blending two existing voices. The blended voice is saved
and can be used with `POST /tts` like any other voice.

A voice has two independent components:

- **Voice texture** — the physical qualities of the vocal apparatus: pitch range,
  formant structure, breathiness, nasality, warmth. Controlled by `texture_mix`.
- **Voice rhythm** — the psychological delivery style: pacing, cadence, emphasis
  patterns. Always taken from `voice_a` (the first voice). To use the other
  voice's rhythm, swap the order.

Accepts `multipart/form-data`.

**Form fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | required | Human-readable name for the new voice. |
| `voice_a` | string | required | Voice ID for the first source (supplies rhythm). |
| `voice_b` | string | required | Voice ID for the second source. |
| `texture_mix` | int | `50` | Texture blend: `0` = pure voice A, `100` = pure voice B. |

**Response** — `201 Created`

```json
{
  "voice_id": "custom-blend",
  "name": "Custom Blend",
  "wpm": 138.2
}
```

After blending, the server synthesizes a calibration passage with the new voice
to estimate words-per-minute.

**Errors:** `404` if either source voice is not found, `409` if the name already exists.

**Example**

```bash
# First, clone two voices
curl -X POST http://localhost:8000/voices/clone \
  -F "name=Sarah" -F "reference_audio=@sarah.wav"

curl -X POST http://localhost:8000/voices/clone \
  -F "name=James" -F "reference_audio=@james.wav"

# Blend: 30% Sarah texture + 70% James texture, with Sarah's rhythm
curl -X POST http://localhost:8000/voices/blend \
  -F "name=Sarah James 70" \
  -F "voice_a=sarah" \
  -F "voice_b=james" \
  -F "texture_mix=70"
```

---

## GET /voices

List all registered voices.

**Response** — `200 OK`

```json
{
  "voices": [
    {
      "voice_id": "kronimi7030",
      "name": "KronNimi 70/30",
      "original_filename": "blend:kroniivoice.mp3+nimivoice.mp3",
      "created_at": "2026-03-13T03:36:02Z",
      "duration_s": 0.0,
      "sample_rate": 24000,
      "wpm": 142.5
    }
  ]
}
```

The `wpm` field is `null` for voices that were registered before WPM calibration
was introduced (legacy voices).

---

## GET /voices/{voice_id}

Get details for a single voice.

**Response** — `200 OK`

```json
{
  "voice_id": "sarah-warm",
  "name": "Sarah Warm",
  "original_filename": "sarah_sample.wav",
  "created_at": "2026-03-13T12:00:00Z",
  "duration_s": 8.5,
  "sample_rate": 24000,
  "wpm": 142.5
}
```

**Errors:** `404` if the voice does not exist.

**Example**

```bash
curl http://localhost:8000/voices/sarah-warm
```

---

## DELETE /voices/{voice_id}

Remove a registered voice and its data from disk.

**Response** — `204 No Content` on success, `404` if not found.

**Example**

```bash
curl -X DELETE http://localhost:8000/voices/sarah-warm
```

---

## Paralinguistic tags

All synthesis supports inline paralinguistic tags embedded in the `text` field.
These insert non-speech vocalisations into the generated audio at the position
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
