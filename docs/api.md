# API Reference

Base URL: `http://localhost:8000`

---

## GET /health

Returns server status.

**Response**
```json
{"status": "ok", "model": "chatterbox-turbo"}
```

---

## POST /tts

Synthesize speech from text. Returns a WAV audio file.

**Request body** (JSON)

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesize. 1–5000 characters, non-whitespace. |
| `temperature` | float | `0.8` | Sampling temperature. Range: `0.1`–`2.0`. |
| `top_p` | float | `0.95` | Nucleus sampling probability. Range: `0.0`–`1.0`. |
| `top_k` | int | `1000` | Top-k sampling. Range: `1`–`5000`. |
| `repetition_penalty` | float | `1.2` | Repetition penalty. Range: `1.0`–`3.0`. |

**Response**

`Content-Type: audio/wav` — mono, 24 kHz, 32-bit float PCM.

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
| `reference_audio` | file | required | Reference WAV file. ≥5 seconds recommended. |
| `temperature` | float | `0.8` | Sampling temperature. Range: `0.1`–`2.0`. |
| `top_p` | float | `0.95` | Nucleus sampling probability. Range: `0.0`–`1.0`. |
| `top_k` | int | `1000` | Top-k sampling. Range: `1`–`5000`. |
| `repetition_penalty` | float | `1.2` | Repetition penalty. Range: `1.0`–`3.0`. |

**Response**

`Content-Type: audio/wav` — mono, 24 kHz, 32-bit float PCM.

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
