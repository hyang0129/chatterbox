# Chatterbox Turbo TTS

Dev container for running [Chatterbox Turbo TTS](https://github.com/resemble-ai/chatterbox) locally — a 350M-parameter, single-step text-to-speech model from Resemble AI.

Supports two usage modes:
- **Server** — FastAPI + Uvicorn HTTP API (`POST /tts`, voice cloning, voice blending)
- **Serverless** — call the model directly from Python

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **WSL 2 backend**
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (enables `--gpus all`)
- VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- **NVIDIA GPU with ≥6 GB VRAM and CUDA capability sm_89+** (required — CPU inference is not supported)
  - Tested on RTX 5070 Ti Laptop (Blackwell sm_120, 12 GB VRAM)
  - Blackwell (RTX 50xx) requires the cu128 PyTorch build — handled automatically

> **GPU required.** This project does not support CPU-only inference. Both the server and
> serverless modes call `ChatterboxTurboTTS.from_pretrained(device="cuda")` and require a
> CUDA-capable GPU. Running on CPU will fail at model load time.

## Setup

### 1. HuggingFace token

The model weights are gated on HuggingFace. Create a read token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then add it to a
`.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

`.env` is gitignored and loaded automatically by the dev environment and test suite.

### 2. Open in dev container

Open this folder in VS Code → `Ctrl+Shift+P` → *Dev Containers: Reopen in Container*.

The container builds with CUDA 12.8, Python 3.11, PyTorch cu128, and all dependencies.
On first model use, weights (~1.5 GB) are downloaded and cached in a named Docker volume
(`chatterbox-hf-cache`) that persists across rebuilds.

> **Note:** If you are running outside the dev container, install dependencies into a
> virtual environment rather than the system Python:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

## Server mode

Start the API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### HTTPS (local dev)

A self-signed certificate is generated automatically by the dev container's
`postCreateCommand`. To start the server with HTTPS:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    --ssl-keyfile certs/localhost-key.pem \
    --ssl-certfile certs/localhost.pem
```

To regenerate the certificate manually:

```bash
./scripts/generate-cert.sh
```

Clients can bypass the self-signed cert warning with `curl -k` or by
setting `verify=False` in Python's `httpx`/`requests`.

### Synthesize speech

The server ships with a default voice (`kronimi7030`) — no setup required:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}' \
  --output output.wav
```

Use a specific voice by passing its ID:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice": "sarah-warm"}' \
  --output output.wav
```

### Clone a voice

Register a new voice from a reference audio file. The server automatically
calibrates the voice's speaking rate (WPM) after registration:

```bash
curl -X POST http://localhost:8000/voices/clone \
  -F "name=Sarah Warm" \
  -F "reference_audio=@sarah_sample.wav"
# → {"voice_id": "sarah-warm", "name": "Sarah Warm", "wpm": 142.5}
```

### Blend voices

Create a novel voice by blending two registered voices. Instead of cloning a single
speaker (which raises legal concerns), blending interpolates the vocal characteristics
of two sources to produce a third voice that never existed.

A voice has two independent components:

- **Voice texture** — the physical qualities of the vocal apparatus: pitch range,
  formant structure, breathiness, nasality, warmth. Blended on a 0–100 scale.
- **Voice rhythm** — the psychological delivery style: pacing, cadence, emphasis
  patterns. Always taken from the first voice (`voice_a`). To use the other voice's
  rhythm, swap the order.

```bash
# Blend: 70% James texture + 30% Sarah texture, with Sarah's rhythm
curl -X POST http://localhost:8000/voices/blend \
  -F "name=Sarah James 70" \
  -F "voice_a=sarah" \
  -F "voice_b=james" \
  -F "texture_mix=70"
```

`texture_mix` controls the blend ratio: `0` = pure voice A, `100` = pure voice B.

### Voice management

```bash
# List all registered voices (includes wpm for each)
curl http://localhost:8000/voices

# Get details for a single voice
curl http://localhost:8000/voices/sarah-warm

# Delete a voice
curl -X DELETE http://localhost:8000/voices/sarah-warm
```

See [docs/api.md](docs/api.md) for all endpoints, parameters, and response formats.

## Serverless mode

Use the model directly from Python (no server required):

> **Thread safety:** A single `ChatterboxTurboTTS` instance must not be used from multiple
> threads concurrently. If you need concurrent synthesis, create one model instance per
> thread, or serialise all `generate()` calls behind a `threading.Lock`. On a single GPU
> with limited VRAM, serial single-instance use is strongly recommended.
>
> Pipeline callers using a direct (in-process) backend must run all `generate()` calls
> serially — do not share one model instance across a `ThreadPoolExecutor`.

```python
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
wav = model.generate("Hello, how are you?")
# model.sr is the native output sample rate (24000 Hz).
# Pass it to sf.write so the WAV header is correct.
sf.write("output.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

> **Output sample rate:** `model.sr` is 24000 Hz. If downstream tools expect a different
> rate (e.g. 44100 Hz for broadcast, 22050 Hz for Rhubarb lip-sync), resample before writing:
>
>     ffmpeg -i output.wav -ar 22050 output_22050.wav

### Voice cloning (serverless)

```python
wav = model.generate(
    "Hello, how are you?",
    audio_prompt_path="reference.wav",
    temperature=0.8,
    top_p=0.95,
)
sf.write("output.wav", wav.squeeze(0).cpu().numpy(), model.sr)
```

## Paralinguistic tags

Inline emotion and sound tags are supported in both modes:

```
Hello [chuckle], how are you? [sigh] I'm doing fine.
```

Available: `[cough]`, `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`

## Testing

```bash
# Unit tests (mocked model, no GPU required)
pytest

# GPU integration tests (real model, saves audio to tests/integration/artifacts/)
pytest tests/integration/ -v -s
```

The GPU tests require a CUDA-capable GPU. `HF_TOKEN` in `.env` is needed for the initial model download but the tests do not skip if it is absent.

## Project structure

```
app/
  main.py               # FastAPI server (POST /tts, /voices/clone, /voices/blend, etc.)
  voices.py             # VoiceStore — file-based voice reference storage
tests/
  conftest.py           # Mocked model fixtures for unit tests
  test_api.py           # TTS synthesis + response header tests
  test_voices.py        # Voice clone, blend, list, delete endpoint tests
  test_blend.py         # _blend_conditionals unit tests
  fixtures/             # Shared test data (ten_second_script.json, nimivoice.mp3)
  integration/
    conftest.py         # Real model fixtures for GPU tests
    artifacts/          # Generated WAV files saved here for review
voices/
  kronimi7030/           # Default shipped voice (70/30 kronii-nimi blend)
scripts/
  generate-cert.sh      # Self-signed TLS cert generator (idempotent)
certs/                  # Generated .pem files (gitignored)
.devcontainer/
  devcontainer.json     # GPU passthrough, port 8000, HF cache + voices volumes
  Dockerfile            # CUDA 12.8 + Python 3.11
.env                    # HF_TOKEN (gitignored — create this yourself)
requirements.txt        # Python dependencies
pyproject.toml          # Project metadata, ruff, pytest config
docs/
  api.md                # Full API reference
```

## Notes

- Model weights are cached in a named Docker volume (`chatterbox-hf-cache`) and persist across container rebuilds.
- Registered voices are stored in a named Docker volume (`chatterbox-voices`) and persist across container rebuilds. Set `CHATTERBOX_VOICES_DIR` to override the storage path.
- `--shm-size 8g` is set in the dev container config to avoid PyTorch shared-memory issues.
- The Turbo model uses ~3–4 GB VRAM at inference time.
- `soundfile` is used for WAV encoding (`torchaudio.save` dropped BytesIO support in v2.9).
