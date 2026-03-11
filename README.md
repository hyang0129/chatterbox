# Chatterbox Turbo TTS

Dev container for running [Chatterbox Turbo TTS](https://github.com/resemble-ai/chatterbox) locally — a 350M-parameter, single-step text-to-speech model from Resemble AI.

Supports two usage modes:
- **Server** — FastAPI + Uvicorn HTTP API (`POST /tts`, `POST /tts/clone`)
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

### Basic synthesis

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}' \
  --output output.wav
```

### Voice cloning

```bash
curl -X POST http://localhost:8000/tts/clone \
  -F "text=Hello, how are you?" \
  -F "reference_audio=@reference.wav" \
  --output output.wav
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

The GPU tests require `HF_TOKEN` to be set in `.env` and a CUDA-capable GPU.

## Project structure

```
app/
  main.py               # FastAPI server (POST /tts, POST /tts/clone)
tests/
  conftest.py           # Mocked model fixtures for unit tests
  fixtures/             # Shared test data (ten_second_script.json)
  integration/
    conftest.py         # Real model fixtures for GPU tests
    artifacts/          # Generated WAV files saved here for review
.devcontainer/
  devcontainer.json     # GPU passthrough, port 8000, HF cache volume
  Dockerfile            # CUDA 12.8 + Python 3.11
.env                    # HF_TOKEN (gitignored — create this yourself)
requirements.txt        # Python dependencies
pyproject.toml          # Project metadata, ruff, pytest config
docs/
  api.md                # Full API reference
```

## Notes

- Model weights are cached in a named Docker volume (`chatterbox-hf-cache`) and persist across container rebuilds.
- `--shm-size 8g` is set in the dev container config to avoid PyTorch shared-memory issues.
- The Turbo model uses ~3–4 GB VRAM at inference time.
- `soundfile` is used for WAV encoding (`torchaudio.save` dropped BytesIO support in v2.9).
