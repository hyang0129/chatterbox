# Chatterbox Turbo TTS

## Project overview
Dev container for running [Chatterbox Turbo TTS](https://github.com/resemble-ai/chatterbox) (350M-param single-step decoder) locally on an NVIDIA GPU.

## Architecture
- **Dev container**: NVIDIA CUDA 12.8 runtime on Ubuntu 22.04, Python 3.11.
- **Inference**: FastAPI + Uvicorn HTTP API; use the `chatterbox-tts` package directly for TTS generation.

## Key commands
```bash
# Run the FastAPI server (port 8000)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or use Python directly
python -c "
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device='cuda')
wav = model.generate('Hello, world!')
sf.write('output.wav', wav.squeeze(0).cpu().numpy(), model.sr)
"

# Lint
ruff check .

# Unit tests (mocked, no GPU)
pytest

# GPU integration tests (real model, saves audio to tests/integration/artifacts/)
pytest tests/integration/ -v -s
```

## Hardware target
- NVIDIA RTX 5070 Ti Laptop GPU (12 GB VRAM, Blackwell / sm_120).
- Requires PyTorch cu128+ (CUDA 12.8 base image) for sm_120 kernel support.
- The Turbo model (~350M params, single diffusion step) fits comfortably in ~3-4 GB VRAM.

## Conventions
- Python 3.11, type hints throughout.
- Ruff for linting (line-length 100).
- Model weights are auto-downloaded from HuggingFace on first run and cached in a Docker volume (`chatterbox-hf-cache`).

## Paralinguistic tags supported by Turbo
`[cough]`, `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`
