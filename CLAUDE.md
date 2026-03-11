# Chatterbox Turbo TTS

## Project overview
Dev container for running [Chatterbox Turbo TTS](https://github.com/resemble-ai/chatterbox) (350M-param single-step decoder) locally on an NVIDIA GPU.

## Architecture
- **Dev container**: NVIDIA CUDA 12.4 runtime on Ubuntu 22.04, Python 3.11.
- **Inference**: Use the `chatterbox-tts` package directly — either via Python scripts or the built-in Gradio apps.

## Key commands
```bash
# Run the Turbo Gradio app (port 7860)
python -m gradio_tts_turbo_app

# Or use Python directly
python -c "
from chatterbox.tts_turbo import ChatterboxTurboTTS
import torchaudio
model = ChatterboxTurboTTS.from_pretrained(device='cuda')
wav = model.generate('Hello, world!')
torchaudio.save('output.wav', wav, model.sr)
"

# Lint
ruff check .
```

## Hardware target
- NVIDIA RTX 4070 Ti Laptop GPU (8 GB VRAM, Ada Lovelace / sm_89).
- The Turbo model (~350M params, single diffusion step) fits comfortably in ~3-4 GB VRAM.

## Conventions
- Python 3.11, type hints throughout.
- Ruff for linting (line-length 100).
- Model weights are auto-downloaded from HuggingFace on first run and cached in a Docker volume (`chatterbox-hf-cache`).

## Paralinguistic tags supported by Turbo
`[cough]`, `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`
