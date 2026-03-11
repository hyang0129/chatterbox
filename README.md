# Chatterbox Turbo TTS

Dev container for running [Chatterbox Turbo TTS](https://github.com/resemble-ai/chatterbox) locally — a 350M-parameter, single-step text-to-speech model from Resemble AI.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **WSL 2 backend**
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (enables `--gpus all`)
- VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- NVIDIA GPU with ≥6 GB VRAM (tested on RTX 4070 Ti Laptop)

## Quick start

1. **Open in dev container**
   Open this folder in VS Code → `Ctrl+Shift+P` → *Dev Containers: Reopen in Container*. The container builds with CUDA 12.4, Python 3.11, and all dependencies.

2. **Run the Gradio app**
   ```bash
   python gradio_tts_turbo_app.py
   ```
   Open `http://localhost:7860` in your browser. On first run, model weights (~1.5 GB) are downloaded from HuggingFace and cached in a Docker volume.

3. **Or use Python directly**
   ```python
   from chatterbox.tts_turbo import ChatterboxTurboTTS
   import torchaudio

   model = ChatterboxTurboTTS.from_pretrained(device="cuda")
   wav = model.generate("Hello, how are you?")
   torchaudio.save("output.wav", wav, model.sr)
   ```

## Voice cloning

Provide a reference WAV file (>5 seconds recommended):

```python
wav = model.generate(
    "Hello, how are you?",
    audio_prompt_path="reference.wav",
    temperature=0.8,
    top_p=0.95,
)
torchaudio.save("output.wav", wav, model.sr)
```

## Paralinguistic tags

The Turbo model supports inline emotion/sound tags:

```
Hello [chuckle], how are you? [sigh] I'm doing fine.
```

Available: `[cough]`, `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[shush]`, `[clear throat]`

## Project structure

```
.devcontainer/
  devcontainer.json    # Dev container config (GPU, port forwarding, HF cache volume)
  Dockerfile           # CUDA 12.4 + Python 3.11 image
requirements.txt       # Dependencies
pyproject.toml         # Project metadata & ruff config
CLAUDE.md              # AI assistant context
```

## Notes

- Model weights are cached in a named Docker volume (`chatterbox-hf-cache`) so they persist across container rebuilds.
- `--shm-size 8g` is set in the dev container config to avoid PyTorch shared-memory issues.
- The 4070 Ti Laptop (8 GB VRAM) handles the Turbo model (~3-4 GB VRAM usage) comfortably.
