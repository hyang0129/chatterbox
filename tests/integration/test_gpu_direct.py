"""
GPU test 1: direct model render, no server components.

Verifies that the model loads on CUDA and produces real, non-silent audio
of a plausible duration for the given script.
"""
from pathlib import Path

import pytest
import soundfile as sf
import torch

pytestmark = pytest.mark.gpu


def test_gpu_direct_render(real_model, script_request: dict, artifacts_dir: Path) -> None:
    wav: torch.Tensor = real_model.generate(
        script_request["text"],
        temperature=script_request["temperature"],
        top_p=script_request["top_p"],
        top_k=script_request["top_k"],
        repetition_penalty=script_request["repetition_penalty"],
    )

    assert wav.ndim == 2, f"Expected 2-D tensor, got shape {wav.shape}"
    assert wav.shape[0] == 1, "Expected mono (1 channel)"

    duration_s = wav.shape[1] / real_model.sr
    assert duration_s >= 5.0, f"Audio too short for a ~10-second script: {duration_s:.2f}s"

    rms = wav.pow(2).mean().sqrt().item()
    assert rms > 1e-4, "Audio appears to be silence — model may not have generated speech"

    out_path = artifacts_dir / "gpu_direct_render.wav"
    sf.write(str(out_path), wav.squeeze(0).cpu().numpy(), real_model.sr)

    print(f"\nSaved : {out_path}")
    print(f"Shape : {tuple(wav.shape)}  |  Duration: {duration_s:.2f}s  |  RMS: {rms:.5f}")
