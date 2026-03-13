"""
Integration test: clone nimivoice from MP3, then synthesize all hamburger facts VO lines.

Requires:
- CUDA GPU with the model loaded
- HF_TOKEN set in .env
- Server NOT required — uses the model directly

Saves WAV artifacts to tests/integration/artifacts/hamburger_voice_clone/ for human review.

Usage:
    pytest tests/integration/test_voice_clone_hamburger.py -v -s
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import soundfile as sf
import torch

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "hamburger_voice_clone"

REFERENCE_MP3 = FIXTURES_DIR / "nimivoice.mp3"
PIPELINE_JSON = FIXTURES_DIR / "hamburger_facts_pipeline_run.json"


def _patch_norm_loudness():
    """Monkey-patch ChatterboxTurboTTS.norm_loudness to preserve float32 dtype.

    Upstream bug: norm_loudness multiplies a float32 numpy array by a Python float,
    which upcasts to float64. This causes a dtype mismatch in s3tokenizer's
    log_mel_spectrogram (mel_filters is float32, magnitudes becomes float64).
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    _orig = ChatterboxTurboTTS.norm_loudness

    def _patched(self, wav, sr, target_lufs=-27):
        orig_dtype = wav.dtype
        result = _orig(self, wav, sr, target_lufs)
        if result.dtype != orig_dtype:
            result = result.astype(orig_dtype)
        return result

    ChatterboxTurboTTS.norm_loudness = _patched


@pytest.fixture(scope="module")
def real_model():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    import os
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    _patch_norm_loudness()
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    return ChatterboxTurboTTS.from_pretrained(device="cuda")


@pytest.fixture(scope="module")
def reference_wav_path(tmp_path_factory) -> Path:
    """Convert the MP3 reference to a temp WAV for the model."""
    import subprocess
    tmp_dir = tmp_path_factory.mktemp("voice_ref")
    wav_path = tmp_dir / "nimivoice.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(REFERENCE_MP3),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    return wav_path


@pytest.fixture(scope="module")
def pipeline_scenes() -> list[dict]:
    with open(PIPELINE_JSON) as f:
        data = json.load(f)
    return data["scenes"]


@pytest.mark.gpu
def test_clone_voice_hamburger_facts(
    real_model, reference_wav_path: Path, pipeline_scenes: list[dict]
) -> None:
    """Clone nimivoice and synthesize all hamburger facts VO lines."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_path = str(reference_wav_path)
    sr = real_model.sr
    results = []

    for scene in pipeline_scenes:
        scene_id = scene["scene_id"]
        text = scene["vo_line"]

        print(f"\n--- {scene_id}: {text[:60]}...")
        t0 = time.perf_counter()

        wav = real_model.generate(
            text,
            audio_prompt_path=ref_path,
            temperature=0.8,
            top_p=0.95,
            top_k=1000,
            repetition_penalty=1.2,
        )

        elapsed = time.perf_counter() - t0
        duration_s = wav.shape[-1] / sr

        out_path = ARTIFACTS_DIR / f"{scene_id}.wav"
        sf.write(str(out_path), wav.squeeze(0).cpu().numpy(), sr)

        results.append({
            "scene_id": scene_id,
            "text": text,
            "duration_s": round(duration_s, 2),
            "inference_s": round(elapsed, 2),
            "file": str(out_path),
        })

        print(f"    Duration: {duration_s:.2f}s | Inference: {elapsed:.2f}s | {out_path.name}")

    # Save a summary JSON alongside the WAVs.
    summary_path = ARTIFACTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {len(results)} scenes synthesized to {ARTIFACTS_DIR}/")
    print(f"    Summary: {summary_path}")

    # Basic assertions.
    assert len(results) == len(pipeline_scenes)
    for r in results:
        assert r["duration_s"] > 0.5, f"{r['scene_id']} audio too short"
        assert Path(r["file"]).exists()
