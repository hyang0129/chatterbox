"""
Integration test: blend kronii + nimi voices (70/30 texture, nimi rhythm),
then synthesize all hamburger facts VO lines with the blended voice.

Requires:
- CUDA GPU with the model loaded
- HF_TOKEN set in .env
- Server NOT required — uses the model directly

Saves WAV artifacts to tests/integration/artifacts/hamburger_voice_blend/ for human review.

Usage:
    pytest tests/integration/test_voice_blend_hamburger.py -v -s
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import soundfile as sf
import torch

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts" / "hamburger_voice_blend"

NIMI_MP3 = FIXTURES_DIR / "nimivoice.mp3"
KRONII_MP3 = FIXTURES_DIR / "kroniivoice.mp3"
PIPELINE_JSON = FIXTURES_DIR / "hamburger_facts_pipeline_run.json"

# Blend parameters: 70% kronii texture + 30% nimi texture, nimi rhythm
TEXTURE_MIX = 70  # 0 = pure voice_a (kronii), 100 = pure voice_b (nimi)
RHYTHM_SOURCE = "voice_b"  # nimi supplies the rhythm


def _patch_norm_loudness():
    """Monkey-patch ChatterboxTurboTTS.norm_loudness to preserve float32 dtype."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    _orig = ChatterboxTurboTTS.norm_loudness

    def _patched(self, wav, sr, target_lufs=-27):
        orig_dtype = wav.dtype
        result = _orig(self, wav, sr, target_lufs)
        if result.dtype != orig_dtype:
            result = result.astype(orig_dtype)
        return result

    ChatterboxTurboTTS.norm_loudness = _patched


def _convert_mp3_to_wav(mp3_path: Path, tmp_dir: Path) -> Path:
    """Convert an MP3 reference to a temp WAV for the model."""
    wav_path = tmp_dir / f"{mp3_path.stem}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(mp3_path),
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
def real_model():
    import os
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    _patch_norm_loudness()
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    return ChatterboxTurboTTS.from_pretrained(device="cuda")


@pytest.fixture(scope="module")
def voice_wav_paths(tmp_path_factory) -> dict[str, Path]:
    """Convert both MP3 references to temp WAVs."""
    tmp_dir = tmp_path_factory.mktemp("blend_refs")
    return {
        "kronii": _convert_mp3_to_wav(KRONII_MP3, tmp_dir),
        "nimi": _convert_mp3_to_wav(NIMI_MP3, tmp_dir),
    }


@pytest.fixture(scope="module")
def pipeline_scenes() -> list[dict]:
    with open(PIPELINE_JSON) as f:
        data = json.load(f)
    return data["scenes"]


@pytest.mark.gpu
def test_blend_voice_hamburger_facts(
    real_model, voice_wav_paths: dict[str, Path], pipeline_scenes: list[dict]
) -> None:
    """Blend kronii+nimi (70/30 texture, nimi rhythm) and synthesize hamburger facts."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts_turbo import Conditionals

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    kronii_path = str(voice_wav_paths["kronii"])
    nimi_path = str(voice_wav_paths["nimi"])
    sr = real_model.sr

    # --- Extract conditionals from both voices ---
    print("\n=== Extracting voice conditionals ===")

    print("  Preparing kronii conditionals (voice_a)...")
    real_model.prepare_conditionals(kronii_path)
    conds_kronii = real_model.conds

    print("  Preparing nimi conditionals (voice_b)...")
    real_model.prepare_conditionals(nimi_path)
    conds_nimi = real_model.conds

    # --- Blend embeddings ---
    alpha = TEXTURE_MIX / 100.0  # 0.7 = 70% nimi texture
    print(f"\n=== Blending: {TEXTURE_MIX}% kronii(A)->nimi(B) texture, "
          f"rhythm from {'nimi' if RHYTHM_SOURCE == 'voice_b' else 'kronii'} ===")

    # T3 speaker embedding (256-dim)
    emb_a = conds_kronii.t3.speaker_emb.float()
    emb_b = conds_nimi.t3.speaker_emb.float()
    blended_t3 = (1.0 - alpha) * emb_a + alpha * emb_b
    blended_t3 = blended_t3 / blended_t3.norm(p=2, dim=-1, keepdim=True)

    # S3Gen x-vector embedding (192-dim)
    xvec_a = conds_kronii.gen["embedding"].float()
    xvec_b = conds_nimi.gen["embedding"].float()
    blended_xvec = (1.0 - alpha) * xvec_a + alpha * xvec_b
    blended_xvec = blended_xvec / blended_xvec.norm(p=2, dim=-1, keepdim=True)

    # Rhythm from nimi (voice_b)
    rhythm = conds_nimi if RHYTHM_SOURCE == "voice_b" else conds_kronii

    t3_cond = T3Cond(
        speaker_emb=blended_t3.to(dtype=emb_a.dtype),
        cond_prompt_speech_tokens=rhythm.t3.cond_prompt_speech_tokens,
        emotion_adv=rhythm.t3.emotion_adv,
    )
    gen_dict = dict(rhythm.gen)
    gen_dict["embedding"] = blended_xvec.to(dtype=xvec_a.dtype)

    real_model.conds = Conditionals(t3_cond, gen_dict)

    # Verify embeddings are L2-normalised
    t3_norm = blended_t3.norm(p=2).item()
    xvec_norm = blended_xvec.norm(p=2).item()
    print(f"  T3 embedding L2 norm: {t3_norm:.6f}")
    print(f"  S3Gen x-vector L2 norm: {xvec_norm:.6f}")
    assert abs(t3_norm - 1.0) < 1e-4
    assert abs(xvec_norm - 1.0) < 1e-4

    # --- Synthesize all scenes ---
    results = []

    for scene in pipeline_scenes:
        scene_id = scene["scene_id"]
        text = scene["vo_line"]

        print(f"\n--- {scene_id}: {text[:60]}...")
        t0 = time.perf_counter()

        wav = real_model.generate(
            text,
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

    # Save summary
    summary = {
        "blend_config": {
            "voice_a": "kronii",
            "voice_b": "nimi",
            "texture_mix": TEXTURE_MIX,
            "rhythm_source": RHYTHM_SOURCE,
            "description": "70% kronii / 30% nimi texture, nimi rhythm",
        },
        "scenes": results,
    }
    summary_path = ARTIFACTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {len(results)} scenes synthesized to {ARTIFACTS_DIR}/")
    print(f"    Summary: {summary_path}")

    # Assertions
    assert len(results) == len(pipeline_scenes)
    for r in results:
        assert r["duration_s"] > 0.5, f"{r['scene_id']} audio too short"
        assert Path(r["file"]).exists()
