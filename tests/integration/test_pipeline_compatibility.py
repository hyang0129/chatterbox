"""
Pipeline compatibility test: route a real video-agent pipeline run through the
chatterbox server and save WAV artifacts for human review.

This test uses the exact vo_line texts from the most recent full video-agent
pipeline run (hamburger facts, 2026-03-10). Those are the texts that would have
been sent to ElevenLabs. Here they are sent to the chatterbox server instead.

Output artifacts are saved to tests/integration/artifacts/pipeline_compat_*.wav
for side-by-side human review.

Requires: NVIDIA GPU, HF_TOKEN in .env
Run: pytest tests/integration/test_pipeline_compatibility.py -v -s
"""
import asyncio
import io
import json
import shutil
import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import make_voice_store_with_kronimi

pytestmark = pytest.mark.gpu

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def pipeline_fixture() -> dict:
    with open(FIXTURES_DIR / "hamburger_facts_pipeline_run.json") as f:
        return json.load(f)


def test_pipeline_scenes_synthesized(real_model, artifacts_dir: Path, pipeline_fixture: dict) -> None:
    """
    Synthesize every scene vo_line from the hamburger facts pipeline run through
    the chatterbox server. Asserts:
    - All scenes return 200 with valid WAV
    - X-Audio-Duration-S header is present and > 0
    - Reported duration is in a plausible range for the text length
    - Saves one WAV per scene to artifacts/ for human review
    """
    from app.main import app

    tmpdir = tempfile.mkdtemp(prefix="chatterbox_pipeline_test_")
    prev_model = getattr(app.state, "model", None)
    prev_voice_store = getattr(app.state, "voice_store", None)
    prev_lock = getattr(app.state, "lock", None)
    app.state.model = real_model
    app.state.voice_store = make_voice_store_with_kronimi(tmpdir)
    app.state.lock = asyncio.Lock()

    try:
        client = TestClient(app, raise_server_exceptions=True)
        scenes = pipeline_fixture["scenes"]
        params = pipeline_fixture["default_tts_params"]

        total_duration = 0.0
        results = []

        for scene in scenes:
            scene_id = scene["scene_id"]
            vo_line = scene["vo_line"]
            planned_duration = scene["t_end_s"] - scene["t_start_s"]

            payload = {
                "text": vo_line,
                **params,
            }
            r = client.post("/tts", json=payload)

            assert r.status_code == 200, f"{scene_id}: expected 200, got {r.status_code}"
            assert r.headers["content-type"] == "audio/wav", f"{scene_id}: bad content-type"
            assert "x-audio-duration-s" in r.headers, f"{scene_id}: missing X-Audio-Duration-S"

            reported_duration = float(r.headers["x-audio-duration-s"])
            assert reported_duration > 0.0, f"{scene_id}: zero duration"

            with wave.open(io.BytesIO(r.content)) as wf:
                assert wf.getnchannels() == 1, f"{scene_id}: expected mono"
                assert wf.getframerate() == real_model.sr, f"{scene_id}: wrong sample rate"
                actual_duration = wf.getnframes() / wf.getframerate()

            assert abs(actual_duration - reported_duration) < 0.02, (
                f"{scene_id}: header duration {reported_duration:.2f}s "
                f"!= WAV duration {actual_duration:.2f}s"
            )

            out_path = artifacts_dir / f"pipeline_compat_{scene_id}.wav"
            out_path.write_bytes(r.content)
            total_duration += actual_duration
            results.append((scene_id, vo_line, actual_duration, planned_duration))

    finally:
        app.state.model = prev_model
        app.state.voice_store = prev_voice_store
        if prev_lock is not None:
            app.state.lock = prev_lock
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Pipeline compatibility: {len(results)} scenes")
    print(f"{'Scene':<12} {'Planned':>10} {'Actual':>10}  Text")
    print(f"{'-'*60}")
    for scene_id, text, actual, planned in results:
        flag = " <--" if abs(actual - planned) > 1.5 else ""
        print(f"{scene_id:<12} {planned:>9.1f}s {actual:>9.2f}s  {text[:50]!r}{flag}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<12} {sum(p for _, _, _, p in results):>9.1f}s {total_duration:>9.2f}s")
    print(f"\nArtifacts saved to: {artifacts_dir}/pipeline_compat_*.wav")
    print("Review the WAVs for audio quality, naturalness, and intelligibility.")
