"""
End-to-end script synthesis test.

Submits the fixture script to POST /tts and writes the response WAV to
tests/artifacts/ for human review.
"""
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from tests.conftest import MOCK_SR

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@pytest.fixture(scope="module")
def artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture(scope="module")
def script_request() -> dict:
    with open(FIXTURES_DIR / "ten_second_script.json") as f:
        return json.load(f)


def _create_precomputed_voice(voice_id: str) -> None:
    """Create a voice directory with conditionals.pt (no reference WAV)."""
    voices_dir = os.environ["CHATTERBOX_VOICES_DIR"]
    voice_dir = os.path.join(voices_dir, voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    meta = {
        "voice_id": voice_id,
        "name": voice_id,
        "original_filename": "blend",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": 0.0,
        "sample_rate": 24000,
    }
    with open(os.path.join(voice_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    torch.save({"dummy": True}, os.path.join(voice_dir, "conditionals.pt"))
    from app.main import app as _app
    _app.state.voice_store._scan()


def test_synthesize_script(
    client: TestClient,
    mock_model,
    script_request: dict,
    artifacts_dir: Path,
) -> None:
    _create_precomputed_voice("kronimi7030")
    response = client.post("/tts", json=script_request)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    # Validate WAV structure
    audio, sr = sf.read(io.BytesIO(response.content))
    assert sr == MOCK_SR
    assert audio.ndim == 1  # mono
    duration_s = len(audio) / sr
    assert duration_s > 0

    # Save for human review
    out_path = artifacts_dir / "ten_second_script.wav"
    out_path.write_bytes(response.content)
    print(f"\nSaved: {out_path}  ({len(response.content):,} bytes, {duration_s:.2f}s)")

    # Verify the exact text and params were forwarded to the model
    call = mock_model.generate.call_args
    assert call.args[0] == script_request["text"]
    assert call.kwargs["temperature"] == pytest.approx(script_request["temperature"])
    assert call.kwargs["top_p"] == pytest.approx(script_request["top_p"])
    assert call.kwargs["top_k"] == script_request["top_k"]
    assert call.kwargs["repetition_penalty"] == pytest.approx(script_request["repetition_penalty"])
