"""
End-to-end script synthesis test.

Submits the fixture script to POST /tts and writes the response WAV to
tests/artifacts/ for human review.
"""
import io
import json
from pathlib import Path

import pytest
import soundfile as sf
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


def test_synthesize_script(
    client: TestClient,
    mock_model,
    script_request: dict,
    artifacts_dir: Path,
    _kronimi_voice,
) -> None:
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
