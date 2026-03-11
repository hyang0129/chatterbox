"""
GPU test 2: server-based render via the FastAPI TestClient.

Injects the already-loaded real model into app.state (bypassing the lifespan
model-load so we don't pay the cost twice) then hits POST /tts exactly as a
real HTTP client would.
"""
import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu


def test_gpu_server_render(real_model, script_request: dict, artifacts_dir: Path) -> None:
    from app.main import app

    # Inject the pre-loaded real model; skip the lifespan by not using 'with'.
    # Save and restore app.state to avoid polluting other tests.
    prev_model = getattr(app.state, "model", None)
    app.state.model = real_model
    try:
        client = TestClient(app, raise_server_exceptions=True)
        response = client.post("/tts", json=script_request)
    finally:
        app.state.model = prev_model

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    with wave.open(io.BytesIO(response.content)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == real_model.sr
        duration_s = wf.getnframes() / wf.getframerate()

    assert duration_s >= 5.0, f"Audio too short for a ~10-second script: {duration_s:.2f}s"

    out_path = artifacts_dir / "gpu_server_render.wav"
    out_path.write_bytes(response.content)

    print(f"\nSaved : {out_path}")
    print(f"Size  : {len(response.content):,} bytes  |  Duration: {duration_s:.2f}s")
