"""
Mock out chatterbox and its heavy transitive dependencies (perth → librosa → numba)
at the sys.modules level before app.main is ever imported. This prevents the librosa
numba-cache error that occurs in the dev container and removes any GPU requirement
from the test suite.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

MOCK_SR = 24000

# --------------------------------------------------------------------------
# Set up a temporary voices directory before app.main is imported, so the
# VoiceStore created during lifespan uses a clean temp dir.
# --------------------------------------------------------------------------
_voices_tmpdir = tempfile.mkdtemp(prefix="chatterbox_test_voices_")
os.environ["CHATTERBOX_VOICES_DIR"] = _voices_tmpdir


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    model.sr = MOCK_SR
    # 0.5 s of silence — real tensor so torchaudio.save works
    model.generate.return_value = torch.zeros(1, MOCK_SR // 2)
    return model


# --------------------------------------------------------------------------
# Stub heavy dependencies before app.main (and therefore chatterbox) is
# imported. setdefault leaves any already-imported real module untouched.
# --------------------------------------------------------------------------
_mock_model_instance = _make_mock_model()
_mock_tts_turbo = MagicMock()
_mock_tts_turbo.ChatterboxTurboTTS.from_pretrained.return_value = _mock_model_instance

for _name in ("perth", "librosa", "chatterbox", "chatterbox.tts", "chatterbox.tts_turbo"):
    sys.modules.setdefault(_name, MagicMock())
sys.modules["chatterbox.tts_turbo"] = _mock_tts_turbo  # always override this one

# --------------------------------------------------------------------------
# Import app after mocks are in place
# --------------------------------------------------------------------------
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def mock_model() -> MagicMock:
    """Return the shared mock model with generate() reset before each test."""
    _mock_model_instance.generate.reset_mock()
    _mock_model_instance.generate.return_value = torch.zeros(1, MOCK_SR // 2)
    return _mock_model_instance


@pytest.fixture(autouse=True)
def _clean_voice_store():
    """Reset the voice store directory between tests that modify it."""
    yield
    # Clean up any voices created during the test.
    import shutil
    for entry in os.listdir(_voices_tmpdir):
        path = os.path.join(_voices_tmpdir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
    # Re-scan so the in-memory index matches disk.
    voice_store = getattr(app.state, "voice_store", None)
    if voice_store is not None:
        voice_store._scan()
