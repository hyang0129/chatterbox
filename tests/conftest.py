"""
Mock out chatterbox and its heavy transitive dependencies (perth → librosa → numba)
at the sys.modules level before app.main is ever imported. This prevents the librosa
numba-cache error that occurs in the dev container and removes any GPU requirement
from the test suite.
"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
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

for _name in (
    "perth", "librosa",
    "chatterbox", "chatterbox.tts", "chatterbox.tts_turbo",
    "chatterbox.models", "chatterbox.models.t3",
    "chatterbox.models.t3.modules", "chatterbox.models.t3.modules.cond_enc",
):
    sys.modules.setdefault(_name, MagicMock())
sys.modules["chatterbox.tts_turbo"] = _mock_tts_turbo  # always override this one

# --------------------------------------------------------------------------
# Provide real T3Cond / Conditionals so _blend_conditionals can construct
# instances instead of getting opaque MagicMock objects back.
# --------------------------------------------------------------------------


@dataclass
class _T3Cond:
    speaker_emb: torch.Tensor
    clap_emb: Optional[torch.Tensor] = None
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None
    cond_prompt_speech_emb: Optional[torch.Tensor] = None
    emotion_adv: object = 0.5

    def to(self, *, device=None, dtype=None):
        return self


class _Conditionals:
    def __init__(self, t3, gen):
        self.t3 = t3
        self.gen = gen

    def to(self, device):
        return self

    def save(self, path):
        """Mock save — write a small torch checkpoint."""
        torch.save({"t3": self.t3, "gen": self.gen}, path)

    @staticmethod
    def load(path, map_location=None):
        """Mock load that returns a stub Conditionals with valid tensors."""
        t3 = _T3Cond(
            speaker_emb=torch.randn(1, 256),
            cond_prompt_speech_tokens=torch.randint(0, 500, (1, 50)),
        )
        gen = {"embedding": torch.randn(1, 192)}
        return _Conditionals(t3, gen)


sys.modules["chatterbox.models.t3.modules.cond_enc"].T3Cond = _T3Cond
_mock_tts_turbo.T3Cond = _T3Cond
_mock_tts_turbo.Conditionals = _Conditionals

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


def _create_precomputed_voice(voice_id: str, wpm: float | None = None) -> None:
    """Create a voice directory with conditionals.pt (no reference WAV)."""
    voice_dir = os.path.join(_voices_tmpdir, voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    meta = {
        "voice_id": voice_id,
        "name": voice_id,
        "original_filename": "blend",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": 0.0,
        "sample_rate": MOCK_SR,
        "wpm": wpm,
    }
    with open(os.path.join(voice_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    torch.save({"dummy": True}, os.path.join(voice_dir, "conditionals.pt"))
    voice_store = getattr(app.state, "voice_store", None)
    if voice_store is not None:
        voice_store._scan()


@pytest.fixture()
def _kronimi_voice() -> None:
    """Create the default kronimi7030 voice. Request this fixture in any test that POSTs to /tts."""
    _create_precomputed_voice("kronimi7030")


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
