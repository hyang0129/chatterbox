"""
GPU integration test configuration.

Strategy for coexisting with the parent tests/conftest.py mock
---------------------------------------------------------------
The parent conftest stubs chatterbox/perth/librosa in sys.modules with MagicMock
objects (numba is never actually imported).  This conftest runs immediately after
and:

1. Sets NUMBA_CACHE_DIR *before* numba is first imported, fixing the "no locator
   available" error that occurs when numba tries to write cache files alongside
   site-packages sources.
2. Removes the MagicMock stubs so the real packages are imported on demand by
   the GPU test fixtures.

The server test re-uses the already-imported app.main (which has a mock
ChatterboxTurboTTS bound to its module namespace) but bypasses the lifespan by
injecting the real model directly into app.state — so it doesn't matter that
app.main's class reference is stale.
"""
import json
import os
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Must be set before numba is first imported.
_numba_cache = os.path.join(tempfile.gettempdir(), "numba_cache")
os.makedirs(_numba_cache, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _numba_cache)

# Remove the MagicMock stubs installed by the parent conftest so the real
# chatterbox / perth / librosa packages are importable from this point on.
_STUB_PREFIXES = ("chatterbox", "perth", "librosa")
for _k in [
    k for k in sys.modules
    if k in _STUB_PREFIXES or any(k.startswith(p + ".") for p in _STUB_PREFIXES)
]:
    if isinstance(sys.modules[_k], MagicMock):
        del sys.modules[_k]

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def make_voice_store_with_kronimi(voices_dir: str):
    """Create a temp VoiceStore with a kronimi7030 reference-WAV entry.

    Uses a reference WAV (not conditionals.pt) so the route takes the
    audio_prompt_path code path — avoiding Conditionals.load, which in this
    test process points to the mock stub imported by the unit-test conftest.
    """
    from app.voices import VoiceStore

    voice_dir = os.path.join(voices_dir, "kronimi7030")
    os.makedirs(voice_dir, exist_ok=True)

    # Write a 6-second silent WAV (minimum accepted by VoiceStore validation).
    sr = 24000
    with wave.open(os.path.join(voice_dir, "reference.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * (sr * 6))

    meta = {
        "voice_id": "kronimi7030",
        "name": "kronimi7030",
        "original_filename": "reference.wav",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": 6.0,
        "sample_rate": sr,
    }
    with open(os.path.join(voice_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    return VoiceStore(Path(voices_dir))


@pytest.fixture(scope="session")
def real_model():
    """Load ChatterboxTurboTTS once for the entire GPU test session."""
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA GPU not available")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    try:
        return ChatterboxTurboTTS.from_pretrained(device="cuda")
    except Exception as e:
        pytest.skip(f"Model load failed: {e}")


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture(scope="session")
def script_request() -> dict:
    with open(FIXTURES_DIR / "ten_second_script.json") as f:
        return json.load(f)
