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


@pytest.fixture(scope="session")
def real_model():
    """Load ChatterboxTurboTTS once for the entire GPU test session."""
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA GPU not available")
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set — run `huggingface-cli login` or export HF_TOKEN=<token>")
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
