---
id: FEAT-4
title: "Export a stdlib TTSRequest dataclass so clients don't depend on FastAPI/Pydantic"
severity: Low
status: open
---

## Problem

`TTSRequest` is currently a Pydantic `BaseModel` defined inside `app/main.py`. Clients
that want to construct a typed request object (e.g. the video-agent's
`src/tools/chatterbox_backend.py`) cannot import it without depending on `fastapi` and
`pydantic`, which are server-side dependencies.

The video-agent runs in a separate Python environment (venv on Python 3.10.11) that
does not install FastAPI or Pydantic. Importing from `app.main` is therefore impossible
without polluting the client's dependency tree.

## Implementation Plan

### Create `app/models.py`

```python
from dataclasses import dataclass


@dataclass
class TTSRequest:
    """
    Typed request object for the Chatterbox TTS API.

    This is a pure-stdlib dataclass with no FastAPI or Pydantic dependency.
    Clients can import it directly:

        from app.models import TTSRequest

    The server-side Pydantic validation model remains in app/main.py and is
    separate from this class.

    Fields mirror the POST /tts JSON body:
        text               — text to synthesize (1..5000 chars, non-whitespace-only)
        temperature        — sampling temperature (0.1..2.0, default 0.8)
        top_p              — nucleus sampling p (0.0..1.0, default 0.95)
        top_k              — top-k sampling (1..5000, default 1000)
        repetition_penalty — repetition penalty (1.0..3.0, default 1.2)
    """

    text: str
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
```

### Keep the Pydantic model in `app/main.py`

The server-side `TTSRequest(BaseModel)` in `app/main.py` handles validation and stays
unchanged. The new dataclass is an additional client-facing type, not a replacement.

No imports in `app/main.py` need to change.

### `app/__init__.py` re-export (optional)

If convenient for callers, re-export from the package root:

```python
# app/__init__.py
from app.models import TTSRequest

__all__ = ["TTSRequest"]
```

## Testing Plan

### Unit tests — `tests/test_models.py`

```python
from app.models import TTSRequest


class TestTTSRequestDataclass:
    def test_defaults(self):
        req = TTSRequest(text="Hello.")
        assert req.temperature == 0.8
        assert req.top_p == 0.95
        assert req.top_k == 1000
        assert req.repetition_penalty == 1.2

    def test_as_dict_contains_all_fields(self):
        req = TTSRequest(text="Hello.", temperature=1.0)
        d = req.as_dict()
        assert d["text"] == "Hello."
        assert d["temperature"] == 1.0
        assert "top_p" in d
        assert "top_k" in d
        assert "repetition_penalty" in d

    def test_no_fastapi_or_pydantic_import(self):
        """Importing app.models must not require fastapi or pydantic."""
        import importlib
        import sys

        # Remove any previously-imported app.models to force a fresh load
        sys.modules.pop("app.models", None)

        # Temporarily hide fastapi and pydantic
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"fastapi": None, "pydantic": None}):
            mod = importlib.import_module("app.models")
            assert hasattr(mod, "TTSRequest")

    def test_video_agent_fixture_segment(self):
        """Canonical video-agent segment texts must be accepted without error."""
        texts = [
            "Did you know Allied soldiers called any enemy tank a Tiger — regardless of its actual type — because the Tiger I was so feared?",
            "The American M4 Sherman was produced over 49,000 times — winning through sheer quantity over German armor quality.",
            "The Soviet T-34 shocked German engineers so much they considered copying its sloped armor for their own future tank designs.",
            "Germany's Panzer VIII Maus weighed 188 tonnes — so heavy it could only cross rivers by driving along the submerged riverbed.",
            "Britain's Hobart's Funnies were Churchill variants that could flail mines, bridge gaps, and lay roads — revolutionizing assault engineering.",
        ]
        for text in texts:
            req = TTSRequest(text=text)
            d = req.as_dict()
            assert d["text"] == text
```

### Fixture

The fixture `tests/fixtures/ww2_tanks_segments.json` contains the five segment texts
above. Tests that verify the dataclass works with real video-agent content import from
there rather than hardcoding the strings.
