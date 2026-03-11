"""Tests for the stdlib TTSRequest dataclass (FEAT-4)."""
import json
from pathlib import Path

from app.models import TTSRequest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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

    def test_as_dict_text_roundtrip(self):
        text = "The quick brown fox."
        req = TTSRequest(text=text)
        assert req.as_dict()["text"] == text

    def test_custom_params(self):
        req = TTSRequest(text="Hi.", temperature=1.5, top_p=0.9, top_k=500, repetition_penalty=1.5)
        d = req.as_dict()
        assert d["temperature"] == 1.5
        assert d["top_p"] == 0.9
        assert d["top_k"] == 500
        assert d["repetition_penalty"] == 1.5

    def test_no_fastapi_or_pydantic_import(self):
        """app.models must be importable without fastapi or pydantic installed."""
        import importlib
        import sys
        import unittest.mock as mock

        sys.modules.pop("app.models", None)
        with mock.patch.dict(sys.modules, {"fastapi": None, "pydantic": None}):
            mod = importlib.import_module("app.models")
            assert hasattr(mod, "TTSRequest")

    def test_video_agent_ww2_tanks_fixture(self):
        """All WW2 tanks narration texts must be accepted and round-trip through as_dict()."""
        with open(FIXTURES_DIR / "ww2_tanks_segments.json") as f:
            fixture = json.load(f)
        for seg in fixture["segments"]:
            req = TTSRequest(text=seg["text"])
            d = req.as_dict()
            assert d["text"] == seg["text"]
            assert set(d.keys()) == {"text", "temperature", "top_p", "top_k", "repetition_penalty"}
