"""
Tests for FEAT-7: per-voice WPM estimation.

Coverage:
- WPM is computed and returned on POST /voices/clone
- WPM is computed and returned on POST /voices/blend
- WPM appears in GET /voices list
- WPM appears in GET /voices/{voice_id} detail
- X-Voice-WPM header is present on POST /tts responses
- Calibration constants are sane
"""
import io
import json as json_mod
import os
import wave
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from app.voices import CALIBRATION_TEXT, CALIBRATION_WORD_COUNT
from tests.conftest import _Conditionals, _T3Cond

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_s: float = 6.0, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    buf.seek(0)
    return buf.read()


def _clone_voice(client: TestClient, name: str = "Test Voice") -> dict:
    r = client.post(
        "/voices/clone",
        data={"name": name},
        files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _make_fake_conditionals():
    t3 = _T3Cond(
        speaker_emb=torch.randn(1, 256),
        cond_prompt_speech_tokens=torch.randint(0, 6000, (1, 375)),
        emotion_adv=torch.tensor([[[0.0]]]),
    )
    gen = {
        "embedding": torch.randn(1, 192),
        "prompt_token": torch.randint(0, 6000, (1, 100)),
        "prompt_token_len": torch.tensor([100]),
        "prompt_feat": torch.randn(1, 200, 80),
        "prompt_feat_len": None,
    }
    return _Conditionals(t3, gen)


def _create_precomputed_voice(voice_id: str, wpm: float | None = None) -> None:
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
        "wpm": wpm,
    }
    with open(os.path.join(voice_dir, "metadata.json"), "w") as f:
        json_mod.dump(meta, f)
    torch.save({"dummy": True}, os.path.join(voice_dir, "conditionals.pt"))
    from app.main import app as _app
    _app.state.voice_store._scan()


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------


class TestCalibrationConstants:
    def test_word_count_matches_text(self) -> None:
        assert CALIBRATION_WORD_COUNT == len(CALIBRATION_TEXT.split())

    def test_word_count_in_expected_range(self) -> None:
        assert 40 <= CALIBRATION_WORD_COUNT <= 100


# ---------------------------------------------------------------------------
# WPM on clone
# ---------------------------------------------------------------------------


class TestCloneWPM:
    def test_clone_returns_wpm(self, client: TestClient, mock_model: MagicMock) -> None:
        body = _clone_voice(client, "WPM Test")
        assert "wpm" in body
        assert isinstance(body["wpm"], float)
        assert body["wpm"] > 0

    def test_clone_wpm_is_reasonable(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        """Mock generates 0.5s of audio for 60 words → very high WPM, but non-zero."""
        body = _clone_voice(client, "WPM Reasonable")
        assert body["wpm"] > 0

    def test_clone_wpm_persisted_to_voice_store(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        body = _clone_voice(client, "WPM Persist")
        r = client.get("/voices")
        voices = r.json()["voices"]
        match = [v for v in voices if v["voice_id"] == body["voice_id"]]
        assert len(match) == 1
        assert match[0]["wpm"] == body["wpm"]


# ---------------------------------------------------------------------------
# WPM on blend
# ---------------------------------------------------------------------------


class TestBlendWPM:
    @pytest.fixture(autouse=True)
    def _setup_blend_mock(self, client: TestClient, mock_model: MagicMock):
        _clone_voice(client, "Blend A")
        _clone_voice(client, "Blend B")
        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        call_count = {"n": 0}

        def fake_prepare(path, **kwargs):
            if call_count["n"] == 0:
                mock_model.conds = conds_a
            else:
                mock_model.conds = conds_b
            call_count["n"] += 1

        mock_model.prepare_conditionals = MagicMock(side_effect=fake_prepare)

    def test_blend_returns_wpm(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={"name": "Blend WPM", "voice_a": "blend-a", "voice_b": "blend-b"},
        )
        assert r.status_code == 201
        body = r.json()
        assert "wpm" in body
        assert isinstance(body["wpm"], float)
        assert body["wpm"] > 0


# ---------------------------------------------------------------------------
# WPM in voice list and detail
# ---------------------------------------------------------------------------


class TestVoiceWPMEndpoints:
    def test_voice_list_includes_wpm(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _clone_voice(client, "List WPM")
        r = client.get("/voices")
        voices = r.json()["voices"]
        assert len(voices) >= 1
        assert "wpm" in voices[0]

    def test_voice_detail_includes_wpm(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _clone_voice(client, "Detail WPM")
        r = client.get("/voices/detail-wpm")
        assert r.status_code == 200
        body = r.json()
        assert "wpm" in body
        assert body["wpm"] is not None

    def test_voice_detail_404(self, client: TestClient) -> None:
        r = client.get("/voices/nonexistent")
        assert r.status_code == 404

    def test_voice_detail_wpm_null_for_legacy(self, client: TestClient) -> None:
        """Pre-existing voices without WPM return null."""
        _create_precomputed_voice("legacy-voice", wpm=None)
        r = client.get("/voices/legacy-voice")
        assert r.status_code == 200
        assert r.json()["wpm"] is None


# ---------------------------------------------------------------------------
# X-Voice-WPM header on /tts
# ---------------------------------------------------------------------------


class TestTTSWPMHeader:
    def test_x_voice_wpm_present_when_calibrated(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _create_precomputed_voice("wpm-voice", wpm=142.5)
        r = client.post("/tts", json={"text": "Hello.", "voice": "wpm-voice"})
        assert r.status_code == 200
        assert "x-voice-wpm" in r.headers
        assert float(r.headers["x-voice-wpm"]) == pytest.approx(142.5, abs=0.1)

    def test_x_voice_wpm_absent_when_not_calibrated(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _create_precomputed_voice("no-wpm-voice", wpm=None)
        r = client.post("/tts", json={"text": "Hello.", "voice": "no-wpm-voice"})
        assert r.status_code == 200
        assert "x-voice-wpm" not in r.headers
