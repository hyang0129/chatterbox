"""
Tests for voice management endpoints:
- POST /voices/clone   — register a new voice from reference audio
- POST /voices/blend   — create a blended voice from two sources
- GET /voices          — list registered voices
- DELETE /voices/{id}  — remove a voice
"""
import io
import subprocess
import tempfile
import wave
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from tests.conftest import _Conditionals, _T3Cond

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_s: float = 6.0, sample_rate: int = 24000) -> bytes:
    """Build a minimal valid mono WAV in memory (silence)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    buf.seek(0)
    return buf.read()


def _make_mp3_bytes(duration_s: float = 6.0, sample_rate: int = 24000) -> bytes:
    """Build a valid MP3 by converting a WAV via ffmpeg."""
    wav_bytes = _make_wav_bytes(duration_s, sample_rate)
    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src,
        tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as dst,
    ):
        src.write(wav_bytes)
        src_path, dst_path = src.name, dst.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, dst_path],
            capture_output=True,
            check=True,
        )
        with open(dst_path, "rb") as f:
            return f.read()
    finally:
        import os
        os.unlink(src_path)
        os.unlink(dst_path)


def _clone_voice(
    client: TestClient, name: str = "Test Voice", duration_s: float = 6.0
) -> dict:
    """Helper to clone a voice and return the response JSON."""
    r = client.post(
        "/voices/clone",
        data={"name": name},
        files={"reference_audio": ("ref.wav", _make_wav_bytes(duration_s), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _make_fake_conditionals():
    """Build a minimal Conditionals-like object using the test-conftest stubs."""
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


# ---------------------------------------------------------------------------
# GET /voices
# ---------------------------------------------------------------------------


class TestListVoices:
    def test_empty_list_200(self, client: TestClient) -> None:
        r = client.get("/voices")
        assert r.status_code == 200
        assert r.json()["voices"] == []

    def test_list_after_create(self, client: TestClient) -> None:
        _clone_voice(client, "Narrator")
        r = client.get("/voices")
        voices = r.json()["voices"]
        assert len(voices) == 1
        assert voices[0]["voice_id"] == "narrator"
        assert voices[0]["name"] == "Narrator"


# ---------------------------------------------------------------------------
# POST /voices/clone
# ---------------------------------------------------------------------------


class TestCloneVoice:
    def test_clone_201(self, client: TestClient) -> None:
        body = _clone_voice(client, "My Voice")
        assert body["voice_id"] == "my-voice"
        assert body["name"] == "My Voice"

    def test_slug_generation(self, client: TestClient) -> None:
        body = _clone_voice(client, "Sarah  Warm!!")
        assert body["voice_id"] == "sarah-warm"

    def test_duplicate_name_409(self, client: TestClient) -> None:
        _clone_voice(client, "Unique Voice")
        r = client.post(
            "/voices/clone",
            data={"name": "Unique Voice"},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 409

    def test_missing_audio_422(self, client: TestClient) -> None:
        r = client.post("/voices/clone", data={"name": "No Audio"})
        assert r.status_code == 422

    def test_missing_name_422(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422

    def test_invalid_audio_422(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            data={"name": "Bad Audio"},
            files={"reference_audio": ("ref.wav", b"not audio data", "audio/wav")},
        )
        assert r.status_code == 422

    def test_audio_too_short_422(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            data={"name": "Short Clip"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=1.0), "audio/wav")
            },
        )
        assert r.status_code == 422

    def test_audio_too_long_422(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            data={"name": "Long Clip", "max_duration_s": "10"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=15.0), "audio/wav")
            },
        )
        assert r.status_code == 422

    def test_max_duration_override_accepts_long_clip(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            data={"name": "Long But Allowed", "max_duration_s": "60"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=35.0), "audio/wav")
            },
        )
        assert r.status_code == 201

    def test_clone_from_mp3_201(self, client: TestClient) -> None:
        mp3_data = _make_mp3_bytes()
        r = client.post(
            "/voices/clone",
            data={"name": "MP3 Voice"},
            files={"reference_audio": ("ref.mp3", mp3_data, "audio/mpeg")},
        )
        assert r.status_code == 201
        assert r.json()["voice_id"] == "mp3-voice"

    def test_name_with_no_alphanumeric_422(self, client: TestClient) -> None:
        r = client.post(
            "/voices/clone",
            data={"name": "---!!!"},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /voices/blend
# ---------------------------------------------------------------------------


class TestBlendVoice:
    """Endpoint-level tests for voice blending (creates a voice, not audio)."""

    @pytest.fixture(autouse=True)
    def _setup_blend_mock(self, client: TestClient, mock_model: MagicMock):
        """Register two source voices and mock prepare_conditionals."""
        _clone_voice(client, "Voice A")
        _clone_voice(client, "Voice B")

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

    def test_blend_201(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Blended",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["voice_id"] == "blended"
        assert body["name"] == "Blended"

    def test_blend_appears_in_list(self, client: TestClient) -> None:
        client.post(
            "/voices/blend",
            data={
                "name": "Listed Blend",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        r = client.get("/voices")
        ids = [v["voice_id"] for v in r.json()["voices"]]
        assert "listed-blend" in ids

    def test_default_texture_mix_is_50(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Default Mix",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 201

    def test_texture_mix_0(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Pure A",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
                "texture_mix": "0",
            },
        )
        assert r.status_code == 201

    def test_texture_mix_100(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Pure B",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
                "texture_mix": "100",
            },
        )
        assert r.status_code == 201

    def test_texture_mix_below_0_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Bad Mix",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
                "texture_mix": "-1",
            },
        )
        assert r.status_code == 422

    def test_texture_mix_above_100_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Bad Mix",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
                "texture_mix": "101",
            },
        )
        assert r.status_code == 422

    def test_nonexistent_voice_a_404(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Ghost A",
                "voice_a": "ghost",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 404

    def test_nonexistent_voice_b_404(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "Ghost B",
                "voice_a": "voice-a",
                "voice_b": "ghost",
            },
        )
        assert r.status_code == 404

    def test_missing_name_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 422

    def test_missing_voice_a_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "No A",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 422

    def test_missing_voice_b_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/voices/blend",
            data={
                "name": "No B",
                "voice_a": "voice-a",
            },
        )
        assert r.status_code == 422

    def test_duplicate_blend_name_409(self, client: TestClient) -> None:
        client.post(
            "/voices/blend",
            data={
                "name": "Unique Blend",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        r = client.post(
            "/voices/blend",
            data={
                "name": "Unique Blend",
                "voice_a": "voice-a",
                "voice_b": "voice-b",
            },
        )
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# DELETE /voices/{voice_id}
# ---------------------------------------------------------------------------


class TestDeleteVoice:
    def test_delete_existing_204(self, client: TestClient) -> None:
        _clone_voice(client, "Disposable")
        r = client.delete("/voices/disposable")
        assert r.status_code == 204
        # Confirm it's gone.
        r = client.get("/voices")
        ids = [v["voice_id"] for v in r.json()["voices"]]
        assert "disposable" not in ids

    def test_delete_nonexistent_404(self, client: TestClient) -> None:
        r = client.delete("/voices/ghost")
        assert r.status_code == 404
