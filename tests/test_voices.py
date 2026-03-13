"""
Tests for voice management endpoints:
- GET /v1/voices       — list registered voices
- POST /v1/voices      — register a new voice
- GET /v1/voices/{id}  — get single voice metadata
- DELETE /v1/voices/{id} — remove a voice
- POST /tts with voice field — synthesize using a registered voice
"""
import io
import subprocess
import tempfile
import wave
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

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


def _create_voice(
    client: TestClient, name: str = "Test Voice", duration_s: float = 6.0
) -> dict:
    """Helper to create a voice and return the response JSON."""
    r = client.post(
        "/v1/voices",
        data={"name": name},
        files={"reference_audio": ("ref.wav", _make_wav_bytes(duration_s), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# GET /v1/voices
# ---------------------------------------------------------------------------

class TestListVoices:
    def test_empty_list_200(self, client: TestClient) -> None:
        r = client.get("/v1/voices")
        assert r.status_code == 200
        assert r.json()["voices"] == []

    def test_list_after_create(self, client: TestClient) -> None:
        _create_voice(client, "Narrator")
        r = client.get("/v1/voices")
        voices = r.json()["voices"]
        assert len(voices) == 1
        assert voices[0]["voice_id"] == "narrator"
        assert voices[0]["name"] == "Narrator"


# ---------------------------------------------------------------------------
# POST /v1/voices
# ---------------------------------------------------------------------------

class TestCreateVoice:
    def test_create_201(self, client: TestClient) -> None:
        body = _create_voice(client, "My Voice")
        assert body["voice_id"] == "my-voice"
        assert body["name"] == "My Voice"
        assert "created_at" in body

    def test_slug_generation(self, client: TestClient) -> None:
        body = _create_voice(client, "Sarah  Warm!!")
        assert body["voice_id"] == "sarah-warm"

    def test_duplicate_name_409(self, client: TestClient) -> None:
        _create_voice(client, "Unique Voice")
        r = client.post(
            "/v1/voices",
            data={"name": "Unique Voice"},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 409

    def test_missing_audio_422(self, client: TestClient) -> None:
        r = client.post("/v1/voices", data={"name": "No Audio"})
        assert r.status_code == 422

    def test_missing_name_422(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422

    def test_invalid_audio_422(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            data={"name": "Bad Audio"},
            files={"reference_audio": ("ref.wav", b"not audio data", "audio/wav")},
        )
        assert r.status_code == 422

    def test_audio_too_short_422(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            data={"name": "Short Clip"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=1.0), "audio/wav")
            },
        )
        assert r.status_code == 422

    def test_audio_too_long_422(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            data={"name": "Long Clip", "max_duration_s": "10"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=15.0), "audio/wav")
            },
        )
        assert r.status_code == 422

    def test_max_duration_override_accepts_long_clip(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            data={"name": "Long But Allowed", "max_duration_s": "60"},
            files={
                "reference_audio": ("ref.wav", _make_wav_bytes(duration_s=35.0), "audio/wav")
            },
        )
        assert r.status_code == 201

    def test_create_from_mp3_201(self, client: TestClient) -> None:
        mp3_data = _make_mp3_bytes()
        r = client.post(
            "/v1/voices",
            data={"name": "MP3 Voice"},
            files={"reference_audio": ("ref.mp3", mp3_data, "audio/mpeg")},
        )
        assert r.status_code == 201
        assert r.json()["voice_id"] == "mp3-voice"

    def test_name_with_no_alphanumeric_422(self, client: TestClient) -> None:
        r = client.post(
            "/v1/voices",
            data={"name": "---!!!"},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/voices/{voice_id}
# ---------------------------------------------------------------------------

class TestGetVoice:
    def test_get_existing_200(self, client: TestClient) -> None:
        _create_voice(client, "Fetchable")
        r = client.get("/v1/voices/fetchable")
        assert r.status_code == 200
        body = r.json()
        assert body["voice_id"] == "fetchable"
        assert body["name"] == "Fetchable"
        assert "duration_s" in body
        assert "sample_rate" in body

    def test_get_nonexistent_404(self, client: TestClient) -> None:
        r = client.get("/v1/voices/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/voices/{voice_id}
# ---------------------------------------------------------------------------

class TestDeleteVoice:
    def test_delete_existing_204(self, client: TestClient) -> None:
        _create_voice(client, "Disposable")
        r = client.delete("/v1/voices/disposable")
        assert r.status_code == 204
        # Confirm it's gone.
        r = client.get("/v1/voices/disposable")
        assert r.status_code == 404

    def test_delete_nonexistent_404(self, client: TestClient) -> None:
        r = client.delete("/v1/voices/ghost")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /tts with voice field
# ---------------------------------------------------------------------------

class TestTTSWithVoice:
    def test_tts_with_registered_voice_200(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _create_voice(client, "Speaker One")
        r = client.post("/tts", json={"text": "Hello.", "voice": "speaker-one"})
        assert r.status_code == 200

    def test_tts_with_nonexistent_voice_404(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "Hello.", "voice": "nope"})
        assert r.status_code == 404

    def test_tts_without_voice_uses_default(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "Hello."})
        assert r.status_code == 200
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs.get("audio_prompt_path") is None

    def test_tts_voice_passes_audio_prompt_path(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _create_voice(client, "Prompted")
        client.post("/tts", json={"text": "Hello.", "voice": "prompted"})
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["audio_prompt_path"] is not None
        assert "prompted" in kwargs["audio_prompt_path"]
        assert kwargs["audio_prompt_path"].endswith("reference.wav")
