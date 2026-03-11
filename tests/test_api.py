"""
Tests for the Chatterbox Turbo TTS FastAPI server.

Coverage:
- GET /health
- POST /tts  — basic synthesis, input validation, parameter ranges, paralinguistic tags
- POST /tts/clone — voice cloning with reference audio upload
"""
import io
import os
import wave
from unittest.mock import MagicMock

import pytest
import soundfile as sf
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


def _assert_valid_wav(data: bytes, expected_sr: int = 24000) -> None:
    audio, sr = sf.read(io.BytesIO(data))
    assert sr == expected_sr
    assert audio.ndim == 1  # mono


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_body(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.json() == {"status": "ok", "model": "chatterbox-turbo"}


# ---------------------------------------------------------------------------
# POST /tts
# ---------------------------------------------------------------------------

class TestTTS:
    def test_basic_synthesis_200(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hello, world!"})
        assert r.status_code == 200

    def test_content_type_is_wav(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Content type check."})
        assert r.headers["content-type"] == "audio/wav"

    def test_response_is_valid_wav(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "WAV structure test."})
        assert r.status_code == 200
        _assert_valid_wav(r.content)

    def test_model_generate_called(self, client: TestClient, mock_model: MagicMock) -> None:
        client.post("/tts", json={"text": "Verify call."})
        mock_model.generate.assert_called_once()

    # --- input validation ---

    def test_empty_string_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": ""})
        assert r.status_code == 422

    def test_whitespace_only_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "   \t\n"})
        assert r.status_code == 422

    def test_missing_text_field_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={})
        assert r.status_code == 422

    def test_text_at_max_length_accepted(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "A" * 5000})
        assert r.status_code == 200

    def test_text_exceeds_max_length_rejected(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "A" * 5001})
        assert r.status_code == 422

    # --- paralinguistic tags ---

    def test_all_paralinguistic_tags(self, client: TestClient, mock_model: MagicMock) -> None:
        tags = "[cough] [laugh] [chuckle] [sigh] [gasp] [groan] [sniff] [shush] [clear throat]"
        r = client.post("/tts", json={"text": f"Test {tags} done."})
        assert r.status_code == 200

    def test_paralinguistic_tags_passed_to_model(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        text = "Hello [chuckle], how are you? [sigh]"
        client.post("/tts", json={"text": text})
        assert mock_model.generate.call_args.args[0] == text

    # --- unicode ---

    def test_unicode_latin_extended(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Héllo wörld, café résumé."})
        assert r.status_code == 200

    def test_unicode_cyrillic(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Привет мир."})
        assert r.status_code == 200

    # --- temperature ---

    def test_temperature_default(self, client: TestClient, mock_model: MagicMock) -> None:
        client.post("/tts", json={"text": "Hi."})
        assert mock_model.generate.call_args.kwargs["temperature"] == pytest.approx(0.8)

    def test_temperature_custom_accepted(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 1.5})
        assert r.status_code == 200
        assert mock_model.generate.call_args.kwargs["temperature"] == pytest.approx(1.5)

    def test_temperature_too_low_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 0.05})
        assert r.status_code == 422

    def test_temperature_too_high_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 2.1})
        assert r.status_code == 422

    def test_temperature_boundary_min(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 0.1})
        assert r.status_code == 200

    def test_temperature_boundary_max(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 2.0})
        assert r.status_code == 200

    # --- top_p ---

    def test_top_p_boundary_zero(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 0.0})
        assert r.status_code == 200

    def test_top_p_boundary_one(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 1.0})
        assert r.status_code == 200

    def test_top_p_above_one_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 1.01})
        assert r.status_code == 422

    # --- repetition_penalty ---

    def test_repetition_penalty_below_min_rejected(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "Hi.", "repetition_penalty": 0.9})
        assert r.status_code == 422

    def test_repetition_penalty_at_min_accepted(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "Hi.", "repetition_penalty": 1.0})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /tts/clone
# ---------------------------------------------------------------------------

class TestTTSClone:
    def test_basic_clone_200(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post(
            "/tts/clone",
            data={"text": "Hello from a cloned voice."},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200

    def test_clone_content_type_is_wav(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post(
            "/tts/clone",
            data={"text": "Content type check."},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.headers["content-type"] == "audio/wav"

    def test_clone_response_is_valid_wav(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post(
            "/tts/clone",
            data={"text": "WAV structure test."},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        _assert_valid_wav(r.content)

    def test_clone_passes_audio_prompt_path(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        client.post(
            "/tts/clone",
            data={"text": "Check audio_prompt_path is set."},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        kwargs = mock_model.generate.call_args.kwargs
        assert "audio_prompt_path" in kwargs
        assert kwargs["audio_prompt_path"] is not None

    def test_clone_temp_file_cleaned_up(self, client: TestClient, mock_model: MagicMock) -> None:
        """The temporary reference audio file must be deleted after the request."""
        client.post(
            "/tts/clone",
            data={"text": "Temp file cleanup check."},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        tmp_path = mock_model.generate.call_args.kwargs.get("audio_prompt_path", "")
        assert not os.path.exists(tmp_path), f"Temp file was not deleted: {tmp_path}"

    def test_clone_custom_temperature(self, client: TestClient, mock_model: MagicMock) -> None:
        client.post(
            "/tts/clone",
            data={"text": "Custom temp.", "temperature": "1.2"},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert mock_model.generate.call_args.kwargs["temperature"] == pytest.approx(1.2)

    def test_clone_empty_text_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post(
            "/tts/clone",
            data={"text": ""},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422

    def test_clone_whitespace_text_rejected(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post(
            "/tts/clone",
            data={"text": "   "},
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422

    def test_clone_missing_audio_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post("/tts/clone", data={"text": "No audio file."})
        assert r.status_code == 422

    def test_clone_missing_text_rejected(self, client: TestClient, mock_model: MagicMock) -> None:
        r = client.post(
            "/tts/clone",
            files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 422
