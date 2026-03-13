"""
Tests for the Chatterbox Turbo TTS FastAPI server.

Coverage:
- GET /health
- POST /tts  — synthesis with default and registered voices, validation, tags
"""
import io
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


def _create_voice(client: TestClient, name: str) -> dict:
    """Register a voice via the clone endpoint."""
    r = client.post(
        "/voices/clone",
        data={"name": name},
        files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_status_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_body(self, client: TestClient) -> None:
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "chatterbox-turbo"
        assert body["sample_rate"] == 24000  # MOCK_SR from conftest


# ---------------------------------------------------------------------------
# POST /tts
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_kronimi_voice")
class TestTTS:
    def test_basic_synthesis_200(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hello, world!"})
        assert r.status_code == 200

    def test_content_type_is_wav(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Content type check."})
        assert r.headers["content-type"] == "audio/wav"

    def test_response_is_valid_wav(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "WAV structure test."})
        assert r.status_code == 200
        _assert_valid_wav(r.content)

    def test_model_generate_called(self, client: TestClient, mock_model: MagicMock) -> None:
        client.post("/tts", json={"text": "Verify call."})
        mock_model.generate.assert_called_once()

    def test_default_voice_is_kronimi7030(self, client: TestClient) -> None:
        """Without explicit voice, the default kronimi7030 is used."""
        r = client.post("/tts", json={"text": "Default voice."})
        assert r.status_code == 200

    def test_explicit_voice_200(self, client: TestClient) -> None:
        _create_voice(client, "Speaker One")
        r = client.post("/tts", json={"text": "Hello.", "voice": "speaker-one"})
        assert r.status_code == 200

    def test_nonexistent_voice_404(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hello.", "voice": "nope"})
        assert r.status_code == 404

    def test_precomputed_conditionals_used(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        """Default voice has conditionals.pt — no audio_prompt_path should be set."""
        client.post("/tts", json={"text": "Hello."})
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs.get("audio_prompt_path") is None

    def test_reference_wav_voice_passes_audio_prompt(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        _create_voice(client, "Prompted")
        client.post("/tts", json={"text": "Hello.", "voice": "prompted"})
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["audio_prompt_path"] is not None
        assert "prompted" in kwargs["audio_prompt_path"]

    # --- input validation ---

    def test_empty_string_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": ""})
        assert r.status_code == 422

    def test_whitespace_only_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "   \t\n"})
        assert r.status_code == 422

    def test_missing_text_field_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={})
        assert r.status_code == 422

    def test_text_at_max_length_accepted(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "A" * 5000})
        assert r.status_code == 200

    def test_text_exceeds_max_length_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "A" * 5001})
        assert r.status_code == 422

    # --- paralinguistic tags ---

    def test_all_paralinguistic_tags(self, client: TestClient) -> None:
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

    def test_unicode_latin_extended(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Héllo wörld, café résumé."})
        assert r.status_code == 200

    def test_unicode_cyrillic(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Привет мир."})
        assert r.status_code == 200

    # --- temperature ---

    def test_temperature_default(self, client: TestClient, mock_model: MagicMock) -> None:
        client.post("/tts", json={"text": "Hi."})
        assert mock_model.generate.call_args.kwargs["temperature"] == pytest.approx(0.8)

    def test_temperature_custom_accepted(
        self, client: TestClient, mock_model: MagicMock
    ) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 1.5})
        assert r.status_code == 200
        assert mock_model.generate.call_args.kwargs["temperature"] == pytest.approx(1.5)

    def test_temperature_too_low_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 0.05})
        assert r.status_code == 422

    def test_temperature_too_high_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 2.1})
        assert r.status_code == 422

    def test_temperature_boundary_min(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 0.1})
        assert r.status_code == 200

    def test_temperature_boundary_max(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "temperature": 2.0})
        assert r.status_code == 200

    # --- top_p ---

    def test_top_p_boundary_zero(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 0.0})
        assert r.status_code == 200

    def test_top_p_boundary_one(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 1.0})
        assert r.status_code == 200

    def test_top_p_above_one_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "top_p": 1.01})
        assert r.status_code == 422

    # --- repetition_penalty ---

    def test_repetition_penalty_below_min_rejected(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "repetition_penalty": 0.9})
        assert r.status_code == 422

    def test_repetition_penalty_at_min_accepted(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Hi.", "repetition_penalty": 1.0})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Response metadata headers
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_kronimi_voice")
class TestAudioResponseHeaders:
    def test_x_audio_duration_s_present(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Header check."})
        assert "x-audio-duration-s" in r.headers

    def test_x_audio_duration_s_is_positive_float(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Duration float check."})
        val = float(r.headers["x-audio-duration-s"])
        assert val > 0.0

    def test_x_sample_rate_matches_model_sr(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Sample rate check."})
        assert r.headers["x-sample-rate"] == "24000"

    def test_x_audio_frames_is_positive_integer(self, client: TestClient) -> None:
        r = client.post("/tts", json={"text": "Frame count check."})
        frames = int(r.headers["x-audio-frames"])
        assert frames > 0

    def test_duration_consistency(self, client: TestClient) -> None:
        """frames / sample_rate must equal reported duration (within rounding)."""
        r = client.post("/tts", json={"text": "Consistency check."})
        sr = int(r.headers["x-sample-rate"])
        frames = int(r.headers["x-audio-frames"])
        duration = float(r.headers["x-audio-duration-s"])
        assert abs(frames / sr - duration) < 0.01
