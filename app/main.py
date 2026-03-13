from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from app.voices import (
    DEFAULT_MAX_DURATION_S,
    VoiceCreateResponse,
    VoiceListResponse,
    VoiceMetadata,
    VoiceStore,
    _convert_to_wav,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LEN = 5000
VOICES_DIR = os.environ.get("CHATTERBOX_VOICES_DIR", "./voices")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ChatterboxTurboTTS on %s...", DEVICE)
    app.state.model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    app.state.lock = asyncio.Lock()
    app.state.voice_store = VoiceStore(Path(VOICES_DIR))
    logger.info(
        "Model ready. %d registered voice(s).",
        len(app.state.voice_store.list_voices()),
    )
    yield


app = FastAPI(title="Chatterbox Turbo TTS", version="0.1.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    voice: str | None = Field(None, description="Registered voice ID for cloning")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(1000, ge=1, le=5000)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)

    @field_validator("text")
    @classmethod
    def no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


def _encode_wav(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a (1, N) float32 waveform tensor to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav[0].cpu().numpy(), sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@app.get("/health")
def health() -> dict:
    model = getattr(app.state, "model", None)
    voice_store: VoiceStore | None = getattr(app.state, "voice_store", None)
    return {
        "status": "ok",
        "model": "chatterbox-turbo",
        "sample_rate": model.sr if model is not None else None,
        "voices": len(voice_store.list_voices()) if voice_store is not None else 0,
    }


def _audio_headers(wav: torch.Tensor, sample_rate: int) -> dict:
    """Build X-Audio-* response headers from a (1, N) waveform tensor."""
    frame_count = wav.shape[-1]
    return {
        "X-Audio-Duration-S": f"{frame_count / sample_rate:.2f}",
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Frames": str(frame_count),
    }


@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize(req: TTSRequest) -> Response:
    model: ChatterboxTurboTTS = app.state.model

    # Resolve voice reference path before acquiring the GPU lock.
    audio_prompt_path: str | None = None
    if req.voice is not None:
        voice_store: VoiceStore = app.state.voice_store
        try:
            audio_prompt_path = str(voice_store.get_reference_path(req.voice))
        except KeyError:
            raise HTTPException(404, detail=f"Voice not found: {req.voice}")

    async with app.state.lock:
        wav = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: model.generate(
                req.text,
                audio_prompt_path=audio_prompt_path,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            ),
        )
    return Response(
        content=_encode_wav(wav, model.sr),
        media_type="audio/wav",
        headers=_audio_headers(wav, model.sr),
    )


@app.post("/tts/clone", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize_clone(
    text: str = Form(..., min_length=1, max_length=MAX_TEXT_LEN),
    temperature: float = Form(0.8, ge=0.1, le=2.0),
    top_p: float = Form(0.95, ge=0.0, le=1.0),
    top_k: int = Form(1000, ge=1, le=5000),
    repetition_penalty: float = Form(1.2, ge=1.0, le=3.0),
    reference_audio: UploadFile = File(...),
) -> Response:
    if not text.strip():
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", "text"),
                    "msg": "Value error, text must contain non-whitespace characters",
                    "input": text,
                    "url": "https://errors.pydantic.dev/2/v/value_error",
                }
            ]
        )

    model: ChatterboxTurboTTS = app.state.model
    audio_bytes = await reference_audio.read()

    # Convert non-WAV formats (e.g. MP3) to WAV so the model can read it.
    filename = reference_audio.filename or ""
    wav_bytes = _convert_to_wav(audio_bytes, filename)

    # Write to a temp file so the model can open it by path.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        async with app.state.lock:
            wav = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: model.generate(
                    text,
                    audio_prompt_path=tmp_path,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                ),
            )
    finally:
        os.unlink(tmp_path)

    return Response(
        content=_encode_wav(wav, model.sr),
        media_type="audio/wav",
        headers=_audio_headers(wav, model.sr),
    )


# ---------------------------------------------------------------------------
# Voice management endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices() -> VoiceListResponse:
    voice_store: VoiceStore = app.state.voice_store
    return VoiceListResponse(voices=voice_store.list_voices())


@app.get("/v1/voices/{voice_id}", response_model=VoiceMetadata)
async def get_voice(voice_id: str) -> VoiceMetadata:
    voice_store: VoiceStore = app.state.voice_store
    meta = voice_store.get_voice(voice_id)
    if meta is None:
        raise HTTPException(404, detail=f"Voice not found: {voice_id}")
    return meta


@app.post(
    "/v1/voices",
    response_model=VoiceCreateResponse,
    status_code=201,
)
async def create_voice(
    name: str = Form(..., min_length=1, max_length=200),
    reference_audio: UploadFile = File(...),
    max_duration_s: float = Form(DEFAULT_MAX_DURATION_S, ge=3.0, le=7200.0),
) -> VoiceCreateResponse:
    voice_store: VoiceStore = app.state.voice_store
    audio_bytes = await reference_audio.read()
    original_filename = reference_audio.filename or "unknown.wav"

    try:
        meta = voice_store.create_voice(
            name, audio_bytes, original_filename, max_duration_s=max_duration_s
        )
    except FileExistsError as exc:
        raise HTTPException(409, detail=f"Voice already exists: {exc}")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    return VoiceCreateResponse(
        voice_id=meta.voice_id,
        name=meta.name,
        created_at=meta.created_at,
    )


@app.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    voice_store: VoiceStore = app.state.voice_store
    if not voice_store.delete_voice(voice_id):
        raise HTTPException(404, detail=f"Voice not found: {voice_id}")
    return Response(status_code=204)
