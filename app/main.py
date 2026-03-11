from __future__ import annotations

import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LEN = 5000


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ChatterboxTurboTTS on %s...", DEVICE)
    app.state.model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    logger.info("Model ready.")
    yield


app = FastAPI(title="Chatterbox Turbo TTS", version="0.1.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
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
    sf.write(buf, wav[0].cpu().numpy(), sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "chatterbox-turbo"}


@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
def synthesize(req: TTSRequest) -> Response:
    model: ChatterboxTurboTTS = app.state.model
    wav = model.generate(
        req.text,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )
    return Response(content=_encode_wav(wav, model.sr), media_type="audio/wav")


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

    # Write to a temp file so the model can open it by path.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        wav = model.generate(
            text,
            audio_prompt_path=tmp_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    finally:
        os.unlink(tmp_path)

    return Response(content=_encode_wav(wav, model.sr), media_type="audio/wav")
