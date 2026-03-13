from __future__ import annotations

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from app.voices import (
    DEFAULT_MAX_DURATION_S,
    VoiceListResponse,
    VoiceStore,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LEN = 5000
VOICES_DIR = os.environ.get("CHATTERBOX_VOICES_DIR", "./voices")
DEFAULT_VOICE = "kronimi7030"


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


app = FastAPI(title="Chatterbox Turbo TTS", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    voice: str = Field(DEFAULT_VOICE, description="Voice ID (default: kronimi7030)")
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


class VoiceCreateResponse(BaseModel):
    voice_id: str
    name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_wav(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a (1, N) float32 waveform tensor to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav[0].cpu().numpy(), sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _audio_headers(wav: torch.Tensor, sample_rate: int) -> dict:
    """Build X-Audio-* response headers from a (1, N) waveform tensor."""
    frame_count = wav.shape[-1]
    return {
        "X-Audio-Duration-S": f"{frame_count / sample_rate:.2f}",
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Frames": str(frame_count),
    }


def _generate_with_voice(
    model: ChatterboxTurboTTS,
    *,
    text: str,
    audio_prompt_path: str | None,
    conditionals_path: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> torch.Tensor:
    """Generate speech, loading pre-computed conditionals when available."""
    if conditionals_path is not None:
        model.conds = Conditionals.load(conditionals_path)
        return model.generate(
            text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    return model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )


def _blend_conditionals(
    conds_a: Conditionals,
    conds_b: Conditionals,
    texture_mix: int,
) -> Conditionals:
    """Blend two sets of voice conditionals into a novel third voice.

    Voice *texture* (speaker identity embeddings) is interpolated on a 0–100
    scale where 0 = pure voice A and 100 = pure voice B.

    Voice *rhythm* (prosodic speech-token prompt and mel reference) is always
    taken from voice A (the first voice). To use voice B's rhythm, swap the
    order of the arguments.
    """
    alpha = texture_mix / 100.0  # 0.0 = all A, 1.0 = all B

    # --- T3 speaker embedding (256-dim, L2-normalised) ---
    emb_a = conds_a.t3.speaker_emb.float()
    emb_b = conds_b.t3.speaker_emb.float()
    blended_t3 = (1.0 - alpha) * emb_a + alpha * emb_b
    blended_t3 = blended_t3 / blended_t3.norm(p=2, dim=-1, keepdim=True)

    # --- S3Gen x-vector embedding (192-dim, L2-normalised) ---
    xvec_a = conds_a.gen["embedding"].float()
    xvec_b = conds_b.gen["embedding"].float()
    blended_xvec = (1.0 - alpha) * xvec_a + alpha * xvec_b
    blended_xvec = blended_xvec / blended_xvec.norm(p=2, dim=-1, keepdim=True)

    # --- Rhythm: always from voice A ---
    t3_cond = T3Cond(
        speaker_emb=blended_t3.to(dtype=emb_a.dtype),
        cond_prompt_speech_tokens=conds_a.t3.cond_prompt_speech_tokens,
        emotion_adv=conds_a.t3.emotion_adv,
    )

    gen_dict = dict(conds_a.gen)  # shallow copy — rhythm from A
    gen_dict["embedding"] = blended_xvec.to(dtype=xvec_a.dtype)

    return Conditionals(t3_cond, gen_dict)


# ---------------------------------------------------------------------------
# 1. POST /tts — synthesize speech with a voice ID
# ---------------------------------------------------------------------------


@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize(req: TTSRequest) -> Response:
    """Synthesize speech from text using a registered voice."""
    model: ChatterboxTurboTTS = app.state.model
    voice_store: VoiceStore = app.state.voice_store

    # Resolve voice: prefer pre-computed conditionals, fall back to reference WAV.
    audio_prompt_path: str | None = None
    conditionals_path: str | None = None
    try:
        cond_path = voice_store.get_conditionals_path(req.voice)
    except KeyError:
        raise HTTPException(404, detail=f"Voice not found: {req.voice}")
    if cond_path is not None:
        conditionals_path = str(cond_path)
    else:
        audio_prompt_path = str(voice_store.get_reference_path(req.voice))

    async with app.state.lock:
        wav = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _generate_with_voice(
                model,
                text=req.text,
                audio_prompt_path=audio_prompt_path,
                conditionals_path=conditionals_path,
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


# ---------------------------------------------------------------------------
# 2. POST /voices/clone — clone a voice from reference audio, save it
# ---------------------------------------------------------------------------


@app.post("/voices/clone", response_model=VoiceCreateResponse, status_code=201)
async def clone_voice(
    name: str = Form(..., min_length=1, max_length=200),
    reference_audio: UploadFile = File(...),
    max_duration_s: float = Form(DEFAULT_MAX_DURATION_S, ge=3.0, le=7200.0),
) -> VoiceCreateResponse:
    """Register a new voice by cloning from a reference audio file."""
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

    return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)


# ---------------------------------------------------------------------------
# 3. POST /voices/blend — blend two voices, save result as a new voice
# ---------------------------------------------------------------------------


@app.post("/voices/blend", response_model=VoiceCreateResponse, status_code=201)
async def blend_voices(
    name: str = Form(..., min_length=1, max_length=200),
    voice_a: str = Form(
        ..., description="Voice ID for the first source (supplies rhythm)"
    ),
    voice_b: str = Form(..., description="Voice ID for the second source"),
    texture_mix: int = Form(
        50,
        ge=0,
        le=100,
        description="Texture blend: 0 = pure voice_a, 100 = pure voice_b",
    ),
) -> VoiceCreateResponse:
    """Create a new voice by blending two existing voices.

    Voice *texture* is interpolated according to ``texture_mix``.
    Voice *rhythm* always comes from ``voice_a`` (the first voice).
    To use voice B's rhythm, swap the order.
    """
    model: ChatterboxTurboTTS = app.state.model
    voice_store: VoiceStore = app.state.voice_store

    # Resolve source voice reference paths.
    for vid, label in [(voice_a, "voice_a"), (voice_b, "voice_b")]:
        if voice_store.get_voice(vid) is None:
            raise HTTPException(404, detail=f"{label} not found: {vid}")

    path_a = str(voice_store.get_reference_path(voice_a))
    path_b = str(voice_store.get_reference_path(voice_b))

    # Extract conditionals and blend (requires GPU lock).
    async with app.state.lock:
        blended = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _extract_and_blend(model, path_a, path_b, texture_mix),
        )

    # Save blended conditionals as a new voice.
    try:
        meta = voice_store.create_blended_voice(
            name=name,
            conditionals=blended,
            blend_config={
                "voice_a": voice_a,
                "voice_b": voice_b,
                "texture_mix": texture_mix,
            },
            sample_rate=model.sr,
        )
    except FileExistsError as exc:
        raise HTTPException(409, detail=f"Voice already exists: {exc}")

    return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)


def _extract_and_blend(
    model: ChatterboxTurboTTS,
    path_a: str,
    path_b: str,
    texture_mix: int,
) -> Conditionals:
    """Prepare conditionals for both voices, blend, return result."""
    model.prepare_conditionals(path_a)
    conds_a = model.conds

    model.prepare_conditionals(path_b)
    conds_b = model.conds

    return _blend_conditionals(conds_a, conds_b, texture_mix)


# ---------------------------------------------------------------------------
# 4. GET /voices — list available voices
# ---------------------------------------------------------------------------


@app.get("/voices", response_model=VoiceListResponse)
async def list_voices() -> VoiceListResponse:
    """List all registered voice IDs."""
    voice_store: VoiceStore = app.state.voice_store
    return VoiceListResponse(voices=voice_store.list_voices())


# ---------------------------------------------------------------------------
# 5. DELETE /voices/{voice_id} — delete a voice
# ---------------------------------------------------------------------------


@app.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    """Delete a registered voice by ID."""
    voice_store: VoiceStore = app.state.voice_store
    if not voice_store.delete_voice(voice_id):
        raise HTTPException(404, detail=f"Voice not found: {voice_id}")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


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
