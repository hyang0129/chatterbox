from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_REFERENCE_SIZE = 50 * 1024 * 1024  # 50 MB
MIN_DURATION_S = 3.0
DEFAULT_MAX_DURATION_S = 300.0  # 5 minutes
MIN_SAMPLE_RATE = 16000

# Audio formats that soundfile (libsndfile) can read natively.
_SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


class VoiceMetadata(BaseModel):
    voice_id: str
    name: str
    original_filename: str
    created_at: datetime
    duration_s: float
    sample_rate: int


class VoiceCreateResponse(BaseModel):
    voice_id: str
    name: str
    created_at: datetime


class VoiceListResponse(BaseModel):
    voices: list[VoiceMetadata]


def _slugify(name: str) -> str:
    """Convert a voice name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError("name must contain at least one alphanumeric character")
    return slug


def _convert_to_wav(audio_bytes: bytes, original_filename: str) -> bytes:
    """Convert audio bytes to WAV using ffmpeg.

    If the file is already a format soundfile can read, returns the bytes unchanged.
    For MP3 and other formats, shells out to ffmpeg to produce PCM 16-bit WAV.
    """
    ext = Path(original_filename).suffix.lower() if original_filename else ""
    if ext in _SOUNDFILE_EXTENSIONS:
        return audio_bytes

    # Write input to a temp file, convert with ffmpeg, read output.
    with (
        tempfile.NamedTemporaryFile(suffix=ext or ".bin", delete=False) as src,
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst,
    ):
        src.write(audio_bytes)
        src_path, dst_path = src.name, dst.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ac", "1",           # mono
                "-acodec", "pcm_s16le",
                dst_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise ValueError(f"ffmpeg conversion failed: {stderr[:200]}")
        return Path(dst_path).read_bytes()
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


def _validate_reference_audio(
    audio_bytes: bytes,
    original_filename: str = "",
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> tuple[bytes, float, int]:
    """Validate and normalise reference audio.

    Converts non-WAV formats (e.g. MP3) to WAV via ffmpeg, then checks duration
    and sample rate constraints.

    Args:
        max_duration_s: Upper limit on reference duration. Defaults to 5 minutes.
            Callers can override for longer reference clips.

    Returns (wav_bytes, duration_s, sample_rate).
    """
    if len(audio_bytes) > MAX_REFERENCE_SIZE:
        raise ValueError(
            f"Reference audio too large: {len(audio_bytes)} bytes "
            f"(max {MAX_REFERENCE_SIZE // (1024 * 1024)} MB)"
        )

    wav_bytes = _convert_to_wav(audio_bytes, original_filename)

    try:
        data, sample_rate = sf.read(io.BytesIO(wav_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid audio file: {exc}") from exc

    if sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate too low: {sample_rate} Hz (min {MIN_SAMPLE_RATE} Hz)"
        )

    frame_count = data.shape[0]
    duration_s = frame_count / sample_rate
    if duration_s < MIN_DURATION_S:
        raise ValueError(
            f"Reference audio too short: {duration_s:.1f}s (min {MIN_DURATION_S}s)"
        )
    if duration_s > max_duration_s:
        raise ValueError(
            f"Reference audio too long: {duration_s:.1f}s (max {max_duration_s}s)"
        )

    return wav_bytes, duration_s, sample_rate


class VoiceStore:
    """File-based voice reference storage.

    Each voice is stored as a directory ``<base_dir>/<voice_id>/`` containing
    ``reference.wav`` and ``metadata.json``.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, VoiceMetadata] = {}
        self._scan()

    def _scan(self) -> None:
        """Rebuild the in-memory index from disk."""
        self._index.clear()
        for meta_path in self._base_dir.glob("*/metadata.json"):
            try:
                raw = json.loads(meta_path.read_text())
                meta = VoiceMetadata(**raw)
                self._index[meta.voice_id] = meta
            except Exception:
                logger.warning("Skipping corrupt voice metadata: %s", meta_path)

    def list_voices(self) -> list[VoiceMetadata]:
        """Return all registered voices sorted by name."""
        return sorted(self._index.values(), key=lambda v: v.name)

    def get_voice(self, voice_id: str) -> VoiceMetadata | None:
        return self._index.get(voice_id)

    def get_reference_path(self, voice_id: str) -> Path:
        """Return the path to a voice's reference WAV. Raises KeyError if not found."""
        if voice_id not in self._index:
            raise KeyError(voice_id)
        return self._base_dir / voice_id / "reference.wav"

    def create_voice(
        self,
        name: str,
        audio_bytes: bytes,
        original_filename: str,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
    ) -> VoiceMetadata:
        """Validate audio, persist to disk, and return metadata."""
        voice_id = _slugify(name)
        if voice_id in self._index:
            raise FileExistsError(voice_id)

        wav_bytes, duration_s, sample_rate = _validate_reference_audio(
            audio_bytes, original_filename, max_duration_s=max_duration_s
        )

        voice_dir = self._base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)

        ref_path = voice_dir / "reference.wav"
        ref_path.write_bytes(wav_bytes)

        meta = VoiceMetadata(
            voice_id=voice_id,
            name=name,
            original_filename=original_filename,
            created_at=datetime.now(timezone.utc),
            duration_s=round(duration_s, 2),
            sample_rate=sample_rate,
        )
        meta_path = voice_dir / "metadata.json"
        meta_path.write_text(meta.model_dump_json(indent=2))

        self._index[voice_id] = meta
        logger.info("Registered voice %r (id=%s, %.1fs)", name, voice_id, duration_s)
        return meta

    def delete_voice(self, voice_id: str) -> bool:
        """Remove a voice from disk and index. Returns False if not found."""
        if voice_id not in self._index:
            return False
        voice_dir = self._base_dir / voice_id
        shutil.rmtree(voice_dir, ignore_errors=True)
        del self._index[voice_id]
        logger.info("Deleted voice %s", voice_id)
        return True
