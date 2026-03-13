from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSRequest:
    """
    Typed request object for the Chatterbox TTS API.

    Pure-stdlib dataclass — no FastAPI or Pydantic dependency. Clients can import it
    directly without installing server-side packages::

        from app.models import TTSRequest
        req = TTSRequest(text="Hello, world!")
        payload = req.as_dict()  # pass to requests.post(..., json=payload)

    Thread safety
    -------------
    This is a plain data container. Instances are safe to read from multiple threads,
    but the model that processes them is not — see README serverless section.

    Fields mirror the POST /tts JSON body:
        text               -- text to synthesize (1..5000 chars, non-whitespace-only)
        voice              -- registered voice ID for cloning (None = default voice)
        temperature        -- sampling temperature (0.1..2.0, default 0.8)
        top_p              -- nucleus sampling p  (0.0..1.0, default 0.95)
        top_k              -- top-k sampling      (1..5000,  default 1000)
        repetition_penalty -- repetition penalty  (1.0..3.0, default 1.2)
    """

    text: str
    voice: str | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2

    def as_dict(self) -> dict:
        d: dict = {
            "text": self.text,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.voice is not None:
            d["voice"] = self.voice
        return d
