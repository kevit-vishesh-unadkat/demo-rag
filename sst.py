"""Speech-to-text with faster-whisper (local, CPU)."""
import io
from functools import lru_cache
from typing import Optional

from faster_whisper import WhisperModel

import config


@lru_cache(maxsize=1)
def get_model():
    return WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes, language: Optional[str] = None) -> str:
    """language: whisper code ('en', 'hi', 'gu') or None to auto-detect."""
    segments, _info = get_model().transcribe(
        io.BytesIO(audio_bytes),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
