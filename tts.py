"""Text-to-speech with edge-tts (free, needs internet). Returns MP3 bytes."""
import asyncio
import re
import threading

import edge_tts


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>]+", "", text)          # markdown leftovers
    text = re.sub(r"https?://\S+", "", text)       # URLs
    return re.sub(r"\s+", " ", text).strip()


async def _synth(text: str, voice: str) -> bytes:
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def synthesize(text: str, voice: str) -> bytes:
    text = clean_for_speech(text)
    if not text:
        return b""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_synth(text, voice))  # normal case

    # A loop is already running in this thread: use a helper thread.
    result = {}
    def runner():
        result["audio"] = asyncio.run(_synth(text, voice))
    t = threading.Thread(target=runner)
    t.start()
    t.join()
    return result["audio"]
