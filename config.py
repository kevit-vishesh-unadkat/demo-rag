import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "docs"

# Models (override in .env)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# For Hindi/Gujarati documents use:
# EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# (then re-run ingest.py)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-prompt-guard-2-86m")

# Chunking / retrieval
CHUNK_WORDS = 400
CHUNK_OVERLAP = 60
TOP_K = 4

# language label -> (whisper code, edge-tts voice)
LANGUAGES = {
    "English": ("en", "en-US-AriaNeural"),
    "Hindi": ("hi", "hi-IN-SwaraNeural"),
    "Gujarati": ("gu", "gu-IN-DhwaniNeural"),
}
