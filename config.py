"""Configuration module for Lecture-to-Living-Course.

Reads all model names and credentials dynamically from environment variables.
Never hardcodes model names.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv(override=True)


class Config:
    """Central configuration class."""

    # Base workspace directory
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Groq API Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

    # Model Names (dynamically read from env vars, never hardcoded in nodes)
    GROQ_STT_MODEL: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3").strip()
    GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile").strip()
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview").strip()

    # LangSmith Tracing & Evaluation Configuration
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "").strip()
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "lecture-to-living-course").strip()
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()

    # Local Models & Paths
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", str(BASE_DIR / "cache"))).resolve()

    # Processing Parameters
    AUDIO_CHUNK_SIZE_MB: int = int(os.getenv("AUDIO_CHUNK_SIZE_MB", "20"))
    SLIDE_SAMPLE_RATE_SEC: int = int(os.getenv("SLIDE_SAMPLE_RATE_SEC", "8"))
    GAP_SIMILARITY_THRESHOLD: float = float(os.getenv("GAP_SIMILARITY_THRESHOLD", "0.38"))
    MAX_CITATIONS_VERIFICATION_RETRIES: int = int(os.getenv("MAX_CITATIONS_VERIFICATION_RETRIES", "2"))

    # Mind Map Hierarchy & Deduplication Parameters
    MINDMAP_MIN_TOPICS: int = int(os.getenv("MINDMAP_MIN_TOPICS", "4"))
    MINDMAP_MAX_TOPICS: int = int(os.getenv("MINDMAP_MAX_TOPICS", "8"))
    MINDMAP_MAX_SUBTOPICS: int = int(os.getenv("MINDMAP_MAX_SUBTOPICS", "6"))
    MINDMAP_MAX_LEAVES: int = int(os.getenv("MINDMAP_MAX_LEAVES", "5"))
    MINDMAP_MERGE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("MINDMAP_MERGE_SIMILARITY_THRESHOLD", "0.82")
    )

    # Cache Schema Versions (invalidates stale node_07 and node_09 cache entries)
    MINDMAP_CACHE_VERSION: str = os.getenv("MINDMAP_CACHE_VERSION", "v2")
    QUIZ_CACHE_VERSION: str = os.getenv("QUIZ_CACHE_VERSION", "v2")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they do not exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_ffmpeg_binary(cls) -> str:
        """Find the available ffmpeg binary on the system.

        First checks PATH, then falls back to imageio-ffmpeg bundled binary.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg

        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            # Fallback string
            return "ffmpeg"

    @classmethod
    def validate_groq_key(cls) -> bool:
        """Check if a valid Groq API key is present."""
        return bool(cls.GROQ_API_KEY and not cls.GROQ_API_KEY.startswith("your_groq"))

    @classmethod
    def validate_langsmith_key(cls) -> bool:
        """Check if a valid LangSmith API key is present."""
        return bool(cls.LANGSMITH_API_KEY and not cls.LANGSMITH_API_KEY.startswith("your_langsmith"))


# Ensure standard directories exist on import
Config.ensure_directories()
