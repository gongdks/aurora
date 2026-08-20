"""Unified config — pydantic-settings loads from .env with validation.

Switch LLM provider by changing one line in .env.
All fields are auto-validated on startup with clear error messages.
"""

from __future__ import annotations

import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_ENV_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)


class Settings(BaseSettings):
    """Application settings auto-loaded from .env with type validation.

    Invalid values (e.g. MAX_ITERATIONS=abc) raise a clear ValidationError
    at startup instead of silently using defaults.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "http://127.0.0.1:13000/v1"
    OPENAI_MODEL: str = "deepseek-v4-pro"
    OPENAI_TEMPERATURE: float = 0.1

    # ---- Agent ----
    MAX_ITERATIONS: int = 20
    MAX_EXECUTION_TIME_SEC: int = 120
    MAX_SHORT_TERM_ROUNDS: int = 6
    MAX_RETRIES: int = 1
    MAX_PLAN_ROUNDS: int = 3

    # ---- Network ----
    LLM_TIMEOUT_SEC: int = 60
    LLM_STREAM_TIMEOUT_SEC: int = 120
    HTTP_TIMEOUT_SEC: int = 30
    HTTP_MAX_RETRIES: int = 2

    # ---- Tools ----
    FILE_READER_ROOT: str = "."
    NOTES_DIR: str = "./agent_notes"
    CODE_WORKDIR: str = "./agent_workspace"
    TOOL_OUTPUT_TRUNCATE: int = 2000

    # ---- Browser ----
    BROWSER_HEADLESS: bool = False

    # ---- Embedding (for vector memory) ----
    # Options: remote | tfidf
    EMBEDDING_PROVIDER: str = "remote"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_DIMENSION: int = 1024

    # Remote embedding (used when EMBEDDING_PROVIDER=remote)
    # Compatible with OpenAI-compatible APIs (DashScope, Azure, OneAPI, etc.)
    REMOTE_EMBEDDING_BASE_URL: str = ""
    REMOTE_EMBEDDING_API_KEY: str = ""
    REMOTE_EMBEDDING_MODEL: str = "text-embedding-3-small"
    REMOTE_EMBEDDING_DIMENSION: int = 1536

    # ---- Vector Store ----
    # Options: sqlite | memory
    #   sqlite:  Persistent SQLite BLOB storage (default, zero-dependency)
    #   memory:  Fast in-memory storage (lost on restart)
    #
    # NOTE: SQLite is ALWAYS used for structured data (facts, skills, configs).
    #       This setting controls ONLY the vector search backend.
    VECTOR_STORE: str = "sqlite"

    # ---- Verbosity ----
    VERBOSE: bool = False

    def get_llm_config(self) -> dict:
        """Return active LLM configuration."""
        return {
            "provider": self.LLM_PROVIDER.strip().lower(),
            "model": self.OPENAI_MODEL,
            "temperature": self.OPENAI_TEMPERATURE,
        }


settings = Settings()
logger.info(
    "Settings loaded | provider=%s model=%s embed=%s",
    settings.LLM_PROVIDER,
    settings.OPENAI_MODEL,
    settings.EMBEDDING_PROVIDER,
)