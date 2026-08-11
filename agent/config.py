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
    LLM_PROVIDER: str = "ollama"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "http://127.0.0.1:13000/v1"
    OPENAI_MODEL: str = "deepseek-v4-pro"
    OPENAI_TEMPERATURE: float = 0.1

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5"
    OLLAMA_TEMPERATURE: float = 0.1

    # ---- Agent ----
    MAX_ITERATIONS: int = 20
    MAX_EXECUTION_TIME_SEC: int = 120
    MAX_SHORT_TERM_ROUNDS: int = 6
    MAX_RETRIES: int = 1

    # ---- Tools ----
    FILE_READER_ROOT: str = "."
    NOTES_DIR: str = "./agent_notes"
    CODE_WORKDIR: str = "./agent_workspace"

    # ---- Browser ----
    BROWSER_HEADLESS: bool = False  # False=visible window, True=headless

    def get_llm_config(self) -> dict:
        """Return active LLM configuration based on current provider."""
        provider = self.LLM_PROVIDER.strip().lower()
        if provider == "ollama":
            model = self.OLLAMA_MODEL
            temperature = self.OLLAMA_TEMPERATURE
        else:
            model = self.OPENAI_MODEL
            temperature = self.OPENAI_TEMPERATURE
        return {
            "provider": provider,
            "model": model,
            "temperature": temperature,
        }


settings = Settings()
logger.info(
    "Settings loaded | provider=%s model=%s",
    settings.LLM_PROVIDER,
    settings.OPENAI_MODEL,
)
