"""
backend/app/core/config.py

Central configuration. All values can be overridden via environment
variables or a .env file in the backend directory.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5:7b"          # swap to "llama3.1:8b" if preferred
    MODEL_TEMPERATURE: float = 0.1           # low = more deterministic scores
    MODEL_TIMEOUT_SECONDS: int = 120

    # ── Batching ──────────────────────────────────────────────────────────────
    FACETS_PER_BATCH: int = 20               # how many facets per LLM call
    MAX_CONCURRENT_WORKERS: int = 2          # parallel LLM calls at once

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR:    Path = Path(__file__).parent.parent / "data"
    FACETS_JSON: Path = DATA_DIR / "processed" / "facets_enriched.json"

    # ── API ───────────────────────────────────────────────────────────────────
    API_TITLE: str = "Facet Eval API"
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton — import this everywhere
settings = Settings()
