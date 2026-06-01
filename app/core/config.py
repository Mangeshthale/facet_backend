"""
backend/app/core/config.py
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5:7b"
    MODEL_TEMPERATURE: float = 0.1
    MODEL_TIMEOUT_SECONDS: int = 120

    # ── Batching ──────────────────────────────────────────────────────────────
    FACETS_PER_BATCH: int = 20
    MAX_CONCURRENT_WORKERS: int = 2

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(__file__).parent.parent.parent.parent / "data"
    FACETS_JSON: Path = DATA_DIR / "processed" / "facets_enriched.json"

    # ── API ───────────────────────────────────────────────────────────────────
    API_TITLE: str = "Facet Eval API"
    API_VERSION: str = "1.0.0"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
