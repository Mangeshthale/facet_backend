"""
backend/app/services/llm_client.py

Thin async wrapper around the Ollama /api/generate endpoint.
Swap OLLAMA_BASE_URL in .env to point at a remote GPU server
if you're not running locally.
"""

from __future__ import annotations
import json
import httpx
import os as _os

from app.core.config import settings


class OllamaClient:
    """Async HTTP client for Ollama."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.MODEL_NAME
        self.timeout = settings.MODEL_TIMEOUT_SECONDS

    async def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the raw response text.
        Uses the /api/generate endpoint with stream=False for simplicity.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.MODEL_TEMPERATURE,
                "num_predict": 2048,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]

    async def health_check(self) -> bool:
        """Returns True if Ollama is reachable and model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                tags = resp.json().get("models", [])
                model_names = [m["name"] for m in tags]
                # Accept partial match (e.g. "qwen2.5:7b" matches "qwen2.5")
                base = self.model.split(":")[0]
                return any(base in name for name in model_names)
        except Exception:
            return False


# Singleton
ollama = OllamaClient()

if _os.getenv("GROQ_API_KEY"):
    from app.services.llm_client_groq import groq_client as ollama  # noqa
    import logging as _log
    _log.getLogger(__name__).info("🌐 Groq API detected — using cloud mode")
else:
    import logging as _log
    _log.getLogger(__name__).info("🖥  No GROQ_API_KEY — using local Ollama")