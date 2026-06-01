"""
backend/app/main.py

FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.batch_router import load_facets
from app.api.evaluate import router as evaluate_router
from app.api.facets import router as facets_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load facets into memory at startup so first request isn't slow."""
    try:
        facets = load_facets()
        logger.info(f"✅ Loaded {len(facets)} facets from {settings.FACETS_JSON}")
    except FileNotFoundError:
        logger.warning(
            "⚠️  Facets JSON not found. "
            "Run: python scripts/preprocess_facets.py"
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Score conversation turns across 300+ psychological and linguistic facets.",
    lifespan=lifespan,
)

# CORS — allow the React dev server and any deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(evaluate_router, tags=["Evaluation"])
app.include_router(facets_router, tags=["Facets"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
