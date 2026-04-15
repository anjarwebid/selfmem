from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        log.info("Embedding model loaded")
    return _model


def get_embedding(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
