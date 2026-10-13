"""One validated embedding space for claims, article search and backfills."""

import logging
import math
import threading
from typing import Any

import httpx
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)
EMBEDDING_DIMENSION = 768


def validate_embedding(values: Any) -> list[float]:
    """Reject incompatible/provider error payloads before they reach pgvector."""
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding must be a numeric vector") from exc
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding must contain {EMBEDDING_DIMENSION} dimensions")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains nonfinite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite, nonzero norm")
    return [value / norm for value in vector]


class EmbeddingService:
    _model: SentenceTransformer | None = None
    _lock = threading.Lock()

    @classmethod
    def load_model(cls, *, force_local: bool = False) -> None:
        """Load once, from a worker thread. Remote mode does not allocate a local model."""
        if get_settings().HUGGINGFACE_API_KEY and not force_local:
            return
        if cls._model is not None:
            return
        with cls._lock:
            if cls._model is None:
                model_name = get_settings().EMBEDDING_MODEL
                logger.info("Loading SentenceTransformer model %s", model_name)
                model = SentenceTransformer(model_name)
                if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Configured model must produce {EMBEDDING_DIMENSION}-dimensional embeddings"
                    )
                cls._model = model

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def model_name(cls) -> str:
        return get_settings().EMBEDDING_MODEL

    @classmethod
    def generate_embedding(cls, text: str, *, allow_load: bool = False) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        settings = get_settings()
        if settings.HUGGINGFACE_API_KEY:
            try:
                return _embed_via_hf_api(text, settings.HUGGINGFACE_API_KEY, settings.EMBEDDING_MODEL)
            except (httpx.HTTPError, ValueError):
                # A request that explicitly forbids loading must never allocate
                # hundreds of MB after a provider outage.
                if cls._model is None and not allow_load:
                    raise RuntimeError("Embedding provider unavailable; local model is not loaded") from None
                logger.warning("Embedding provider unavailable; using the configured local model")
        if cls._model is None:
            if not allow_load:
                raise RuntimeError("Embedding model not loaded. Call load_model() first.")
            cls.load_model(force_local=True)
        assert cls._model is not None
        return validate_embedding(cls._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True,
        ))

    @classmethod
    def generate_document_embedding(cls, text: str, *, allow_load: bool = False) -> list[float]:
        """Pool overlapping windows so late-article content contributes to search.

        Short windows bound the encoder context cost. Mean pooling is a document
        representation, not a claim score or a substitute for passage retrieval.
        """
        words = text.split()
        if not words:
            raise ValueError("Cannot embed empty document")
        vectors = []
        for start in range(0, len(words), 160):
            vectors.append(cls.generate_embedding(" ".join(words[start:start + 192]), allow_load=allow_load))
            if start + 192 >= len(words):
                break
        return validate_embedding([sum(values) for values in zip(*vectors)])


def _embed_via_hf_api(text: str, api_key: str, model_name: str) -> list[float]:
    """Accept sentence vectors only; token matrices must not become first-token vectors."""
    if "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"
    api_url = "https://router.huggingface.co/hf-inference/models/" + model_name
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": text},
        )
        response.raise_for_status()
        result = response.json()
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        result = result[0]
    return validate_embedding(result)
