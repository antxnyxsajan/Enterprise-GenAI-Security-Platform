"""Lazy-loaded Sentence-Transformers embedder for Module 2.

Uses `all-MiniLM-L6-v2` (384 dimensions) with L2-normalized embeddings so
cosine similarity equals dot product in Qdrant.
"""

import logging
import threading
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def resolve_torch_device(preferred_device: Optional[str] = None) -> str:
    """Resolves CUDA if available, otherwise safely falls back to CPU."""
    if preferred_device:
        return preferred_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class PromptEmbedder:
    """Thread-safe, lazy-loaded wrapper around SentenceTransformer."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.embedding_model_name
        self._preferred_device = device
        self._device: Optional[str] = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_torch_device(self._preferred_device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Loads the SentenceTransformer model lazily in a thread-safe manner."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            target_device = self.device
            logger.info(
                "Loading embedding model '%s' on device '%s'",
                self.model_name,
                target_device,
            )
            self._model = SentenceTransformer(self.model_name, device=target_device)

    def embed(self, text: str) -> List[float]:
        """Generates a normalized 384-dim embedding vector for a single prompt."""
        self.load()
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates normalized embedding vectors for a batch of prompts."""
        if not texts:
            return []
        self.load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]


_default_embedder: Optional[PromptEmbedder] = None


def get_default_embedder() -> PromptEmbedder:
    """Returns the shared lazy-loaded default PromptEmbedder singleton."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = PromptEmbedder()
    return _default_embedder
