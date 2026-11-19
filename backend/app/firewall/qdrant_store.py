"""Qdrant vector store manager for Module 2 attack signature matching.

Supports dual operation modes:
- in_memory=True:  QdrantClient(":memory:") for isolated, lock-free unit tests.
- in_memory=False: Local embedded disk storage (./data/qdrant_data) for persistent runtime.
"""

from pathlib import Path
from typing import Dict, List, Optional
import uuid

from app.config import settings
from app.firewall.schemas import QdrantMatch


def _clamp_score(score: float) -> float:
    """Clamps cosine similarity scores cleanly into [0.0, 1.0]."""
    return max(0.0, min(1.0, float(score)))


def deterministic_signature_id(prompt: str, category: str = "unknown") -> str:
    """Generates a deterministic UUIDv5 for idempotent upserts (avoids duplicates on re-seed)."""
    key = f"{category.strip().lower()}::{prompt.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantSignatureStore:
    """Manages Qdrant collection initialization, signature upserts, and similarity search."""

    def __init__(
        self,
        in_memory: bool = False,
        storage_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        self.in_memory = in_memory
        self.storage_path = storage_path or settings.qdrant_storage_path
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size or settings.embedding_dimension
        self._client = None

    @property
    def client(self):
        """Lazily initializes and returns the underlying QdrantClient."""
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.in_memory:
                self._client = QdrantClient(":memory:")
            else:
                path_obj = Path(self.storage_path)
                path_obj.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path_obj))
        return self._client

    def init_collection(self) -> None:
        """Creates the jailbreak_signatures collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_signatures(
        self,
        signatures: List[Dict[str, str]],
        vectors: List[List[float]],
    ) -> int:
        """Idempotently inserts or updates attack signature vectors and payloads.

        Args:
            signatures: List of dicts containing at least 'prompt' and optional 'category'.
            vectors: Corresponding list of normalized embedding vectors.

        Returns:
            Number of signature points upserted.
        """
        if len(signatures) != len(vectors):
            raise ValueError(
                f"Mismatched counts: {len(signatures)} signatures vs {len(vectors)} vectors."
            )
        if not signatures:
            return 0

        from qdrant_client.models import PointStruct

        self.init_collection()

        points = []
        for sig, vec in zip(signatures, vectors):
            prompt = sig["prompt"]
            category = sig.get("category", "unknown")
            point_id = sig.get("id") or deterministic_signature_id(prompt, category)
            payload = {
                "prompt": prompt,
                "category": category,
                "source": sig.get("source", "local_seed"),
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def query(self, vector: List[float], top_k: int = 3) -> List[QdrantMatch]:
        """Queries Qdrant for the top-k closest known attack signatures."""
        if not self.client.collection_exists(self.collection_name):
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
        )
        hits = response.points if hasattr(response, "points") else response

        matches: List[QdrantMatch] = []
        for hit in hits:
            payload = hit.payload or {}
            matches.append(
                QdrantMatch(
                    signature_id=str(hit.id),
                    score=_clamp_score(hit.score),
                    category=str(payload.get("category", "unknown")),
                    matched_prompt=str(payload.get("prompt", "")),
                )
            )
        return matches

    def count(self) -> int:
        """Returns the number of indexed attack signatures in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(collection_name=self.collection_name).count)

    def close(self) -> None:
        """Closes the Qdrant client and releases any embedded disk file locks."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def __enter__(self) -> "QdrantSignatureStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
