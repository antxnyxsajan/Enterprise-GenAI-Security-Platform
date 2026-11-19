"""Hybrid Prompt Firewall orchestrator for Module 2.

Executes both independent detectors in parallel on every prompt:
1. Qdrant vector similarity search (all-MiniLM-L6-v2)
2. Pretrained ML classifier (protectai/deberta-v3-base-prompt-injection)

Neither detector gates the other — both scores are preserved independently for
Module 4 (Security Policy Engine) and Module 6 (Complementarity Dashboard).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Optional, Tuple

from app.config import settings
from app.firewall.classifier import PromptClassifier, get_default_classifier
from app.firewall.embedder import PromptEmbedder, get_default_embedder
from app.firewall.qdrant_store import QdrantSignatureStore
from app.firewall.schemas import FirewallResult, QdrantMatch
from app.sanitizer import SanitizationResult, sanitize_prompt


class HybridFirewall:
    """Parallel dual-detector prompt firewall (Qdrant + DeBERTa-v3)."""

    def __init__(
        self,
        embedder: Optional[PromptEmbedder] = None,
        store: Optional[QdrantSignatureStore] = None,
        classifier: Optional[PromptClassifier] = None,
        qdrant_threshold: Optional[float] = None,
        ml_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        self.embedder = embedder or get_default_embedder()
        self.store = store or QdrantSignatureStore(in_memory=False)
        self.classifier = classifier or get_default_classifier()
        self.qdrant_threshold = (
            settings.qdrant_risk_threshold if qdrant_threshold is None else qdrant_threshold
        )
        self.ml_threshold = (
            settings.ml_risk_threshold if ml_threshold is None else ml_threshold
        )
        self.top_k = settings.qdrant_top_k if top_k is None else top_k

    def _run_qdrant(self, prompt: str) -> Tuple[float, List[QdrantMatch], float]:
        """Runs embedding + Qdrant cosine similarity lookup and returns (score, matches, ms)."""
        t0 = time.perf_counter()
        vector = self.embedder.embed(prompt)
        matches = self.store.query(vector, top_k=self.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        top_score = matches[0].score if matches else 0.0
        return top_score, matches, elapsed_ms

    def _run_classifier(self, prompt: str) -> Tuple[float, float]:
        """Runs DeBERTa-v3 inference and returns (score, ms)."""
        t0 = time.perf_counter()
        score = self.classifier.classify(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return score, elapsed_ms

    def _build_result(
        self,
        prompt: str,
        qdrant_score: float,
        matches: List[QdrantMatch],
        qdrant_ms: float,
        ml_score: float,
        ml_ms: float,
        total_ms: float,
    ) -> FirewallResult:
        top_match = matches[0] if matches else None
        device = getattr(self.classifier, "device", "unknown")
        return FirewallResult(
            prompt=prompt,
            qdrant_score=round(qdrant_score, 6),
            ml_score=round(ml_score, 6),
            qdrant_flagged=qdrant_score >= self.qdrant_threshold,
            ml_flagged=ml_score >= self.ml_threshold,
            matched_category=top_match.category if top_match else None,
            matched_signature=top_match.matched_prompt if top_match else None,
            top_matches=matches,
            qdrant_latency_ms=round(qdrant_ms, 3),
            ml_latency_ms=round(ml_ms, 3),
            total_latency_ms=round(total_ms, 3),
            device_used=device,
        )

    def _empty_result(self, prompt: str) -> FirewallResult:
        return FirewallResult(
            prompt=prompt,
            qdrant_score=0.0,
            ml_score=0.0,
            qdrant_flagged=False,
            ml_flagged=False,
            device_used=getattr(self.classifier, "device", "cpu"),
        )

    def evaluate_sync(self, prompt: str) -> FirewallResult:
        """Evaluates a normalized prompt by running both detectors concurrently in threads."""
        if not prompt or not prompt.strip():
            return self._empty_result(prompt or "")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            qdrant_future = executor.submit(self._run_qdrant, prompt)
            ml_future = executor.submit(self._run_classifier, prompt)

            qdrant_score, matches, qdrant_ms = qdrant_future.result()
            ml_score, ml_ms = ml_future.result()

        total_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(
            prompt=prompt,
            qdrant_score=qdrant_score,
            matches=matches,
            qdrant_ms=qdrant_ms,
            ml_score=ml_score,
            ml_ms=ml_ms,
            total_ms=total_ms,
        )

    async def evaluate(self, prompt: str) -> FirewallResult:
        """Async evaluation using asyncio.gather for FastAPI / parallel DPE integration."""
        if not prompt or not prompt.strip():
            return self._empty_result(prompt or "")

        t0 = time.perf_counter()
        (qdrant_score, matches, qdrant_ms), (ml_score, ml_ms) = await asyncio.gather(
            asyncio.to_thread(self._run_qdrant, prompt),
            asyncio.to_thread(self._run_classifier, prompt),
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(
            prompt=prompt,
            qdrant_score=qdrant_score,
            matches=matches,
            qdrant_ms=qdrant_ms,
            ml_score=ml_score,
            ml_ms=ml_ms,
            total_ms=total_ms,
        )

    def evaluate_raw_sync(self, raw_prompt: str) -> Tuple[SanitizationResult, FirewallResult]:
        """Chains Module 1 (Sanitizer) -> Module 2 (Hybrid Firewall) synchronously."""
        sanitized = sanitize_prompt(raw_prompt)
        firewall_result = self.evaluate_sync(sanitized.cleaned_text)
        return sanitized, firewall_result

    async def evaluate_raw(self, raw_prompt: str) -> Tuple[SanitizationResult, FirewallResult]:
        """Chains Module 1 (Sanitizer) -> Module 2 (Hybrid Firewall) asynchronously."""
        sanitized = sanitize_prompt(raw_prompt)
        firewall_result = await self.evaluate(sanitized.cleaned_text)
        return sanitized, firewall_result
