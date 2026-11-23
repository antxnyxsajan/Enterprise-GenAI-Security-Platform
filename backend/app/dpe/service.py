"""Data Protection Engine (DPE) service orchestrator for Module 3.

Scans normalized prompts for sensitive Personally Identifiable Information (PII)
and redacts matches using standardized replacement tokens.

Provides:
- Synchronous evaluation: evaluate_sync(prompt)
- Asynchronous evaluation (non-blocking event loop): evaluate(prompt) via asyncio.to_thread
- Module 1 -> Module 3 chaining: evaluate_raw(prompt)
"""

import asyncio
from collections import Counter
import time
from typing import List, Optional, Sequence, Tuple

from app.dpe.analyzer import PIIAnalyzer, get_default_analyzer
from app.dpe.anonymizer import PIIAnonymizer, get_default_anonymizer
from app.dpe.schemas import DPEResult, PIIEntityMatch
from app.sanitizer import SanitizationResult, sanitize_prompt


class DataProtectionEngine:
    """Orchestrates PII analysis and anonymization for Module 3."""

    def __init__(
        self,
        analyzer: Optional[PIIAnalyzer] = None,
        anonymizer: Optional[PIIAnonymizer] = None,
        entities: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ):
        self.analyzer = analyzer or get_default_analyzer()
        self.anonymizer = anonymizer or get_default_anonymizer()
        self.entities = list(entities) if entities else None
        self.score_threshold = score_threshold

    def evaluate_sync(self, prompt: str) -> DPEResult:
        """Synchronously analyzes and redacts PII from prompt text."""
        if not prompt or not prompt.strip():
            return DPEResult(
                original_text=prompt or "",
                anonymized_text="",
                has_pii=False,
                entities_found=[],
                entity_counts={},
                matches=[],
                total_entities_count=0,
                latency_ms=0.0,
            )

        t0 = time.perf_counter()

        # Step 1: Scan for PII matches
        matches: List[PIIEntityMatch] = self.analyzer.analyze(
            text=prompt,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

        has_pii = len(matches) > 0

        # Step 2: Redact if matches were detected
        if has_pii:
            anonymized = self.anonymizer.anonymize(prompt, matches)
            type_counter = Counter(m.entity_type for m in matches)
            entity_counts = dict(type_counter)
            entities_found = list(entity_counts.keys())
        else:
            anonymized = prompt
            entity_counts = {}
            entities_found = []

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DPEResult(
            original_text=prompt,
            anonymized_text=anonymized,
            has_pii=has_pii,
            entities_found=entities_found,
            entity_counts=entity_counts,
            matches=matches,
            total_entities_count=len(matches),
            latency_ms=round(elapsed_ms, 3),
        )

    async def evaluate(self, prompt: str) -> DPEResult:
        """Asynchronously evaluates prompt text without blocking the event loop.

        Offloads CPU-bound spaCy / Presidio execution to background worker threads
        via asyncio.to_thread.
        """
        return await asyncio.to_thread(self.evaluate_sync, prompt)

    def evaluate_raw_sync(self, raw_prompt: str) -> Tuple[SanitizationResult, DPEResult]:
        """Chains Module 1 (Sanitizer) -> Module 3 (DPE) synchronously."""
        sanitized = sanitize_prompt(raw_prompt)
        dpe_result = self.evaluate_sync(sanitized.cleaned_text)
        return sanitized, dpe_result

    async def evaluate_raw(self, raw_prompt: str) -> Tuple[SanitizationResult, DPEResult]:
        """Chains Module 1 (Sanitizer) -> Module 3 (DPE) asynchronously."""
        sanitized = sanitize_prompt(raw_prompt)
        dpe_result = await self.evaluate(sanitized.cleaned_text)
        return sanitized, dpe_result


_default_dpe: Optional[DataProtectionEngine] = None


def get_default_dpe() -> DataProtectionEngine:
    """Returns the shared lazy-loaded DataProtectionEngine singleton."""
    global _default_dpe
    if _default_dpe is None:
        _default_dpe = DataProtectionEngine()
    return _default_dpe
