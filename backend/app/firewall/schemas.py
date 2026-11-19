"""Data contracts for Module 2: Hybrid Prompt Firewall."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QdrantMatch:
    """Represents a single similarity match returned from the Qdrant signature store."""

    signature_id: str
    score: float
    category: str
    matched_prompt: str


@dataclass
class FirewallResult:
    """Structured dual-detector output from Module 2.

    Both detectors run independently in parallel and report separate scores so that
    Module 4 (Security Policy Engine) and Module 6 (Dashboard) can inspect individual
    signals and measure detector complementarity.
    """

    prompt: str
    qdrant_score: float
    ml_score: float
    qdrant_flagged: bool
    ml_flagged: bool
    matched_category: Optional[str] = None
    matched_signature: Optional[str] = None
    top_matches: List[QdrantMatch] = field(default_factory=list)
    qdrant_latency_ms: float = 0.0
    ml_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    device_used: str = "cpu"

    @property
    def max_score(self) -> float:
        """Returns the maximum risk score across both independent detectors."""
        return max(self.qdrant_score, self.ml_score)

    @property
    def is_flagged(self) -> bool:
        """Returns True if either detector exceeded its configured risk threshold."""
        return self.qdrant_flagged or self.ml_flagged

    @property
    def detection_source(self) -> str:
        """Describes which detector(s) flagged the prompt (useful for complementarity analysis)."""
        if self.qdrant_flagged and self.ml_flagged:
            return "both"
        if self.qdrant_flagged:
            return "qdrant_only"
        if self.ml_flagged:
            return "ml_only"
        return "none"
