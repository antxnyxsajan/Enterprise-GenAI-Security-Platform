"""Data contracts for Module 3: Data Protection Engine (DPE)."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PIIEntityMatch:
    """Represents a single PII entity detected by the Presidio Analyzer."""

    entity_type: str
    start: int
    end: int
    score: float
    text: str = ""


@dataclass
class DPEResult:
    """Structured output contract for Module 3.

    Delivers the sanitized (redacted) prompt for the LLM along with granular
    entity counts and telemetry for Module 4 (Policy Engine) and Module 6 (Dashboard).
    """

    original_text: str
    anonymized_text: str
    has_pii: bool
    entities_found: List[str] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    matches: List[PIIEntityMatch] = field(default_factory=list)
    total_entities_count: int = 0
    latency_ms: float = 0.0

    @property
    def was_anonymized(self) -> bool:
        """Returns True if any PII was redacted from the original text."""
        return self.has_pii and (self.original_text != self.anonymized_text)
