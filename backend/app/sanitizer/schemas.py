from dataclasses import dataclass, field
from typing import List


@dataclass
class SanitizationResult:
    """Structured output contract for Module 1.
    
    Provides both the clean text required by downstream models (Firewall, Presidio, LLM)
    and security evasion telemetry consumed by Module 4 (Security Policy Engine).
    """
    original_text: str
    cleaned_text: str
    was_modified: bool
    invisible_chars_count: int = 0
    homoglyphs_replaced_count: int = 0
    suspicious_suffix_detected: bool = False
    stripped_suffix: str = ""
    evasion_flags: List[str] = field(default_factory=list)

    @property
    def has_evasion_indicators(self) -> bool:
        """Returns True if any evasive behavior was detected in the prompt."""
        return len(self.evasion_flags) > 0
