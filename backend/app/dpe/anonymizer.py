"""Thread-safe lazy-loaded Presidio AnonymizerEngine wrapper for Module 3.

Redacts detected PII entities using standardized replacement tokens:
e.g., `<EMAIL_ADDRESS_REDACTED>`, `<PHONE_NUMBER_REDACTED>`, `<PERSON_REDACTED>`.
"""

import logging
import threading
from typing import Dict, List, Optional

from app.dpe.schemas import PIIEntityMatch

logger = logging.getLogger(__name__)


def default_redaction_token(entity_type: str) -> str:
    """Returns standard uppercase replacement tag for an entity type."""
    cleaned_type = entity_type.strip().upper()
    return f"<{cleaned_type}_REDACTED>"


class PIIAnonymizer:
    """Thread-safe, lazy-loaded wrapper around Presidio AnonymizerEngine."""

    def __init__(self, custom_operators: Optional[Dict[str, str]] = None):
        self.custom_operators = custom_operators or {}
        self._engine = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        """Lazily initializes the Presidio AnonymizerEngine."""
        if self._engine is not None:
            return
        with self._lock:
            if self._engine is not None:
                return
            from presidio_anonymizer import AnonymizerEngine

            logger.info("Initializing Presidio AnonymizerEngine")
            self._engine = AnonymizerEngine()

    def anonymize(
        self,
        text: str,
        matches: List[PIIEntityMatch],
        custom_replacements: Optional[Dict[str, str]] = None,
    ) -> str:
        """Replaces PII entities in text with redaction tokens.

        Args:
            text: Original raw or normalized prompt text.
            matches: List of PIIEntityMatch instances from the analyzer.
            custom_replacements: Optional map of entity_type -> custom replacement string.

        Returns:
            Redacted string with all detected PII masked.
        """
        if not text or not matches:
            return text

        self.load()
        from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

        # Convert PIIEntityMatch back to Presidio's RecognizerResult for AnonymizerEngine
        presidio_results = [
            RecognizerResult(
                entity_type=m.entity_type,
                start=m.start,
                end=m.end,
                score=m.score,
            )
            for m in matches
        ]

        # Build replacement operators for every entity type present in matches
        active_entities = {m.entity_type for m in matches}
        replacements = {**self.custom_operators, **(custom_replacements or {})}

        operators = {
            ent: OperatorConfig(
                "replace",
                {
                    "new_value": replacements.get(
                        ent, default_redaction_token(ent)
                    )
                },
            )
            for ent in active_entities
        }

        anonymized_result = self._engine.anonymize(
            text=text,
            analyzer_results=presidio_results,
            operators=operators,
        )
        return anonymized_result.text


_default_anonymizer: Optional[PIIAnonymizer] = None


def get_default_anonymizer() -> PIIAnonymizer:
    """Returns the shared lazy-loaded PIIAnonymizer singleton."""
    global _default_anonymizer
    if _default_anonymizer is None:
        _default_anonymizer = PIIAnonymizer()
    return _default_anonymizer
