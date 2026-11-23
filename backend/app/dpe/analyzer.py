"""Thread-safe lazy-loaded Presidio AnalyzerEngine wrapper for Module 3.

Handles:
1. Lazy initialization of Presidio AnalyzerEngine with spaCy NLP pipeline (default: en_core_web_lg).
2. Graceful fallback if en_core_web_lg is not installed (falls back to en_core_web_sm if available).
3. Configurable entity list and score confidence thresholding.
4. Converting raw Presidio RecognizerResults into clean PIIEntityMatch schemas.
"""

import logging
import threading
from typing import Any, List, Optional, Sequence

from app.config import settings
from app.dpe.schemas import PIIEntityMatch

logger = logging.getLogger(__name__)


def create_analyzer_engine(model_name: str = "en_core_web_lg"):
    """Instantiates Presidio AnalyzerEngine with custom NlpEngineProvider."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Attempt to configure the requested spaCy model
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
        )
        nlp_engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception as exc:
        logger.warning(
            "Failed to load requested spaCy model '%s': %s. Attempting fallback.",
            model_name,
            exc,
        )

    # Fallback attempt 1: en_core_web_sm
    if model_name != "en_core_web_sm":
        try:
            logger.info("Attempting fallback to 'en_core_web_sm'...")
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            nlp_engine = provider.create_engine()
            return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        except Exception as fallback_exc:
            logger.warning("Fallback to en_core_web_sm failed: %s", fallback_exc)

    # Default fallback: default AnalyzerEngine (uses whatever is available)
    logger.info("Using standard default AnalyzerEngine configuration.")
    return AnalyzerEngine()


class PIIAnalyzer:
    """Thread-safe, lazy-loaded wrapper around Presidio AnalyzerEngine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_entities: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ):
        self.model_name = model_name or settings.spacy_model_name
        self.default_entities = list(default_entities or settings.dpe_entities)
        self.score_threshold = (
            score_threshold if score_threshold is not None else settings.dpe_score_threshold
        )
        self._engine = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        """Lazily initializes the Presidio AnalyzerEngine."""
        if self._engine is not None:
            return
        with self._lock:
            if self._engine is not None:
                return
            logger.info("Initializing Presidio Analyzer with model '%s'", self.model_name)
            self._engine = create_analyzer_engine(self.model_name)

    def analyze(
        self,
        text: str,
        entities: Optional[Sequence[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[PIIEntityMatch]:
        """Scans prompt text and returns all detected PII entity matches."""
        if not text or not text.strip():
            return []

        self.load()
        target_entities = list(entities or self.default_entities)
        threshold = self.score_threshold if score_threshold is None else score_threshold

        raw_results = self._engine.analyze(
            text=text,
            entities=target_entities,
            language="en",
            score_threshold=threshold,
        )

        matches: List[PIIEntityMatch] = []
        for r in raw_results:
            matched_slice = text[r.start : r.end]
            matches.append(
                PIIEntityMatch(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=float(r.score),
                    text=matched_slice,
                )
            )

        # Sort matches by start position for deterministic ordering
        matches.sort(key=lambda m: (m.start, m.end))
        return matches


_default_analyzer: Optional[PIIAnalyzer] = None


def get_default_analyzer() -> PIIAnalyzer:
    """Returns the shared lazy-loaded PIIAnalyzer singleton."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = PIIAnalyzer()
    return _default_analyzer
