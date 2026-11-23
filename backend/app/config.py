"""Centralized configuration for Enterprise GenAI Security Platform.

Reads overrides from environment variables while providing safe local defaults.
"""

from dataclasses import dataclass
import os
from pathlib import Path

# Resolve backend root directory cleanly regardless of current working directory
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    """Application settings and tunable security thresholds."""

    # Module 2: Hybrid Prompt Firewall
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    classifier_model_name: str = os.getenv(
        "CLASSIFIER_MODEL_NAME", "protectai/deberta-v3-base-prompt-injection"
    )
    classifier_max_length: int = int(os.getenv("CLASSIFIER_MAX_LENGTH", "512"))

    qdrant_storage_path: str = os.getenv(
        "QDRANT_STORAGE_PATH", str(DEFAULT_DATA_DIR / "qdrant_data")
    )
    qdrant_collection_name: str = os.getenv(
        "QDRANT_COLLECTION_NAME", "jailbreak_signatures"
    )
    qdrant_top_k: int = int(os.getenv("QDRANT_TOP_K", "3"))

    # Tunable detection thresholds (to be calibrated during empirical testing)
    qdrant_risk_threshold: float = float(os.getenv("QDRANT_RISK_THRESHOLD", "0.85"))
    ml_risk_threshold: float = float(os.getenv("ML_RISK_THRESHOLD", "0.90"))

    # Module 3: Data Protection Engine (DPE)
    spacy_model_name: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_lg")
    dpe_score_threshold: float = float(os.getenv("DPE_SCORE_THRESHOLD", "0.60"))
    dpe_entities: tuple = (
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_SSN",
        "PERSON",
        "LOCATION",
    )


settings = Settings()
