"""Lazy-loaded DeBERTa-v3 prompt injection classifier for Module 2.

Uses `protectai/deberta-v3-base-prompt-injection` with:
- Thread-safe lazy loading (no import-time downloads or VRAM allocation)
- Dynamic `id2label` introspection (prevents inverted risk scoring)
- Automatic CUDA GPU acceleration with seamless CPU fallback
"""

import logging
import threading
from typing import Any, Dict, Optional

from app.config import settings
from app.firewall.embedder import resolve_torch_device

logger = logging.getLogger(__name__)

POSITIVE_LABEL_KEYWORDS = (
    "INJECTION",
    "JAILBREAK",
    "MALICIOUS",
    "UNSAFE",
    "ATTACK",
    "ADVERSARIAL",
)

BENIGN_LABEL_KEYWORDS = (
    "SAFE",
    "BENIGN",
    "LEGIT",
    "NORMAL",
    "CLEAN",
    "HAM",
)


def resolve_injection_label_index(id2label: Optional[Dict[Any, str]]) -> int:
    """Dynamically resolves the logits index corresponding to the injection class.

    Prevents silent score inversion if a model checkpoint maps index 0 vs index 1
    differently from expectations.
    """
    if not id2label:
        raise ValueError("Model config is missing 'id2label'; cannot determine injection label index.")

    normalized = {int(idx): str(label).strip().upper() for idx, label in id2label.items()}

    # Pass 1: Explicit semantic match for injection/malicious class
    for idx, label in normalized.items():
        if any(keyword in label for keyword in POSITIVE_LABEL_KEYWORDS):
            return idx

    # Pass 2: Standard Hugging Face generic binary convention (LABEL_1)
    for idx, label in normalized.items():
        if label in ("LABEL_1", "1", "POSITIVE", "TRUE"):
            return idx

    # Pass 3: Binary model where only the benign class is named explicitly
    if len(normalized) == 2:
        benign_indices = [
            idx
            for idx, label in normalized.items()
            if any(keyword in label for keyword in BENIGN_LABEL_KEYWORDS)
        ]
        if len(benign_indices) == 1:
            other_indices = [idx for idx in normalized if idx != benign_indices[0]]
            return other_indices[0]

    raise ValueError(
        f"Unable to unambiguously resolve injection label index from id2label={id2label}"
    )


class PromptClassifier:
    """Thread-safe, lazy-loaded DeBERTa-v3 sequence classifier."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = model_name or settings.classifier_model_name
        self._preferred_device = device
        self.max_length = max_length or settings.classifier_max_length
        self._device: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._injection_idx: Optional[int] = None
        self._id2label: Dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_torch_device(self._preferred_device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(self._id2label)

    @property
    def injection_label_index(self) -> Optional[int]:
        return self._injection_idx

    def load(self) -> None:
        """Loads tokenizer and sequence classification weights lazily."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            target_device = self.device
            logger.info(
                "Loading prompt injection classifier '%s' on device '%s'",
                self.model_name,
                target_device,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(target_device)
            self._model.eval()

            raw_id2label = getattr(self._model.config, "id2label", None)
            self._injection_idx = resolve_injection_label_index(raw_id2label)
            self._id2label = {int(k): str(v) for k, v in raw_id2label.items()}
            logger.info(
                "Classifier ready on '%s' | id2label=%s | injection_idx=%d",
                target_device,
                self._id2label,
                self._injection_idx,
            )

    def classify(self, text: str) -> float:
        """Returns probability in [0.0, 1.0] that the prompt is a prompt injection/jailbreak."""
        if not text or not text.strip():
            return 0.0

        self.load()
        import torch

        with torch.inference_mode():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            score = float(probs[0, self._injection_idx].item())
            return max(0.0, min(1.0, score))


_default_classifier: Optional[PromptClassifier] = None


def get_default_classifier() -> PromptClassifier:
    """Returns the shared lazy-loaded default PromptClassifier singleton."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = PromptClassifier()
    return _default_classifier
