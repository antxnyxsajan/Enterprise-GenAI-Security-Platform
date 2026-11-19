"""Attack signature seeding utility for Module 2 (Qdrant).

Loads known prompt injection / jailbreak signatures from JSONL files (both the bundled
offline baseline set and external datasets such as JBB-Behaviors), normalizes them via
Module 1, embeds them in batches, and idempotently upserts them into Qdrant.

Usage (from backend/):
    python -m app.firewall.seed_signatures
    python -m app.firewall.seed_signatures --file path/to/jbb_behaviors.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from app.firewall.embedder import PromptEmbedder, get_default_embedder
from app.firewall.qdrant_store import QdrantSignatureStore
from app.sanitizer import sanitize_prompt

DEFAULT_SEED_FILE = Path(__file__).resolve().parent / "seed_data" / "baseline_signatures.jsonl"

PROMPT_FIELD_CANDIDATES = ("prompt", "goal", "Goal", "text", "behavior", "Behavior")
CATEGORY_FIELD_CANDIDATES = ("category", "Category", "type", "attack_type", "label")


def _extract_first_non_empty(row: Dict, candidates: tuple, default: str = "") -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def load_signatures_from_jsonl(
    path: Union[str, Path] = DEFAULT_SEED_FILE,
    source_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Loads attack signatures from a JSONL file supporting multiple dataset schemas."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Seed signature file not found: {file_path}")

    resolved_source = source_name or file_path.stem
    signatures: List[Dict[str, str]] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {file_path}: {exc}") from exc

            prompt = _extract_first_non_empty(row, PROMPT_FIELD_CANDIDATES)
            if not prompt:
                continue

            category = _extract_first_non_empty(row, CATEGORY_FIELD_CANDIDATES, default="unknown")
            signatures.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "source": resolved_source,
                }
            )

    return signatures


def seed_store(
    store: QdrantSignatureStore,
    embedder: PromptEmbedder,
    jsonl_path: Union[str, Path] = DEFAULT_SEED_FILE,
    normalize_before_embed: bool = True,
) -> int:
    """Loads signatures from JSONL, embeds them in batch, and upserts them into Qdrant.

    Returns:
        Number of signatures upserted into Qdrant.
    """
    raw_signatures = load_signatures_from_jsonl(jsonl_path)
    if not raw_signatures:
        return 0

    prepared_signatures: List[Dict[str, str]] = []
    texts_to_embed: List[str] = []

    for sig in raw_signatures:
        text = sig["prompt"]
        if normalize_before_embed:
            text = sanitize_prompt(text).cleaned_text
        if not text:
            continue
        prepared_signatures.append(
            {
                "prompt": text,
                "category": sig["category"],
                "source": sig["source"],
            }
        )
        texts_to_embed.append(text)

    if not prepared_signatures:
        return 0

    vectors = embedder.embed_batch(texts_to_embed)
    return store.upsert_signatures(prepared_signatures, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed local Qdrant signature store with known jailbreak/injection prompts."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_SEED_FILE),
        help="Path to JSONL seed file (defaults to bundled baseline_signatures.jsonl).",
    )
    args = parser.parse_args()

    embedder = get_default_embedder()
    with QdrantSignatureStore(in_memory=False) as store:
        upserted = seed_store(store, embedder, jsonl_path=args.file)
        total = store.count()
        print(f"Seeded {upserted} signatures from {args.file}")
        print(f"Qdrant collection '{store.collection_name}' now contains {total} total signatures.")


if __name__ == "__main__":
    main()
