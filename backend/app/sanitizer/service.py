"""Sanitizer service orchestrator for Module 1.

Orchestrates:
1. NFKC Unicode normalization.
2. Cross-script confusable homoglyph translation.
3. Full-spectrum invisible character, Unicode tag, and Bidi override stripping.
4. Syntax-preserving whitespace cleaning and runaway newline mitigation.
5. Adversarial suffix detection & heuristic stripping.
6. Generation of structured SanitizationResult telemetry.
"""

from typing import List

from .confusables import normalize_confusables
from .normalize import (
    normalize_unicode,
    normalize_whitespace_and_syntax,
    sanitize_invisible_characters,
)
from .schemas import SanitizationResult
from .suffix_strip import detect_and_strip_adversarial_suffix


def sanitize_prompt(raw_prompt: str) -> SanitizationResult:
    """Sanitizes an incoming raw prompt and extracts security evasion telemetry.
    
    Args:
        raw_prompt: The raw user prompt string received by the API.
        
    Returns:
        SanitizationResult: Cleaned text ready for downstream modules (Firewall,
                            Presidio, LLM) along with evasion flags and metrics.
    """
    if not raw_prompt or not raw_prompt.strip():
        return SanitizationResult(
            original_text=raw_prompt or "",
            cleaned_text="",
            was_modified=bool(raw_prompt),
            invisible_chars_count=0,
            homoglyphs_replaced_count=0,
            suspicious_suffix_detected=False,
            stripped_suffix="",
            evasion_flags=[],
        )

    flags: List[str] = []

    # Step 1: Unicode NFKC compatibility normalization
    normalized = normalize_unicode(raw_prompt)

    # Step 2: Cross-script confusable homoglyph mapping (Cyrillic/Greek to Latin)
    confusable_cleaned, homoglyphs_count = normalize_confusables(normalized)
    if homoglyphs_count > 0:
        flags.append("HOMOGLYPH_SPOOFING_DETECTED")

    # Step 3: Full-spectrum invisible characters, Unicode tags (ASCII smuggling), Bidi
    invisible_cleaned, invisible_count, invisible_flags = sanitize_invisible_characters(
        confusable_cleaned
    )
    flags.extend(invisible_flags)

    # Step 4: Syntax-preserving whitespace & control character normalization
    ws_cleaned, ws_flags = normalize_whitespace_and_syntax(invisible_cleaned)
    flags.extend(ws_flags)

    # Step 5: Adversarial suffix detection & stripping
    (
        final_cleaned,
        suffix_detected,
        stripped_suffix,
        suffix_flags,
    ) = detect_and_strip_adversarial_suffix(ws_cleaned)
    flags.extend(suffix_flags)

    # Deduplicate flags while preserving order
    unique_flags = list(dict.fromkeys(flags))

    was_modified = final_cleaned != raw_prompt

    return SanitizationResult(
        original_text=raw_prompt,
        cleaned_text=final_cleaned,
        was_modified=was_modified,
        invisible_chars_count=invisible_count,
        homoglyphs_replaced_count=homoglyphs_count,
        suspicious_suffix_detected=suffix_detected,
        stripped_suffix=stripped_suffix,
        evasion_flags=unique_flags,
    )
