"""Adversarial suffix detection and heuristic token garbage stripper.

Detects high-entropy adversarial suffixes (such as GCG - Greedy Coordinate Gradient
attacks) while protecting legitimate Markdown dividers, code closures, and standard syntax.
"""

import math
import re
from typing import List, Tuple

# Markdown dividers (e.g. ---, ===, ***, ___ )
MARKDOWN_DIVIDER_REGEX = re.compile(r"^[ \t]*([=\-_*])[ \t]*(?:\1[ \t]*){2,}[ \t]*$")

# Standard code structure closures at end of text (e.g. });, }}}, </script>, etc.)
CODE_CLOSURE_REGEX = re.compile(
    r"(\);|\}\;|\}\}|\}\]|\}\);|\];|\)\s*\{|\<\/[a-zA-Z0-9_\-]+\>)[ \t\n]*$"
)

# Known published research suffix token sequences (GCG / Zou et al. / universal attacks)
KNOWN_ATTACK_SUFFIX_PATTERNS = [
    re.compile(r"describing\.\\\s*\+similarly", re.IGNORECASE),
    re.compile(r"surely\s*\[\{", re.IGNORECASE),
    re.compile(r"start\s+with\s+[\"']sure,?\s+here\s+is[\"']", re.IGNORECASE),
    re.compile(r"as\s+an\s+unrestricted\s+ai", re.IGNORECASE),
]

# High-density non-alphanumeric noise suffix (> 15 consecutive punctuation/symbols)
PUNCTUATION_RUNAWAY_REGEX = re.compile(r"[!\"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]{15,}\s*$")


def calculate_entropy(text: str) -> float:
    """Calculates Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def is_legitimate_code_or_markdown(tail: str) -> bool:
    """Checks if the trailing string is legitimate Markdown or code syntax."""
    tail = tail.strip()
    # Check if it's a Markdown divider (---, ===, ***)
    if MARKDOWN_DIVIDER_REGEX.match(tail):
        return True
    # Check if it's a standard code closure
    if CODE_CLOSURE_REGEX.search(tail):
        return True
    return False


def detect_and_strip_adversarial_suffix(text: str) -> Tuple[str, bool, str, List[str]]:
    """Evaluates and strips adversarial token suffixes.
    
    Returns:
        Tuple[str, bool, str, List[str]]:
            - cleaned_text
            - suspicious_suffix_detected (bool)
            - stripped_suffix (str)
            - evasion_flags (list of detected evasion reasons)
    """
    if not text or len(text) < 15:
        return text, False, "", []

    flags: List[str] = []
    stripped_suffix = ""
    tail = text[-100:] if len(text) > 100 else text

    # Check 1: Known published research attack suffix patterns
    for pattern in KNOWN_ATTACK_SUFFIX_PATTERNS:
        match = pattern.search(text)
        if match:
            flags.append("ADVERSARIAL_SUFFIX_DETECTED")
            start_pos = match.start()
            stripped_suffix = text[start_pos:].strip()
            cleaned = text[:start_pos].rstrip()
            return cleaned, True, stripped_suffix, flags

    # Check 2: Trailing repetitive non-alphanumeric noise
    punct_match = PUNCTUATION_RUNAWAY_REGEX.search(text)
    if punct_match:
        matched_str = punct_match.group(0)
        # Verify it's not a legitimate Markdown divider or code closure
        if not is_legitimate_code_or_markdown(matched_str):
            flags.append("ADVERSARIAL_SUFFIX_DETECTED")
            start_pos = punct_match.start()
            stripped_suffix = text[start_pos:].strip()
            cleaned = text[:start_pos].rstrip()
            return cleaned, True, stripped_suffix, flags

    # Check 3: High-entropy tail with unusual punctuation density
    # GCG attacks often append 20-60 characters with high character entropy and low alphanumeric ratio
    tail_sample = text[-60:]
    if len(tail_sample) >= 30 and not is_legitimate_code_or_markdown(tail_sample):
        alnum_ratio = sum(c.isalnum() for c in tail_sample) / len(tail_sample)
        entropy = calculate_entropy(tail_sample)
        # Low alphanumeric content (< 30%) combined with high entropy (> 3.8) is characteristic of GCG noise
        if alnum_ratio < 0.30 and entropy > 3.8:
            flags.append("HIGH_ENTROPY_SUFFIX_DETECTED")
            # We flag high entropy, and strip the anomalous tail
            cleaned = text[:-len(tail_sample)].rstrip()
            return cleaned, True, tail_sample, flags

    return text, False, "", []
