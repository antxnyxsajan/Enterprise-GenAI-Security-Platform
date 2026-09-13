"""Unicode normalization and invisible character sanitization.

Handles:
1. NFKC compatibility normalization.
2. Full-spectrum invisible character stripping (Zero-Width, Unicode Tag ASCII smuggling,
   Bidi overrides, variation selectors).
3. Syntax-preserving whitespace normalization (protects Python/YAML indentation and code).
4. Runaway newline capping (mitigating prompt-push / context displacement attacks).
"""

import re
import unicodedata
from typing import List, Tuple

# Full-spectrum invisible characters regex
# Covers:
# - Zero-Width spaces & joiners: \u200b, \u200c, \u200d, \u2060, \ufeff
# - Left-to-Right & Right-to-Left marks: \u200e, \u200f
# - Soft hyphen: \u00ad
# - Invisible separators: \u2061, \u2062, \u2063, \u2064
# - Mongolian vowel separator: \u180e
# - Braille blank space (frequently used as invisible filler): \u2800
# - Variation selectors: \ufe00-\ufe0f, \U000e0100-\U000e01ef
INVISIBLE_CHARS_REGEX = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u180e\u2060-\u2064\u2800\ufe00-\ufe0f]"
    r"|[\U000e0100-\U000e01ef]"
)

# Unicode Tag characters (U+E0000 - U+E007F) used for ASCII smuggling attacks
UNICODE_TAGS_REGEX = re.compile(r"[\U000e0000-\U000e007f]")

# Bidirectional overrides and isolate characters
# LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI
BIDI_OVERRIDES_REGEX = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

# Non-printable control characters (excluding standard \t and \n)
CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Runaway newlines (3 or more consecutive newlines)
RUNAWAY_NEWLINES_REGEX = re.compile(r"\n{3,}")

# Intra-line multiple horizontal whitespace (excluding leading indentation)
INTRA_LINE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def sanitize_invisible_characters(text: str) -> Tuple[str, int, List[str]]:
    """Strips invisible characters, Unicode tag characters, and Bidi overrides.
    
    Returns:
        Tuple[str, int, List[str]]: (cleaned_text, total_removed_count, evasion_flags)
    """
    flags: List[str] = []
    total_removed = 0

    # 1. Detect and strip Unicode Tag plane (ASCII smuggling)
    tag_matches = len(UNICODE_TAGS_REGEX.findall(text))
    if tag_matches > 0:
        total_removed += tag_matches
        flags.append("ASCII_SMUGGLING_TAGS_DETECTED")
        text = UNICODE_TAGS_REGEX.sub("", text)

    # 2. Detect and strip Bidirectional overrides (RLO / LRO log spoofing)
    bidi_matches = len(BIDI_OVERRIDES_REGEX.findall(text))
    if bidi_matches > 0:
        total_removed += bidi_matches
        flags.append("BIDI_OVERRIDE_DETECTED")
        text = BIDI_OVERRIDES_REGEX.sub("", text)

    # 3. Detect and strip other invisible characters (Zero-width, variation selectors)
    invisible_matches = len(INVISIBLE_CHARS_REGEX.findall(text))
    if invisible_matches > 0:
        total_removed += invisible_matches
        flags.append("ZERO_WIDTH_CHARS_DETECTED")
        text = INVISIBLE_CHARS_REGEX.sub("", text)

    return text, total_removed, flags


def normalize_whitespace_and_syntax(text: str) -> Tuple[str, List[str]]:
    """Normalizes whitespace while strictly preserving code indentation and linebreaks.
    
    - Standardizes Windows (CRLF) and legacy Mac (CR) line endings to LF.
    - Strips non-printable ASCII control characters.
    - Caps runaway consecutive newlines to maximum 2 (preventing prompt displacement attacks).
    - Collapses intra-line redundant whitespace while keeping leading line indentation.
    
    Returns:
        Tuple[str, List[str]]: (cleaned_text, evasion_flags)
    """
    flags: List[str] = []

    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip control characters
    text = CONTROL_CHARS_REGEX.sub("", text)

    # Detect runaway newlines (e.g., 50 newlines pushing system prompts out of view)
    if RUNAWAY_NEWLINES_REGEX.search(text):
        flags.append("RUNAWAY_NEWLINES_DETECTED")
        text = RUNAWAY_NEWLINES_REGEX.sub("\n\n", text)

    # Process line by line to preserve code indentation (e.g., Python / YAML)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        
        # Extract leading whitespace (indentation)
        leading_ws_len = len(line) - len(line.lstrip(" \t"))
        indentation = line[:leading_ws_len]
        content = line[leading_ws_len:]
        
        # Collapse internal runs of spaces/tabs in the line content only
        content = INTRA_LINE_MULTI_SPACE.sub(" ", content)
        # Strip trailing whitespace on line
        cleaned_lines.append(indentation + content.rstrip(" \t"))

    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text.strip(), flags


def normalize_unicode(text: str) -> str:
    """Applies NFKC compatibility decomposition and canonical composition."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)
