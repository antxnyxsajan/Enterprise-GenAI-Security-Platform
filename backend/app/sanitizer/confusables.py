"""Cross-script confusable homoglyph mapping.

Maps Cyrillic, Greek, and other cross-script lookalike characters to their
Latin visual equivalents to prevent token-splitting evasion in downstream
tokenizers (DeBERTa, Sentence-Transformers).
"""

# Map of visually identical or near-identical non-Latin characters to Latin characters
CONFUSABLE_MAP = {
    # Cyrillic lowercase
    "\u0430": "a",  # Cyrillic Small Letter A
    "\u0441": "c",  # Cyrillic Small Letter Es
    "\u0435": "e",  # Cyrillic Small Letter Ie
    "\u0451": "e",  # Cyrillic Small Letter Io
    "\u0456": "i",  # Cyrillic Small Letter Byelorussian-Ukrainian I
    "\u0458": "j",  # Cyrillic Small Letter Je
    "\u043e": "o",  # Cyrillic Small Letter O
    "\u0440": "p",  # Cyrillic Small Letter Er
    "\u0455": "s",  # Cyrillic Small Letter Dze
    "\u0445": "x",  # Cyrillic Small Letter Ha
    "\u0443": "y",  # Cyrillic Small Letter U
    "\u04bb": "h",  # Cyrillic Small Letter Shha
    "\u0501": "d",  # Cyrillic Small Letter Komi De
    "\u051b": "q",  # Cyrillic Small Letter Qa
    "\u051d": "w",  # Cyrillic Small Letter We
    
    # Cyrillic uppercase
    "\u0410": "A",  # Cyrillic Capital Letter A
    "\u0412": "B",  # Cyrillic Capital Letter Ve
    "\u0421": "C",  # Cyrillic Capital Letter Es
    "\u0415": "E",  # Cyrillic Capital Letter Ie
    "\u0401": "E",  # Cyrillic Capital Letter Io
    "\u041d": "H",  # Cyrillic Capital Letter En
    "\u0406": "I",  # Cyrillic Capital Letter Byelorussian-Ukrainian I
    "\u0408": "J",  # Cyrillic Capital Letter Je
    "\u041a": "K",  # Cyrillic Capital Letter Ka
    "\u041c": "M",  # Cyrillic Capital Letter Em
    "\u041e": "O",  # Cyrillic Capital Letter O
    "\u0420": "P",  # Cyrillic Capital Letter Er
    "\u0422": "T",  # Cyrillic Capital Letter Te
    "\u0425": "X",  # Cyrillic Capital Letter Ha
    "\u0423": "Y",  # Cyrillic Capital Letter U
    "\u0405": "S",  # Cyrillic Capital Letter Dze

    # Greek lowercase
    "\u03b1": "a",  # Greek Small Letter Alpha
    "\u03b2": "b",  # Greek Small Letter Beta
    "\u03b5": "e",  # Greek Small Letter Epsilon
    "\u03b9": "i",  # Greek Small Letter Iota
    "\u03ba": "k",  # Greek Small Letter Kappa
    "\u03bf": "o",  # Greek Small Letter Omicron
    "\u03c1": "p",  # Greek Small Letter Rho
    "\u03c5": "u",  # Greek Small Letter Upsilon
    "\u03bd": "v",  # Greek Small Letter Nu
    "\u03c7": "x",  # Greek Small Letter Chi

    # Greek uppercase
    "\u0391": "A",  # Greek Capital Letter Alpha
    "\u0392": "B",  # Greek Capital Letter Beta
    "\u0395": "E",  # Greek Capital Letter Epsilon
    "\u0396": "Z",  # Greek Capital Letter Zeta
    "\u0397": "H",  # Greek Capital Letter Eta
    "\u0399": "I",  # Greek Capital Letter Iota
    "\u039a": "K",  # Greek Capital Letter Kappa
    "\u039c": "M",  # Greek Capital Letter Mu
    "\u039d": "N",  # Greek Capital Letter Nu
    "\u039f": "O",  # Greek Capital Letter Omicron
    "\u03a1": "P",  # Greek Capital Letter Rho
    "\u03a4": "T",  # Greek Capital Letter Tau
    "\u03a5": "Y",  # Greek Capital Letter Upsilon
    "\u03a7": "X",  # Greek Capital Letter Chi
}

# Translation table for O(N) mapping
_TRANSLATION_TABLE = str.maketrans(CONFUSABLE_MAP)


def normalize_confusables(text: str) -> tuple[str, int]:
    """Replaces cross-script confusable homoglyphs with their Latin equivalents.
    
    Returns:
        tuple[str, int]: (normalized_text, count_of_replaced_homoglyphs)
    """
    if not text:
        return text, 0

    count = 0
    # Count occurrences
    for char in text:
        if char in CONFUSABLE_MAP:
            count += 1

    if count == 0:
        return text, 0

    return text.translate(_TRANSLATION_TABLE), count
