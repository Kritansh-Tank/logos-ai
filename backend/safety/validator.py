"""
validator.py — Hallucination detection for scripture references.
Scans LLM response for citation patterns like "John 3:16" or "Genesis 1:1"
and verifies each against the actual corpus. Flags unverified references.

Uses an explicit list of all 66 KJV canonical book names to avoid
false positives from words like "In Psalms 23:1" (where "In" would
be incorrectly captured as part of the book name by a generic regex).
"""

import re
from rag.retriever import retriever

# ---------------------------------------------------------------------------
# Canonical KJV book names — explicit list to avoid false regex matches
# ---------------------------------------------------------------------------
_BOOK_NAMES = (
    r"Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
    r"1\s*Samuel|2\s*Samuel|1\s*Kings|2\s*Kings|"
    r"1\s*Chronicles|2\s*Chronicles|Ezra|Nehemiah|Esther|Job|"
    r"Psalms?|Proverbs|Ecclesiastes|Song\s+of\s+Solomon|"
    r"Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|"
    r"Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|"
    r"Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|"
    r"Matthew|Mark|Luke|John|Acts|Romans|"
    r"1\s*Corinthians|2\s*Corinthians|Galatians|Ephesians|"
    r"Philippians|Colossians|1\s*Thessalonians|2\s*Thessalonians|"
    r"1\s*Timothy|2\s*Timothy|Titus|Philemon|Hebrews|James|"
    r"1\s*Peter|2\s*Peter|1\s*John|2\s*John|3\s*John|Jude|Revelation"
)

# Matches: "John 3:16", "1 Corinthians 13:4", "Song of Solomon 2:1"
# Word boundary ensures we don't match mid-word; explicit book list prevents
# prepositions like "In" or "As" from being captured as a book name.
REFERENCE_PATTERN = re.compile(
    r'\b(' + _BOOK_NAMES + r')\s+(\d{1,3}):(\d{1,3})\b',
    re.IGNORECASE
)


def validate_response(response_text: str) -> dict:
    """
    Scan response for scripture references and verify against corpus.

    Returns:
        {
          "is_clean": bool,
          "flagged_references": list[str],   # references NOT found in corpus
          "verified_references": list[str],
          "warning_message": str | None
        }
    """
    matches = REFERENCE_PATTERN.findall(response_text)

    if not matches:
        return {
            "is_clean": True,
            "flagged_references": [],
            "verified_references": [],
            "warning_message": None
        }

    flagged  = []
    verified = []
    seen     = set()

    for book, chapter, verse in matches:
        # Normalise whitespace inside book name (e.g. "1  Kings" → "1 Kings")
        book = re.sub(r'\s+', ' ', book).strip()
        reference = f"{book} {chapter}:{verse}"
        if reference in seen:
            continue
        seen.add(reference)

        if retriever.verse_exists(reference):
            verified.append(reference)
        else:
            flagged.append(reference)

    warning = None
    if flagged:
        refs_str = ", ".join(flagged)
        warning = (
            f"Verification note: The reference(s) {refs_str} could not be "
            f"verified in my grounded KJV corpus. Please double-check these with a Bible."
        )

    return {
        "is_clean": len(flagged) == 0,
        "flagged_references": flagged,
        "verified_references": verified,
        "warning_message": warning
    }
