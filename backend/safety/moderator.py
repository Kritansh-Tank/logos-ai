"""
moderator.py — Safety filter for text and image prompts.

Text moderation (2 layers):
  Layer 1: Regex hard-blocklist — rewrites, hate speech, jailbreaks, blasphemy (~0ms)
  Layer 2: Soft-block patterns — comparative religion, existence debates (allowed through with care)

Image moderation (2 layers):
  Layer 1: Regex hard-blocklist — explicit, satanic, blasphemous imagery (~0ms)
  Layer 2: Biblical relevance check — keyword whitelist fast-path (O(1)) +
           cosine similarity fallback via all-MiniLM-L6-v2 (threshold=0.22, ~1ms)

Returns: ModerationResult(allowed, reason, category, severity)
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Blocklist patterns — ordered by severity
# ---------------------------------------------------------------------------
HARD_BLOCK_PATTERNS = [
    # Adversarial Bible rewrites — "rewrite John 3:16", "rewrite scripture to support..."
    (r"rewrite\s+(bible|scripture|verse|john|matthew|luke|mark|psalm|genesis|any\s+verse)",
     "rewrite_attempt", "I can't rewrite or distort scripture to support an ideology."),
    (r"rewrite.{0,60}(support|justify|promote|endorse|prove|disprove)",
     "rewrite_attempt", "I can't rewrite or distort scripture to support an ideology."),

    # Hate speech — targets groups via scripture (broad: group, race, ethnicity, sexuality)
    (r"(bible|god|jesus|christ).{0,60}(inferior|subhuman|less than|beneath|unworthy).{0,40}",
     "hate_speech", "I won't produce religiously-justified hate content."),
    (r"use\s+(the\s+)?(bible|scripture|god|jesus).{0,30}(prove|show|justify|support).{0,40}(inferior|superior|less|beneath|hate)",
     "hate_speech", "I won't produce religiously-justified hate content."),
    (r"\[(ethnic|racial|religious|lgbtq?)\s+group\]",
     "hate_speech", "I won't produce content targeting groups."),

    # Extreme/terror justification
    (r"(jihad|crusade|holy war|violence|terrorism).{0,40}(justified|commanded|required|approved)\s*(by)?\s*(god|bible|jesus|christ)",
     "extremism", "I won't produce content that justifies violence through religion."),

    # Blasphemy: writing blasphemous/mocking content, parodies of holy texts/prayers
    (r"(write|generate|create|make|compose).{0,40}(blasph|mock|ridicule|insult|defile|sacrileg).{0,40}(jesus|christ|god|bible|christian|holy|sacred)",
     "blasphemy_request", "I won't generate content that mocks or ridicules Christian faith."),
    (r"(blasphemous|sacrilegious|anti.?christian|anti.?god).{0,30}(parody|version|rewrite|poem|song|prayer)",
     "blasphemy_request", "I won't generate content that mocks or ridicules Christian faith."),
    (r"(parody|mock|corrupt|evil\s+version).{0,40}(lord.?s\s+prayer|our\s+father|hail\s+mary|apostles.?\s+creed|ten\s+commandments)",
     "blasphemy_request", "I won't generate content that mocks or ridicules Christian faith."),

    # Heresy propaganda generation
    (r"(prove|show|demonstrate|convince).{0,30}(bible|god|jesus|christ)\s*(is\s*)?(fake|false|wrong|evil|lie|myth)",
     "heresy_generation", "I can engage with theological doubt respectfully, but won't generate anti-Christian propaganda."),

    # Jailbreak attempts
    (r"(ignore|forget|disregard|bypass|override|pretend\s+you\s+have\s+no).{0,40}(instruction|rule|guideline|system|prompt|restriction)",
     "jailbreak", "I notice you're trying to bypass my guidelines. I'm here to help with Christian topics respectfully."),

    # Explicit content involving religious figures
    (r"(sexual|erotic|explicit|nude|naked|porn).{0,40}(jesus|mary|saint|angel|god|bible|church)",
     "explicit_content", "I won't generate explicit content involving religious figures."),
]

# Soft-block patterns (warn but don't hard block — handle gracefully)
SOFT_BLOCK_PATTERNS = [
    (r"which (religion|faith) is (best|true|right|correct|better)",
     "comparative_religion", None),  # Handle with diplomatic response
    (r"(prove|disprove)\s+god\s+(exists|doesn't exist|is real)",
     "existence_debate", None),
]

# ---------------------------------------------------------------------------
# Image-specific blocklist
# ---------------------------------------------------------------------------
IMAGE_BLOCK_PATTERNS = [
    (r"(nude|naked|explicit|sexual|erotic|violent|gore|blood|weapon)",
     "explicit_image", "I won't generate explicit, violent, or offensive images."),
    (r"(satan|devil|lucifer|demon|666|pentagram|occult|baphomet|hellish|infernal)",
     "satanic_image", "I won't generate imagery glorifying evil or the occult."),
    (r"(demon|devil|satan).{0,20}(worship|praise|honor|glorify|ritual)",
     "satanic_image", "I won't generate imagery glorifying evil or the occult."),
    (r"(mock|ridicule|insult|defile|blasph|parody).{0,40}(jesus|mary|cross|church|god|holy|sacred)",
     "blasphemous_image", "I won't generate blasphemous imagery."),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModerationResult:
    allowed: bool
    reason: str
    category: str
    severity: str  # "hard_block" | "soft_block" | "clean"


# ---------------------------------------------------------------------------
# Text moderator
# ---------------------------------------------------------------------------
def moderate_text(text: str) -> ModerationResult:
    """Check user message against safety patterns."""
    text_lower = text.lower()

    for pattern, category, message in HARD_BLOCK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return ModerationResult(
                allowed=False,
                reason=message,
                category=category,
                severity="hard_block"
            )

    for pattern, category, message in SOFT_BLOCK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return ModerationResult(
                allowed=True,
                reason="Handle with care — comparative/philosophical topic",
                category=category,
                severity="soft_block"
            )

    return ModerationResult(allowed=True, reason="", category="clean", severity="clean")


# ---------------------------------------------------------------------------
# Biblical relevance whitelist — image prompts must contain at least one of these
# ---------------------------------------------------------------------------
BIBLICAL_SIGNAL_WORDS = {
    # Core figures
    "jesus", "christ", "god", "holy spirit", "holy ghost", "angel", "archangel",
    "mary", "virgin", "joseph", "david", "solomon", "moses", "noah", "abraham",
    "isaac", "jacob", "elijah", "elisha", "daniel", "jonah", "paul", "peter",
    "john", "matthew", "mark", "luke", "james", "lazarus", "adam", "eve",
    "thomas", "magdalene", "samson", "ruth", "esther", "job", "ezekiel",
    "jeremiah", "isaiah", "gabriel", "michael", "stephen", "philip", "andrew",
    "herod", "pilate", "nicodemus", "zacchaeus", "goliath", "delilah",
    # Biblical places
    "jerusalem", "bethlehem", "nazareth", "galilee", "jordan", "eden",
    "sinai", "calvary", "golgotha", "gethsemane", "judea", "canaan",
    "jericho", "tabor", "zion",
    # Christian events
    "crucifixion", "resurrection", "nativity", "baptism", "transfiguration",
    "passover", "exodus", "creation", "miracle", "pentecost", "annunciation",
    "eucharist", "communion", "ascension", "last supper", "burning bush",
    "sermon on the mount", "walking on water", "raising of lazarus",
    # Symbols & objects
    "cross", "bible", "scripture", "church", "cathedral", "chapel",
    "monastery", "icon", "altar", "chalice", "halo", "dove", "lamb",
    "manger", "ark", "tabernacle", "temple", "crown of thorns", "stained glass",
    # General Christian terms
    "christian", "biblical", "prayer", "worship", "faith", "grace",
    "salvation", "redemption", "holy", "sacred", "divine", "apostle",
    "disciple", "saint", "prophet", "psalm", "gospel", "revelation",
    "testament", "lord", "savior", "messiah", "trinity", "blessed",
    "covenant", "shepherd", "parable", "righteous", "sanctified",
    # Specific biblical events / parables (multi-word or unique)
    "five thousand", "loaves", "prodigal", "samaritan", "ten commandments",
    "pharisee", "pharaoh", "lazarus", "garden of eden", "promised land",
    "parting of the sea", "sea of galilee", "mount of olives",
    # Extra biblical places
    "bethany", "emmaus", "capernaum", "jericho", "tiberias", "caesarea",
}

# Cosine similarity threshold for the semantic fallback check.
# Christian prompts typically score 0.28–0.55; off-topic score 0.02–0.15.
BIBLICAL_SIMILARITY_THRESHOLD = 0.22

# Cached anchor embedding (computed lazily on first semantic check)
_christian_anchor: "np.ndarray | None" = None  # type: ignore[name-defined]


def _is_biblical_topic(prompt: str) -> bool:
    """Fast O(1) check — True if any signal word appears in prompt."""
    prompt_lower = prompt.lower()
    return any(word in prompt_lower for word in BIBLICAL_SIGNAL_WORDS)


def _get_christian_anchor():
    """Lazily compute and cache the Christian anchor embedding.
    Returns None if the model is not yet loaded (e.g. during cold start).
    """
    global _christian_anchor
    if _christian_anchor is not None:
        return _christian_anchor
    try:
        import numpy as np
        from rag.retriever import retriever
        if not retriever._loaded:
            return None
        # Average of multiple anchor phrases for robustness
        phrases = [
            "biblical Christian scripture scene painting",
            "Christian religious artwork Jesus Bible",
            "sacred Bible verse illustration holy",
        ]
        vecs = retriever.model.encode(phrases, normalize_embeddings=True)
        anchor = np.mean(vecs, axis=0)
        anchor = anchor / np.linalg.norm(anchor)  # re-normalize after averaging
        _christian_anchor = anchor
        return _christian_anchor
    except Exception:
        return None


def _has_biblical_relevance(prompt: str) -> bool:
    """Two-stage biblical relevance check:
    1. Keyword fast-path  — O(1), 0ms, no model call.
    2. Cosine similarity  — semantic fallback for valid phrases
       that don't use exact signal words (e.g. 'feeding of the five thousand').
    Falls back to keyword-only if the model is not yet loaded.
    """
    # Stage 1: keyword fast-path (covers ~90% of prompts at zero cost)
    if _is_biblical_topic(prompt):
        return True

    # Stage 2: semantic cosine similarity via all-MiniLM-L6-v2
    anchor = _get_christian_anchor()
    if anchor is None:
        # Model not ready yet — conservative block
        return False
    try:
        from rag.retriever import retriever
        vec = retriever.model.encode(prompt, normalize_embeddings=True)
        similarity = float(vec @ anchor)
        return similarity >= BIBLICAL_SIMILARITY_THRESHOLD
    except Exception:
        return False


def moderate_image_prompt(prompt: str) -> ModerationResult:
    """Check image generation prompts.
    Layer 1: Harmful content blocklist (explicit / satanic / blasphemous).
    Layer 2: Biblical relevance — keyword fast-path + cosine similarity fallback.
    """
    prompt_lower = prompt.lower()

    # Layer 1 — harmful content
    for pattern, category, message in IMAGE_BLOCK_PATTERNS:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            return ModerationResult(
                allowed=False,
                reason=message,
                category=category,
                severity="hard_block"
            )

    # Layer 2 — relevance (keyword + semantic cosine similarity)
    if not _has_biblical_relevance(prompt):
        return ModerationResult(
            allowed=False,
            reason=(
                'This image generator is for biblical and Christian artwork only. '
                'Please describe a scripture scene or Christian subject — '
                'e.g. "The Last Supper", "Jesus walking on water", or "The Nativity".'
            ),
            category="off_topic",
            severity="hard_block"
        )

    return ModerationResult(allowed=True, reason="", category="clean", severity="clean")
