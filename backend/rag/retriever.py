"""
retriever.py — Scripture RAG retriever.
Loads pre-built numpy index once at startup, serves cosine similarity
search in ~10ms for 31k verses. No DB, no cold-start re-embedding.

Hybrid retrieval strategy:
  1. Exact reference lookup  — scans query for "Book Chapter:Verse" patterns
     and fetches those verses directly (score = 1.0, guaranteed correct)
  2. Semantic search         — cosine similarity on the query embedding
  Results are merged: exact matches pinned to top, semantic fills the rest.
"""

import re
import json
import numpy as np
from fastembed import TextEmbedding
from config import EMBEDDINGS_PATH, METADATA_PATH, EMBEDDING_MODEL, TOP_K_PASSAGES

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

# Explicit book name list prevents prepositions like "In" / "As" being
# captured as part of the book name in exact reference extraction.
REFERENCE_RE = re.compile(
    r'\b(' + _BOOK_NAMES + r')\s+(\d{1,3}):(\d{1,3})\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Stage 0 — Keyword → canonical reference map
# Maps well-known theological concepts/titles to their KJV verse references.
# Fires before exact reference lookup and semantic search.
# Format: { "keyword_pattern": ["Book Ch:V", ...] }
# ---------------------------------------------------------------------------
KEYWORD_MAP: list[tuple[str, list[str]]] = [
    # Sermon on the Mount / Beatitudes
    (r"beatitude",                ["Matthew 5:3", "Matthew 5:4", "Matthew 5:5",
                                   "Matthew 5:6", "Matthew 5:7", "Matthew 5:8"]),
    (r"sermon\s+on\s+the\s+mount",["Matthew 5:1", "Matthew 5:3", "Matthew 5:9",
                                   "Matthew 5:17", "Matthew 6:9"]),
    (r"blessed\s+are\s+the\s+poor",["Matthew 5:3", "Luke 6:20"]),
    (r"salt\s+of\s+the\s+earth",  ["Matthew 5:13"]),
    (r"light\s+of\s+the\s+world", ["Matthew 5:14"]),
    (r"turn\s+the\s+other\s+cheek",["Matthew 5:39"]),
    (r"love\s+your\s+enemies",    ["Matthew 5:44", "Luke 6:27"]),

    # Lord's Prayer
    (r"lord.?s\s+prayer|our\s+father", ["Matthew 6:9", "Matthew 6:10",
                                        "Matthew 6:11", "Matthew 6:12",
                                        "Matthew 6:13"]),

    # Ten Commandments
    (r"ten\s+commandments?",      ["Exodus 20:3", "Exodus 20:4", "Exodus 20:7",
                                   "Exodus 20:8", "Exodus 20:12", "Exodus 20:13"]),
    (r"thou\s+shalt\s+not\s+kill",["Exodus 20:13", "Deuteronomy 5:17"]),
    (r"honour\s+thy\s+father",    ["Exodus 20:12", "Ephesians 6:2"]),

    # Parables
    (r"parable\s+of\s+the\s+prodigal",  ["Luke 15:11", "Luke 15:20", "Luke 15:24"]),
    (r"prodigal\s+son",                  ["Luke 15:11", "Luke 15:20", "Luke 15:24"]),
    (r"parable\s+of\s+the\s+sower",     ["Matthew 13:3", "Mark 4:3", "Luke 8:5"]),
    (r"parable\s+of\s+the\s+mustard",   ["Matthew 13:31", "Mark 4:31"]),
    (r"good\s+samaritan",               ["Luke 10:30", "Luke 10:33", "Luke 10:37"]),
    (r"parable\s+of\s+the\s+talents",   ["Matthew 25:14", "Matthew 25:29"]),
    (r"lost\s+sheep",                   ["Luke 15:4", "Matthew 18:12"]),

    # Jesus — family / siblings / brothers
    (r"jesus.{0,20}(brother|sister|sibling|family|mother)",
                                        ["Matthew 13:55", "Mark 6:3",
                                         "John 2:12", "Matthew 12:46"]),
    (r"(brother|sister).{0,20}jesus",   ["Matthew 13:55", "Mark 6:3"]),
    (r"james.*brother.*lord",           ["Matthew 13:55"]),

    # Resurrection
    (r"resurrect|risen\s+from\s+the\s+dead|empty\s+tomb",
                                        ["Matthew 28:6", "Mark 16:6",
                                         "Luke 24:6", "John 20:1"]),
    (r"he\s+is\s+risen",               ["Matthew 28:6", "Mark 16:6"]),

    # Nativity / Birth of Jesus
    (r"birth\s+of\s+jesus|nativity|born\s+in\s+bethlehem",
                                        ["Luke 2:7", "Luke 2:11", "Matthew 2:1"]),
    (r"wise\s+men|magi",               ["Matthew 2:1", "Matthew 2:11"]),
    (r"virgin\s+birth|immaculate",     ["Luke 1:27", "Luke 1:31", "Matthew 1:23"]),

    # Creation
    (r"in\s+the\s+beginning|creation\s+of\s+(the\s+)?world",
                                        ["Genesis 1:1", "Genesis 1:2",
                                         "Genesis 1:3", "John 1:1"]),

    # Great Commission
    (r"great\s+commission|go\s+and\s+make\s+disciples",
                                        ["Matthew 28:19", "Matthew 28:20",
                                         "Mark 16:15"]),

    # Golden Rule
    (r"golden\s+rule|do\s+unto\s+others",["Matthew 7:12", "Luke 6:31"]),

    # Faith / Salvation
    (r"faith\s+(can\s+)?move\s+mountains",["Matthew 17:20", "1 Corinthians 13:2"]),
    (r"saved\s+by\s+grace|salvation\s+through\s+faith",
                                        ["Ephesians 2:8", "Ephesians 2:9",
                                         "Romans 10:9"]),
    (r"born\s+again",                  ["John 3:3", "John 3:7", "1 Peter 1:23"]),

    # Fruits of the Spirit
    (r"fruits?\s+of\s+the\s+spirit",   ["Galatians 5:22", "Galatians 5:23"]),

    # Armour of God
    (r"armou?r\s+of\s+god|whole\s+armou?r",["Ephesians 6:11", "Ephesians 6:13"]),

    # Psalms 23
    (r"shepherd\s+psalm|lord\s+is\s+my\s+shepherd|psalm\s+of\s+david",
                                        ["Psalms 23:1", "Psalms 23:4", "Psalms 23:6"]),

    # Love chapter
    (r"love\s+is\s+patient|love\s+chapter|1\s*cor\w*\s+13",
                                        ["1 Corinthians 13:4", "1 Corinthians 13:7",
                                         "1 Corinthians 13:13"]),
]

# ---------------------------------------------------------------------------
# Singleton — loaded once when FastAPI starts
# ---------------------------------------------------------------------------
class ScriptureRetriever:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("[INFO] Loading scripture index...")
        self.embeddings = np.load(EMBEDDINGS_PATH)          # (31102, 384) float32
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build a lowercase reference → index map for O(1) exact lookups
        self._ref_index: dict[str, int] = {
            v["reference"].lower(): i for i, v in enumerate(self.metadata)
        }

        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self._loaded = True
        print(f"[DONE] Scripture index ready -- {len(self.metadata):,} verses loaded")

    # ------------------------------------------------------------------
    # Exact reference lookup
    # ------------------------------------------------------------------
    def _exact_lookup(self, query: str) -> list[dict]:
        """
        Extract any Bible references from the query (e.g. 'John 3:16')
        and fetch them directly from metadata. Returns verses with score=1.0.
        """
        matches = REFERENCE_RE.findall(query)
        results = []
        seen = set()
        for book, chapter, verse in matches:
            ref = f"{book} {chapter}:{verse}".lower()
            idx = self._ref_index.get(ref)
            if idx is not None and idx not in seen:
                seen.add(idx)
                v = dict(self.metadata[idx])
                v["score"] = 1.0
                v["exact"] = True
                results.append(v)
        return results

    # ------------------------------------------------------------------
    # Stage 0 — Keyword lookup
    # ------------------------------------------------------------------
    def _keyword_lookup(self, query: str, exclude_indices: set) -> list[dict]:
        """
        Scan query against KEYWORD_MAP. Returns matched verses with score=0.99
        (just below exact reference matches). Skips already-found indices.
        """
        query_lower = query.lower()
        results = []
        seen_refs: set[str] = set()

        for pattern, references in KEYWORD_MAP:
            if re.search(pattern, query_lower, re.IGNORECASE):
                for ref in references:
                    if ref.lower() in seen_refs:
                        continue
                    idx = self._ref_index.get(ref.lower())
                    if idx is not None and idx not in exclude_indices:
                        seen_refs.add(ref.lower())
                        exclude_indices.add(idx)
                        v = dict(self.metadata[idx])
                        v["score"] = 0.99
                        v["exact"] = True
                        results.append(v)
        return results

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------
    def _semantic_search(self, query: str, top_k: int, exclude_indices: set) -> list[dict]:
        """
        Cosine similarity search. Skips indices already found by exact lookup.
        """
        query_vec = np.array(
            list(self.model.embed([query]))[0],
            dtype=np.float32
        )                                                    # (384,) already normalized

        scores = self.embeddings @ query_vec                 # (N,) dot product
        # Zero out already-included indices so they don't duplicate
        for idx in exclude_indices:
            scores[idx] = -1.0

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] < 0:
                continue
            v = dict(self.metadata[idx])
            v["score"] = float(scores[idx])
            v["exact"] = False
            results.append(v)
        return results

    # ------------------------------------------------------------------
    # Public retrieve — hybrid
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K_PASSAGES) -> list[dict]:
        """
        Three-stage hybrid retrieval:
          Stage 0 — Keyword map  (theological concepts → canonical refs, score=0.99)
          Stage 1 — Exact ref   (Book Ch:V in query → direct lookup, score=1.0)
          Stage 2 — Semantic    (cosine similarity fills remaining slots)
        Returns up to top_k results, highest-scoring first.
        """
        if not self._loaded:
            self.load()

        # Stage 1: exact reference lookup (highest priority)
        exact = self._exact_lookup(query)
        already_found = {self._ref_index[v["reference"].lower()] for v in exact}

        # Stage 0: keyword map (concept-based, e.g. "Beatitudes" → Matthew 5)
        keyword = self._keyword_lookup(query, already_found)  # already_found mutated in-place

        pinned = exact + keyword

        # Stage 2: semantic search fills remaining slots
        semantic_needed = max(0, top_k - len(pinned))
        semantic = self._semantic_search(query, semantic_needed, already_found) if semantic_needed > 0 else []

        return pinned + semantic

    # ------------------------------------------------------------------
    # Hallucination validator helper
    # ------------------------------------------------------------------
    def verse_exists(self, reference: str) -> bool:
        """
        O(1) lookup — check if a reference exists in corpus.
        Used by post-processing hallucination validator.
        """
        if not self._loaded:
            self.load()
        return reference.lower().strip() in self._ref_index


# Global singleton
retriever = ScriptureRetriever()
