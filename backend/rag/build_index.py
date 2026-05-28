"""
build_index.py — Run ONCE locally.
Downloads KJV Bible JSON, embeds all verses with sentence-transformers,
saves embeddings.npy + metadata.json to /data/.
These files are committed to Git and loaded at runtime — no re-embedding on server.
"""

import os
import json
import numpy as np
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Public domain KJV Bible JSON
KJV_BOOKS_URL = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master/Books.json"
KJV_BOOK_BASE = "https://raw.githubusercontent.com/aruljohn/Bible-kjv/master"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Step 1: Download Bible
# ---------------------------------------------------------------------------
def download_bible() -> list[dict]:
    """
    Download KJV Bible per-book JSON files and flatten into verse records.
    Structure: Books.json = ["Genesis", "Exodus", ...]
    Each book file: {book, chapters: [{chapter, verses: [{verse, text}]}]}
    """
    print("[INFO] Fetching book list...")
    response = requests.get(KJV_BOOKS_URL, timeout=30)
    response.raise_for_status()
    book_names = response.json()  # plain list of strings

    verses = []
    for book_name in tqdm(book_names, desc="Fetching books"):
        # File names have no spaces: "1 Samuel" -> "1Samuel.json"
        file_name = book_name.replace(" ", "")
        book_url = f"{KJV_BOOK_BASE}/{file_name}.json"
        book_resp = requests.get(book_url, timeout=30)
        book_resp.raise_for_status()
        book_data = book_resp.json()

        for chapter_obj in book_data["chapters"]:
            chap_num = int(chapter_obj["chapter"])
            for verse_obj in chapter_obj["verses"]:
                verse_num = int(verse_obj["verse"])
                text = verse_obj["text"].strip()
                verses.append({
                    "book": book_name,
                    "chapter": chap_num,
                    "verse": verse_num,
                    "text": text,
                    "reference": f"{book_name} {chap_num}:{verse_num}"
                })

    print(f"[DONE] Loaded {len(verses):,} verses from {len(book_names)} books")
    return verses


# ---------------------------------------------------------------------------
# Step 2: Embed
# ---------------------------------------------------------------------------
def embed_verses(verses: list[dict]) -> np.ndarray:
    """Embed all verses in batches. Returns float32 ndarray (N, 384)."""
    print(f"\n[INFO] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [v["text"] for v in verses]
    print(f"[INFO] Embedding {len(texts):,} verses in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine via dot product
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3: Save
# ---------------------------------------------------------------------------
def save(verses: list[dict], embeddings: np.ndarray):
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"[DONE] Saved embeddings -> {EMBEDDINGS_PATH}  ({embeddings.nbytes / 1e6:.1f} MB)")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[DONE] Saved metadata  -> {METADATA_PATH}")

    print(f"\n[SUCCESS] Done! Shape: {embeddings.shape} | dtype: {embeddings.dtype}")
    print("   Commit data/embeddings.npy and data/metadata.json to Git.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verses = download_bible()
    embeddings = embed_verses(verses)
    save(verses, embeddings)
