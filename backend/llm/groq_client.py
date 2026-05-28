"""
groq_client.py — Groq LLM wrapper with streaming support and retry logic.
"""

import time
from groq import Groq, APIError, RateLimitError
from config import GROQ_API_KEY, GROQ_MODEL
from llm.prompts import build_system_prompt, build_user_message, DenominationHint
from rag.retriever import retriever, REFERENCE_RE
from typing import Generator

client = Groq(api_key=GROQ_API_KEY)


def _find_nonexistent_refs(query: str) -> list[str]:
    """
    Extract any Bible references from the query and check which ones
    do NOT exist in the KJV corpus. Returns list of fake/nonexistent refs.
    E.g. "What does Ezekiel 48:99 say?" → ["Ezekiel 48:99"]
    """
    if not retriever._loaded:
        retriever.load()
    matches = REFERENCE_RE.findall(query)
    nonexistent = []
    for book, chapter, verse in matches:
        ref = f"{book} {chapter}:{verse}".lower()
        if ref not in retriever._ref_index:
            nonexistent.append(f"{book} {chapter}:{verse}")
    return nonexistent


def chat_stream(
    user_message: str,
    history: list[dict],
    denomination: DenominationHint = "nondenominational",
    is_soft_block: bool = False,
    top_k: int = 5,
    max_retries: int = 3
) -> Generator[str, None, dict]:
    """
    Full RAG + streaming pipeline:
    1. Retrieve relevant passages
    2. Build grounded messages
    3. Stream Groq response token by token
    Yields: text chunks (str)
    Returns (via StopIteration value): {passages, usage}
    """
    # Step 1: RAG retrieval
    passages = retriever.retrieve(user_message, top_k=top_k)

    # Step 1b: Detect nonexistent refs in query (hallucination guard)
    nonexistent_refs = _find_nonexistent_refs(user_message)

    # Step 2: Build messages
    system_prompt = build_system_prompt(denomination, is_soft_block, nonexistent_refs)
    messages = build_user_message(user_message, passages, history)

    # Step 3: Stream with retry
    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                temperature=0.2,       # Low = strict grounding, minimal hallucination
                max_tokens=1024,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

            return {"passages": passages}

        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except APIError as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise


def chat_sync(
    user_message: str,
    history: list[dict],
    denomination: DenominationHint = "nondenominational",
    is_soft_block: bool = False,
    top_k: int = 5,
) -> dict:
    """
    Non-streaming version — used by evaluation pipeline.
    Returns: {response: str, passages: list}
    """
    passages = retriever.retrieve(user_message, top_k=top_k)
    nonexistent_refs = _find_nonexistent_refs(user_message)
    system_prompt = build_system_prompt(denomination, is_soft_block, nonexistent_refs)
    messages = build_user_message(user_message, passages, history)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=False,
    )

    return {
        "response": completion.choices[0].message.content,
        "passages": passages
    }
