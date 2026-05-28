"""
main.py — FastAPI application entry point.
Routes: /chat (streaming SSE), /image, /health, /clear-session
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional

from config import FRONTEND_URL
from rag.retriever import retriever
from llm.groq_client import chat_stream
from safety.moderator import moderate_text
from safety.validator import validate_response
from memory.session_store import get_history, add_turn, clear_session
from image.generator import generate_christian_image


# ---------------------------------------------------------------------------
# Lifespan — load index once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading scripture index...")
    retriever.load()
    print("[STARTUP] Ready to serve")
    yield
    print("[SHUTDOWN] Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Christian AI Assistant API",
    description="A grounded, scripture-aware Christian AI assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    denomination: Literal["protestant", "catholic", "orthodox", "nondenominational"] = "nondenominational"
    session_id: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)

class ClearRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": retriever._loaded,
        "verse_count": len(retriever.metadata) if retriever._loaded else 0
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streaming chat endpoint — returns SSE stream.
    Flow: moderate → retrieve passages → stream LLM → validate → emit final metadata
    """
    session_id = req.session_id or str(uuid.uuid4())

    # 1. Moderation
    mod = moderate_text(req.message)
    if not mod.allowed:
        return JSONResponse(
            status_code=200,
            content={
                "blocked": True,
                "reason": mod.reason,
                "category": mod.category,
                "session_id": session_id
            }
        )

    is_soft = mod.severity == "soft_block"

    # 2. Get history
    history = get_history(session_id)

    # 3. Add user turn to history
    add_turn(session_id, "user", req.message)

    # 4. Stream response
    async def event_stream():
        full_response = []
        passages = []

        try:
            gen = chat_stream(
                user_message=req.message,
                history=history,
                denomination=req.denomination,
                is_soft_block=is_soft,
            )

            for chunk in gen:
                if isinstance(chunk, dict):
                    # Final metadata yielded by generator
                    passages = chunk.get("passages", [])
                else:
                    full_response.append(chunk)
                    data = json.dumps({"type": "token", "content": chunk})
                    yield f"data: {data}\n\n"

        except Exception as e:
            error_data = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_data}\n\n"
            return

        # 5. Validate for hallucinations
        response_text = "".join(full_response)
        validation = validate_response(response_text)

        # 6. Save assistant turn
        add_turn(session_id, "assistant", response_text)

        # 7. Send final metadata event
        meta = {
            "type": "done",
            "session_id": session_id,
            "passages": [
                {
                    "reference": p["reference"],
                    "text": p["text"],
                    "book": p["book"],
                    "score": round(p["score"], 3)
                }
                for p in passages
            ],
            "validation": {
                "is_clean": validation["is_clean"],
                "warning": validation["warning_message"],
                "flagged": validation["flagged_references"],
                "verified": validation["verified_references"]
            },
            "denomination": req.denomination
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id
        }
    )


@app.post("/image")
async def generate_image(req: ImageRequest):
    """Generate a Christian-themed image via pollinations.ai.
    Blocks until the image is confirmed ready on pollinations.ai (up to 120s).
    """
    result = generate_christian_image(req.prompt)
    return JSONResponse(
        status_code=200,
        content={
            "success": result["success"],
            "url": result.get("url"),
            "blocked": result.get("blocked", False),
            "reason": result.get("reason"),
            "enhanced_prompt": result.get("enhanced_prompt"),
        }
    )


@app.post("/clear-session")
async def clear(req: ClearRequest):
    """Clear conversation history for a session."""
    clear_session(req.session_id)
    return {"cleared": True, "session_id": req.session_id}


@app.get("/")
async def root():
    return {"message": "Christian AI Assistant API — see /docs for endpoints"}
