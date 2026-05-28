"""
session_store.py — In-memory per-session conversation history.
Uses a rolling deque capped at MAX_HISTORY_TURNS turns.
Session IDs are client-generated UUIDs passed in headers.
"""

from collections import deque
from config import MAX_HISTORY_TURNS

# Global store: {session_id: deque of {role, content}}
_sessions: dict[str, deque] = {}


def get_history(session_id: str) -> list[dict]:
    """Return conversation history for a session as a list."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_HISTORY_TURNS * 2)  # *2 for user+assistant pairs
    return list(_sessions[session_id])


def add_turn(session_id: str, role: str, content: str):
    """Append a message turn to the session history."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_HISTORY_TURNS * 2)
    _sessions[session_id].append({"role": role, "content": content})


def clear_session(session_id: str):
    """Clear a session's history."""
    _sessions.pop(session_id, None)


def session_count() -> int:
    return len(_sessions)
