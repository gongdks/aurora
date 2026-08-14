"""FastAPI headless API — REST interface for the AI Agent.

Usage:
    python -m agent.api          # start server on port 8080
    uvicorn agent.api:app        # via uvicorn directly

Endpoints:
    POST /chat                   — send a message, get agent response
    GET  /health                 — health check
    GET  /memory/stats           — memory statistics
    DELETE /memory/long-term     — clear long-term memory
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.agent import AgentSession
from agent.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent API",
    description="Plan-and-Execute AI Agent with tool-calling and memory",
    version="5.0.0",
)

_MAX_MESSAGE_LENGTH = 10000
_MAX_ACTIVE_SESSIONS = 100
_SESSION_TTL_SECONDS = 3600
_MAX_HISTORY_PER_SESSION = 50

# ---- Session Manager ----

class SessionManager:
    """Per-user session isolation — each session has its own AgentSession and chat history."""

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None = None) -> tuple[str, dict[str, Any]]:
        with self._lock:
            now = time.time()

            if session_id and session_id in self._sessions:
                entry = self._sessions[session_id]
                entry["last_access"] = now
                self._sessions.move_to_end(session_id)
                return session_id, entry

            self._evict_expired_locked(now)

            sid = session_id or secrets.token_hex(8)
            agent = AgentSession()
            entry = {
                "agent": agent,
                "history": [],
                "created_at": now,
                "last_access": now,
            }
            self._sessions[sid] = entry
            logger.info("Session created: %s (total: %d)", sid, len(self._sessions))
            return sid, entry

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session removed: %s", session_id)

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            sid for sid, entry in self._sessions.items()
            if now - entry["last_access"] > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired session removed: %s", sid)

        while len(self._sessions) > _MAX_ACTIVE_SESSIONS:
            oldest_sid, _ = next(iter(self._sessions.items()))
            del self._sessions[oldest_sid]
            logger.info("Oldest session evicted: %s", oldest_sid)


_session_mgr = SessionManager()
_cancel_events: dict[str, tuple[threading.Event, float]] = {}
_cancel_event_ttl = 600  # 10 minutes


def _cleanup_old_cancel_events() -> None:
    """Remove stale cancel events that have lived beyond their TTL."""
    now = time.time()
    expired = [sid for sid, (_, ts) in _cancel_events.items() if now - ts > _cancel_event_ttl]
    for sid in expired:
        _cancel_events.pop(sid, None)


def _get_cancel_event(session_id: str) -> threading.Event | None:
    """Get or create a cancel event for a session."""
    _cleanup_old_cancel_events()
    entry = _cancel_events.get(session_id)
    if entry is not None:
        event, _ = entry
        return event
    return None


def _set_cancel_event(session_id: str) -> threading.Event:
    """Create and register a cancel event for a session."""
    _cleanup_old_cancel_events()
    event = threading.Event()
    _cancel_events[session_id] = (event, time.time())
    return event


def _clear_cancel_event(session_id: str) -> None:
    """Remove cancel event for a session."""
    _cancel_events.pop(session_id, None)


# ---- Request / Response models ----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_MESSAGE_LENGTH, description="User message")
    stream: bool = Field(default=False, description="Stream response tokens as SSE")
    session_id: str | None = Field(default=None, description="Session ID from previous response")


class ChatResponse(BaseModel):
    answer: str
    model: str
    session_id: str


class MemoryStats(BaseModel):
    short_term_entries: int
    memories: int
    summaries: int


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    active_sessions: int


class ModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(graph|autogen)$")


# ---- Input sanitization ----

def _sanitize_message(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) > _MAX_MESSAGE_LENGTH:
        cleaned = cleaned[:_MAX_MESSAGE_LENGTH]
    return cleaned


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — returns agent status and model info."""
    config = settings.get_llm_config()
    return HealthResponse(
        status="ok",
        model=config["model"],
        provider=config["provider"],
        active_sessions=len(_session_mgr._sessions),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the agent and get a response (non-streaming)."""
    message = _sanitize_message(req.message)
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty after sanitization")

    sid, entry = _session_mgr.get_or_create(req.session_id)
    agent: AgentSession = entry["agent"]
    history: list = entry["history"]

    cancel_event = _get_cancel_event(sid)
    if cancel_event:
        cancel_event.clear()

    events: list[dict[str, Any]] = []

    def _collect(event: dict[str, Any]) -> None:
        events.append(event)

    try:
        answer = agent.invoke(
            message, history,
            progress_callback=_collect,
        )
    except Exception as exc:
        logger.error("Agent error in session %s: %s", sid, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    if len(history) > _MAX_HISTORY_PER_SESSION:
        entry["history"] = history[-_MAX_HISTORY_PER_SESSION:]

    return ChatResponse(
        answer=answer,
        model=settings.get_llm_config()["model"],
        session_id=sid,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Send a message and stream the agent's response as Server-Sent Events."""
    message = _sanitize_message(req.message)
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty after sanitization")

    sid, entry = _session_mgr.get_or_create(req.session_id)
    agent: AgentSession = entry["agent"]
    history: list = entry["history"]

    cancel_event = _set_cancel_event(sid)

    event_queue: list[dict[str, Any]] = []

    def _collect(event: dict[str, Any]) -> None:
        event_queue.append(event)

    def _run_agent() -> None:
        try:
            answer = agent.invoke(
                message, history,
                progress_callback=_collect,
                cancel_event=cancel_event,
            )
            event_queue.append({"type": "done", "answer": answer, "session_id": sid})
        except Exception as exc:
            event_queue.append({"type": "error", "message": str(exc), "session_id": sid})

    thread = threading.Thread(target=_run_agent, daemon=True)
    thread.start()

    async def _event_generator():
        sent_idx = 0
        while thread.is_alive() or sent_idx < len(event_queue):
            while sent_idx < len(event_queue):
                ev = event_queue[sent_idx]
                sent_idx += 1
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            import asyncio
            await asyncio.sleep(0.05)

        final_answer = ""
        for ev in event_queue:
            if ev.get("type") == "done":
                final_answer = ev.get("answer", "")
                break
        if final_answer:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": final_answer})
            if len(history) > _MAX_HISTORY_PER_SESSION:
                entry["history"] = history[-_MAX_HISTORY_PER_SESSION:]

        _clear_cancel_event(sid)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/cancel")
def cancel_chat(session_id: str | None = None) -> dict[str, str]:
    """Cancel the currently running agent invocation for a session."""
    if session_id:
        evt = _get_cancel_event(session_id)
        if evt:
            evt.set()
            return {"status": "cancelled", "session_id": session_id}

    _cleanup_old_cancel_events()
    for sid, (evt, _) in list(_cancel_events.items()):
        evt.set()
        return {"status": "cancelled", "session_id": sid}

    return {"status": "no_active_session"}


@app.post("/agent/mode")
def set_agent_mode(req: ModeRequest, session_id: str | None = None) -> dict[str, str]:
    """Switch orchestrator mode at runtime.

    Body: {"mode": "graph" | "autogen", "session_id": "..."}
    """
    sid, entry = _session_mgr.get_or_create(session_id)
    entry["agent"].switch_mode(req.mode)
    return {"status": "ok", "mode": req.mode, "session_id": sid}


@app.get("/agent/mode")
def get_agent_mode(session_id: str | None = None) -> dict[str, str]:
    """Get current orchestrator mode."""
    sid, entry = _session_mgr.get_or_create(session_id)
    return {"mode": entry["agent"].orchestrator_mode, "session_id": sid}


@app.get("/memory/stats", response_model=MemoryStats)
def memory_stats(session_id: str | None = None) -> MemoryStats:
    """Get current memory statistics."""
    sid, entry = _session_mgr.get_or_create(session_id)
    stats = entry["agent"].memory.memory_stats
    return MemoryStats(
        short_term_entries=stats.get("short_term_entries", 0),
        memories=stats.get("memories", 0),
        summaries=stats.get("summaries", 0),
    )


@app.delete("/memory/long-term")
def clear_long_term_memory(session_id: str | None = None) -> dict[str, str]:
    """Clear all long-term memory."""
    sid, entry = _session_mgr.get_or_create(session_id)
    result = entry["agent"].clear_long_term_memory()
    return {"status": "cleared", "detail": result, "session_id": sid}


@app.delete("/memory/short-term")
def clear_short_term_memory(session_id: str | None = None) -> dict[str, str]:
    """Clear short-term memory and chat history."""
    sid, entry = _session_mgr.get_or_create(session_id)
    entry["agent"].memory.clear_short_term()
    entry["history"] = []
    return {"status": "cleared", "session_id": sid}


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Destroy a session and free its resources."""
    _session_mgr.remove(session_id)
    _clear_cancel_event(session_id)
    return {"status": "destroyed", "session_id": session_id}


# ---- Entry point ----

def main() -> None:
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "agent.api:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )


if __name__ == "__main__":
    main()