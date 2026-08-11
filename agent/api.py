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
import threading
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
    version="4.0.0",
)

# Global agent session (one per process)
_agent = AgentSession()
_agent_lock = threading.Lock()
_cancel_event = threading.Event()

# In-memory chat history per session (resets on restart)
_chat_history: list[dict[str, str]] = []


# ---- Request / Response models ----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    stream: bool = Field(default=False, description="Stream response tokens as SSE")


class ChatResponse(BaseModel):
    answer: str
    model: str


class MemoryStats(BaseModel):
    short_term_entries: int
    memories: int
    summaries: int


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — returns agent status and model info."""
    config = settings.get_llm_config()
    return HealthResponse(
        status="ok",
        model=config["model"],
        provider=config["provider"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the agent and get a response (non-streaming)."""
    global _chat_history

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    _cancel_event.clear()

    events: list[dict[str, Any]] = []

    def _collect(event: dict[str, Any]) -> None:
        events.append(event)

    with _agent_lock:
        try:
            answer = _agent.invoke(
                req.message, _chat_history,
                progress_callback=_collect,
            )
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    _chat_history.append({"role": "user", "content": req.message})
    _chat_history.append({"role": "assistant", "content": answer})

    # Trim history to prevent unbounded growth
    if len(_chat_history) > 50:
        _chat_history = _chat_history[-50:]

    return ChatResponse(answer=answer, model=settings.get_llm_config()["model"])


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Send a message and stream the agent's response as Server-Sent Events."""
    global _chat_history

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    _cancel_event.clear()

    event_queue: list[dict[str, Any]] = []

    def _collect(event: dict[str, Any]) -> None:
        event_queue.append(event)

    def _run_agent() -> None:
        with _agent_lock:
            try:
                answer = _agent.invoke(
                    req.message, _chat_history,
                    progress_callback=_collect,
                )
                event_queue.append({"type": "done", "answer": answer})
            except Exception as exc:
                event_queue.append({"type": "error", "message": str(exc)})

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

        # Update chat history after completion
        final_answer = ""
        for ev in event_queue:
            if ev.get("type") == "done":
                final_answer = ev.get("answer", "")
        if final_answer:
            _chat_history.append({"role": "user", "content": req.message})
            _chat_history.append({"role": "assistant", "content": final_answer})
            if len(_chat_history) > 50:
                _chat_history = _chat_history[-50:]

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/cancel")
def cancel_chat() -> dict[str, str]:
    """Cancel the currently running agent invocation."""
    _cancel_event.set()
    _agent.stop()
    return {"status": "cancelled"}


@app.get("/memory/stats", response_model=MemoryStats)
def memory_stats() -> MemoryStats:
    """Get current memory statistics."""
    stats = _agent.memory.memory_stats
    return MemoryStats(
        short_term_entries=stats.get("short_term_entries", 0),
        memories=stats.get("memories", 0),
        summaries=stats.get("summaries", 0),
    )


@app.delete("/memory/long-term")
def clear_long_term_memory() -> dict[str, str]:
    """Clear all long-term memory."""
    with _agent_lock:
        result = _agent.clear_long_term_memory()
    return {"status": "cleared", "detail": result}


@app.delete("/memory/short-term")
def clear_short_term_memory() -> dict[str, str]:
    """Clear short-term memory and chat history."""
    global _chat_history
    _agent.memory.clear_short_term()
    _chat_history = []
    return {"status": "cleared"}


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
