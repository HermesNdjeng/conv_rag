"""FastAPI app exposing the conversational RAG agent.

Run:  poetry run uvicorn app.main:app --reload
"""

import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk

from agent.service import AgentService, build_agent_service
from app.schemas import ChatRequest, ChatResponse, SessionUpdate
from rag.utils.logging_utils import setup_logger


logger = setup_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Assemble the agent once at startup (heavy: embeddings + Redis) and share it via app.state."""
    app.state.agent_service = build_agent_service()
    yield


app = FastAPI(title="Conversational RAG — Cameroon history", lifespan=lifespan)

_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log any unhandled error and return a stable JSON 500 (never leak the stack to the client)."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def get_agent_service(request: Request) -> AgentService:
    """Dependency: return the shared AgentService assembled at startup."""
    return request.app.state.agent_service


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# TODO(ux): also stream tool events via stream_mode=["updates", "messages"] so the front can
# show "searching…" during retrieval instead of a blank before the answer starts.
@app.post("/users/{user_id}/sessions/{thread_id}/messages", response_model=None)
def post_message(
    user_id: str,
    thread_id: str,
    payload: ChatRequest,
    request: Request,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse | StreamingResponse:
    """Create a message in the session and return the agent's answer.

    Representation is negotiated via Accept: ``text/event-stream`` streams the answer token by
    token (SSE); otherwise the full answer is returned as JSON.
    """
    if "text/event-stream" in request.headers.get("accept", ""):

        def event_stream() -> Iterator[str]:
            # The exception handler can't help here: the 200 and headers are already sent, so a
            # failure mid-stream must surface as an SSE error event, not a dead connection.
            try:
                for chunk, _ in service.stream(
                    user_id=user_id,
                    message=payload.message,
                    thread_id=thread_id,
                    stream_mode="messages",
                ):
                    # Only stream the LLM's answer tokens; skip tool outputs (retrieved chunks),
                    # which also flow through "messages" mode as ToolMessages.
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    token = str(chunk.content)
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception:
                logger.exception("Streaming failed for thread %s", thread_id)
                yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = service.run(user_id=user_id, message=payload.message, thread_id=thread_id)
    answer = str(result["messages"][-1].content)
    return ChatResponse(response=answer)


# TODO(auth): user_id must come from the authenticated identity (token), not the path — and the
# caller must be authorized against it (token.user_id == path user_id) to prevent IDOR.
@app.patch("/users/{user_id}/sessions/{thread_id}")
def update_session(
    user_id: str,
    thread_id: str,
    payload: SessionUpdate,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> SessionUpdate:
    """Update a session. status='ended' consolidates the whole conversation into episodic memory."""
    service.end_session(user_id=user_id, thread_id=thread_id)
    return payload
