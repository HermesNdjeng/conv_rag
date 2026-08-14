"""FastAPI app exposing the conversational RAG agent.

Run:  poetry run uvicorn app.main:app --reload
"""

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from agent.service import AgentService, build_agent_service
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Assemble the agent once at startup (heavy: embeddings + Redis) and share it via app.state."""
    app.state.agent_service = build_agent_service()
    yield


app = FastAPI(title="Conversational RAG — Cameroon history", lifespan=lifespan)


def get_agent_service(request: Request) -> AgentService:
    """Dependency: return the shared AgentService assembled at startup."""
    return request.app.state.agent_service


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/chat")
def chat(
    payload: ChatRequest, service: Annotated[AgentService, Depends(get_agent_service)]
) -> ChatResponse:
    """Run one conversation turn and return the agent's final answer."""
    result = service.run(
        user_id=payload.user_id, message=payload.message, thread_id=payload.thread_id
    )
    answer = str(result["messages"][-1].content)
    return ChatResponse(response=answer)


# TODO(ux): also stream tool events via stream_mode=["updates", "messages"] so the front can
# show "searching…" during retrieval instead of a blank before the answer starts.
@app.post("/chat/stream")
def chat_stream(
    payload: ChatRequest, service: Annotated[AgentService, Depends(get_agent_service)]
) -> StreamingResponse:
    """Stream the agent's answer token by token as Server-Sent Events."""

    def event_stream() -> Iterator[str]:
        for chunk, _ in service.stream(
            user_id=payload.user_id,
            message=payload.message,
            thread_id=payload.thread_id,
            stream_mode="messages",
        ):
            token = str(chunk.content)
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/sessions/{thread_id}/end")
def end_session(
    thread_id: str,
    user_id: str,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> dict[str, str]:
    """End a session: consolidate the whole conversation into episodic memory."""
    service.end_session(user_id=user_id, thread_id=thread_id)
    return {"status": "session consolidated"}
