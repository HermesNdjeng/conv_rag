from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """One user turn. user_id and thread_id come from the path; the body carries the message."""

    message: str


class ChatResponse(BaseModel):
    """The agent's final answer for the turn."""

    response: str


class SessionUpdate(BaseModel):
    """Partial update of a session. status='ended' triggers episodic consolidation."""

    status: Literal["ended"]
