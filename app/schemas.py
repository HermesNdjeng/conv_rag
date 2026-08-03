from pydantic import BaseModel


class ChatRequest(BaseModel):
    """One user turn. thread_id selects the conversation (working memory)."""

    user_id: str
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    """The agent's final answer for the turn."""

    response: str
