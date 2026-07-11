from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model_name: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.0)
    max_iterations: int = Field(default=6)


class AgentState(BaseModel):
    """Working memory: the running message list plus the user we're serving."""

    user_id: str
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
