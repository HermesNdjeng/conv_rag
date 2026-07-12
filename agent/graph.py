from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph.state import CompiledStateGraph

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import AgentConfig, AgentState
from agent.tools import make_search_knowledge
from rag.retriever import DocumentRetriever


def build_redis_checkpointer(redis_url: str, *, ttl_minutes: int | None = None) -> RedisSaver:
    """Working memory: persist the conversation state per thread in Redis.

    ttl_minutes expires idle threads (refreshed on each read, so active conversations
    stay alive) — keeps working memory from accumulating in Redis forever.
    """
    ttl = {"default_ttl": ttl_minutes, "refresh_on_read": True} if ttl_minutes else None
    checkpointer = RedisSaver(redis_url=redis_url, ttl=ttl)
    checkpointer.setup()
    return checkpointer


def build_agent(
    retriever: DocumentRetriever,
    config: AgentConfig | None = None,
    *,
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the ReAct agent: an LLM bound to search_knowledge, looping until it answers.

    The provider is chosen by config (model_name/model_provider) via init_chat_model, or an
    LLM can be injected directly — so nothing here is tied to a specific provider.
    """
    config = config or AgentConfig()
    llm = llm or init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
        temperature=config.temperature,
    )
    return create_agent(
        llm,
        tools=[make_search_knowledge(retriever)],
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,  # type: ignore
        checkpointer=checkpointer,
    )


def run_agent(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    *,
    user_id: str,
    message: str,
    thread_id: str,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Run one turn. `thread_id` selects the conversation (working memory)."""
    config = config or AgentConfig()
    state = AgentState(user_id=user_id, messages=[HumanMessage(content=message)])
    return agent.invoke(
        state,
        {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * config.max_iterations + 1,
        },
    )
