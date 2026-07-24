from collections.abc import Iterator
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import StreamMode

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import AgentConfig, AgentState
from agent.tools import make_search_knowledge, recall_memory
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
    store: BaseStore | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the ReAct agent: an LLM bound to the tools, looping until it answers.

    The provider is chosen by config (model_name/model_provider) via init_chat_model, or an
    LLM can be injected directly — so nothing here is tied to a specific provider. recall_memory
    is only offered when a store is provided (episodic memory).
    """
    config = config or AgentConfig()
    llm = llm or init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
        temperature=config.temperature,
    )
    tools: list[BaseTool] = [make_search_knowledge(retriever)]
    if store is not None:
        tools.append(recall_memory)
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,  # type: ignore
        checkpointer=checkpointer,
        store=store,
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


def stream_agent(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    *,
    user_id: str,
    message: str,
    thread_id: str,
    config: AgentConfig | None = None,
    stream_mode: StreamMode | list[StreamMode] = "updates",
) -> Iterator[Any]:
    """Stream one turn's events — reusable by the CLI (tool visibility) and the app frontend
    (progressive response and tool indicators).

    Args:
        agent: The compiled agent graph.
        user_id: Identifies the user, for memory scoping.
        message: The user's message for this turn.
        thread_id: Selects the conversation (working memory).
        config: Agent configuration; defaults to AgentConfig().
        stream_mode: LangGraph stream mode — "updates" for per-node events (tool calls),
            "messages" for LLM token streaming, or a list of both.

    Yields:
        Whatever the chosen stream_mode emits (per-node update dicts for "updates";
        (message_chunk, metadata) tuples for "messages").
    """
    config = config or AgentConfig()
    state = AgentState(user_id=user_id, messages=[HumanMessage(content=message)])
    return agent.stream(
        state,
        {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * config.max_iterations + 1,
        },
        stream_mode=stream_mode,
    )
