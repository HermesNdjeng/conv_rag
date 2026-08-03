import os
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import StreamMode

from agent.graph import build_agent, build_redis_checkpointer, run_agent, stream_agent
from agent.memory import build_redis_store, consolidate_session
from agent.schemas import AgentConfig
from rag.retriever import DocumentRetriever


class AgentService:
    """The assembled agent plus its store and LLM, with one-call turn/stream/end methods.

    Built once (heavy: embeddings, Redis connections) and reused across requests. Wraps the
    lower-level run_agent/stream_agent/consolidate_session so callers hold a single object.
    """

    def __init__(
        self,
        agent: CompiledStateGraph[Any, Any, Any, Any],
        store: BaseStore,
        llm: BaseChatModel,
        config: AgentConfig,
    ) -> None:
        self._agent = agent
        self._store = store
        self._llm = llm
        self._config = config

    def run(self, *, user_id: str, message: str, thread_id: str) -> dict[str, Any]:
        """Run one turn and return the final agent state (working memory scoped to thread_id)."""
        return run_agent(
            self._agent, user_id=user_id, message=message, thread_id=thread_id, config=self._config
        )

    def stream(
        self,
        *,
        user_id: str,
        message: str,
        thread_id: str,
        stream_mode: StreamMode | list[StreamMode] = "updates",
    ) -> Iterator[Any]:
        """Stream one turn's events (per-node updates or token stream); see stream_agent."""
        return stream_agent(
            self._agent,
            user_id=user_id,
            message=message,
            thread_id=thread_id,
            config=self._config,
            stream_mode=stream_mode,
        )

    def end_session(self, *, user_id: str, thread_id: str) -> None:
        """Consolidate the whole conversation into episodic memory (call when the session ends)."""
        consolidate_session(
            self._agent, self._store, self._llm, user_id=user_id, thread_id=thread_id
        )


def build_agent_service(config: AgentConfig | None = None) -> AgentService:
    """Assemble the full agent stack from environment configuration.

    Loads .env, then builds embeddings, retriever, Redis checkpointer (working memory), Redis
    store (episodic memory), the LLM (provider-agnostic via config), and the agent graph.

    Args:
        config: Agent configuration; defaults to AgentConfig() (model, temperature, etc.).

    Returns:
        A ready-to-use AgentService. Requires REDIS_URL and EMBEDDING_MODEL in the environment,
        plus the provider's API key (e.g. GOOGLE_API_KEY).
    """
    load_dotenv()
    config = config or AgentConfig()
    redis_url = os.environ["REDIS_URL"]

    embeddings = HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])
    retriever = DocumentRetriever(embeddings=embeddings)
    checkpointer = build_redis_checkpointer(redis_url)
    store = build_redis_store(redis_url, embeddings)
    llm = init_chat_model(
        config.model_name, model_provider=config.model_provider, temperature=config.temperature
    )
    agent = build_agent(retriever, config, llm=llm, checkpointer=checkpointer, store=store)
    return AgentService(agent, store, llm, config)
