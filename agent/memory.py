from typing import Any

import redis as redis_lib
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.redis import RedisStore

from agent.prompts import CONSOLIDATION_PROMPT


def build_redis_store(redis_url: str, embeddings: Embeddings) -> RedisStore:
    """Episodic memory: store past-session summaries in Redis, retrievable by semantic search.

    Episodes are embedded on their `text` field so a new session can recall relevant ones.
    """
    conn = redis_lib.from_url(redis_url)
    dims = len(embeddings.embed_query("dimension probe"))
    store = RedisStore(conn, index={"dims": dims, "embed": embeddings, "fields": ["text"]})
    store.setup()
    return store


def _format_transcript(messages: list[AnyMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


# TODO(app): trigger this on session end — client disconnect or inactivity timeout. Sessions
# never closed cleanly won't be consolidated; add a sweeper that finds idle threads (no activity
# past a threshold) and consolidates them so no episode is lost.
def consolidate_session(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    store: BaseStore,
    llm: BaseChatModel,
    *,
    user_id: str,
    thread_id: str,
) -> None:
    """At the end of a session, summarize the whole thread into one episodic memory.

    Reads the full conversation from working memory (the checkpointer) and writes a single
    episode — one summary per session, not per turn.
    """
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    transcript = _format_transcript(snapshot.values.get("messages", []))
    if not transcript:
        return
    response = llm.invoke(
        [SystemMessage(content=CONSOLIDATION_PROMPT), HumanMessage(content=transcript)]
    )
    store.put(("episodes", user_id), thread_id, {"text": str(response.content)})
