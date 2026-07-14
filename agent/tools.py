from typing import Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from rag.constants import GLOBAL_INDEX
from rag.indexer import RedisIndexer
from rag.retriever import DocumentRetriever


def make_search_knowledge(retriever: DocumentRetriever) -> BaseTool:
    """Build the search_knowledge tool bound to a retriever (kept out of the LLM's view)."""

    @tool
    def search_knowledge(query: str, user_id: Annotated[str, InjectedState("user_id")]) -> str:
        """Search the Cameroon history knowledge base for passages relevant to a query.

        Use this to ground factual claims. Rewrite the query to be self-contained, and
        for a question with several parts, search each part separately. Search again
        with a refined query if the results are insufficient.
        """
        indexes = [GLOBAL_INDEX, RedisIndexer.user_index(user_id)]
        docs = retriever.retrieve(query, indexes)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(
            f"[source: {doc.metadata.get('document_id', '?')}]\n{doc.page_content}" for doc in docs
        )

    return search_knowledge


@tool
def recall_memory(
    query: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Recall relevant memories from earlier conversations with this user.

    Use this when the current question may build on something discussed before, or to
    reuse context, facts, or conclusions established in a past exchange.
    """
    items = store.search(("episodes", user_id), query=query, limit=5)
    if not items:
        return "No relevant past memories."
    return "\n\n".join(f"[memory] {item.value['text']}" for item in items)
