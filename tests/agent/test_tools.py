from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.tools import make_search_knowledge, recall_memory


def test_search_knowledge_queries_global_and_user_index() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Document(page_content="Contenu A", metadata={"document_id": "doc_a"}),
    ]
    tool = make_search_knowledge(retriever)

    out = tool.func(query="Um Nyobè", user_id="42")

    retriever.retrieve.assert_called_once_with("Um Nyobè", ["global", "user_42"])
    assert "[source: doc_a]" in out
    assert "Contenu A" in out


def test_search_knowledge_no_results() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    tool = make_search_knowledge(retriever)

    assert tool.func(query="x", user_id="42") == "No relevant documents found."


def test_recall_memory_searches_user_episodes() -> None:
    store = MagicMock()
    store.search.return_value = [SimpleNamespace(value={"text": "Échange passé sur l'UPC."})]

    out = recall_memory.func(query="UPC", user_id="42", store=store)

    store.search.assert_called_once_with(("episodes", "42"), query="UPC", limit=5)
    assert "[memory]" in out
    assert "Échange passé sur l'UPC." in out


def test_recall_memory_no_results() -> None:
    store = MagicMock()
    store.search.return_value = []

    assert recall_memory.func(query="x", user_id="42", store=store) == "No relevant past memories."
