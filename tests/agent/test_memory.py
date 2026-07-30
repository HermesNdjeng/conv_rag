from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.memory import _format_transcript, consolidate_session


def test_format_transcript_keeps_human_and_ai_only() -> None:
    messages = [
        HumanMessage(content="Qui est Um Nyobè ?"),
        AIMessage(content=""),  # empty AI (tool-call turn) — ignored
        ToolMessage(content="search result", tool_call_id="1"),  # ignored
        AIMessage(content="Leader de l'UPC."),
    ]
    assert _format_transcript(messages) == "User: Qui est Um Nyobè ?\nAssistant: Leader de l'UPC."


def test_consolidate_session_writes_episode() -> None:
    agent = MagicMock()
    agent.get_state.return_value = SimpleNamespace(
        values={"messages": [HumanMessage(content="Q"), AIMessage(content="A")]}
    )
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Résumé de la session.")
    store = MagicMock()

    consolidate_session(agent, store, llm, user_id="42", thread_id="conv_1")

    call = store.put.call_args
    assert call.args[0] == ("episodes", "42")  # user-scoped episodes namespace
    assert call.args[2] == {"text": "Résumé de la session."}


def test_consolidate_session_skips_empty_transcript() -> None:
    agent = MagicMock()
    agent.get_state.return_value = SimpleNamespace(values={"messages": []})
    llm = MagicMock()
    store = MagicMock()

    consolidate_session(agent, store, llm, user_id="42", thread_id="conv_1")

    llm.invoke.assert_not_called()
    store.put.assert_not_called()
