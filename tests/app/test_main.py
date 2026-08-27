from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_agent_answer(client: TestClient, service: MagicMock) -> None:
    service.run.return_value = {"messages": [AIMessage(content="Leader de l'UPC.")]}

    response = client.post(
        "/chat", json={"user_id": "u1", "thread_id": "c1", "message": "Qui est Um Nyobè ?"}
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Leader de l'UPC."}
    service.run.assert_called_once_with(user_id="u1", message="Qui est Um Nyobè ?", thread_id="c1")


def test_chat_stream_yields_non_empty_tokens(client: TestClient, service: MagicMock) -> None:
    service.stream.return_value = iter(
        [
            (AIMessageChunk(content="Um "), {}),
            (AIMessageChunk(content=""), {}),  # tool-call chunk — must be filtered out
            (AIMessageChunk(content="Nyobe"), {}),
        ]
    )

    response = client.post(
        "/chat/stream", json={"user_id": "u1", "thread_id": "c1", "message": "Q"}
    )

    assert response.status_code == 200
    assert '"token": "Um "' in response.text
    assert '"token": "Nyobe"' in response.text
    assert '"token": ""' not in response.text  # empty chunk filtered


def test_end_session_consolidates(client: TestClient, service: MagicMock) -> None:
    response = client.post("/sessions/c1/end?user_id=u1")

    assert response.status_code == 200
    assert response.json() == {"status": "session consolidated"}
    service.end_session.assert_called_once_with(user_id="u1", thread_id="c1")
