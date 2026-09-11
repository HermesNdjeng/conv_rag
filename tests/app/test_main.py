from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_message_returns_agent_answer(client: TestClient, service: MagicMock) -> None:
    service.run.return_value = {"messages": [AIMessage(content="Leader de l'UPC.")]}

    response = client.post("/users/u1/sessions/c1/messages", json={"message": "Qui est Um Nyobè ?"})

    assert response.status_code == 200
    assert response.json() == {"response": "Leader de l'UPC."}
    service.run.assert_called_once_with(user_id="u1", message="Qui est Um Nyobè ?", thread_id="c1")


def test_post_message_streams_when_accept_event_stream(
    client: TestClient, service: MagicMock
) -> None:
    service.stream.return_value = iter([])

    response = client.post(
        "/users/u1/sessions/c1/messages",
        json={"message": "Q"},
        headers={"Accept": "text/event-stream"},
    )

    # Content negotiation: the Accept header routes to the streaming branch, not the JSON one.
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    service.stream.assert_called_once_with(
        user_id="u1", message="Q", thread_id="c1", stream_mode="messages"
    )
    service.run.assert_not_called()


def test_patch_session_ended_consolidates(client: TestClient, service: MagicMock) -> None:
    response = client.patch("/users/u1/sessions/c1", json={"status": "ended"})

    assert response.status_code == 200
    assert response.json() == {"status": "ended"}
    service.end_session.assert_called_once_with(user_id="u1", thread_id="c1")
