from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_agent_service


@pytest.fixture
def service() -> MagicMock:
    """A stand-in AgentService whose run/stream/end_session are mocked per test."""
    return MagicMock()


@pytest.fixture
def client(service: MagicMock) -> Iterator[TestClient]:
    # Override the dependency so endpoints use the mock; no lifespan (no `with`) => the real
    # build_agent_service() is never called.
    app.dependency_overrides[get_agent_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()
