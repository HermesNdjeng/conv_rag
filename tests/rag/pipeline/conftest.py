from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def make_minio() -> Callable[..., MagicMock]:
    """Factory building a fake MinIO client that lists the given object names and returns a
    readable response for each get_object call."""

    def _make(*object_names: str) -> MagicMock:
        client = MagicMock()
        client.list_objects.return_value = [
            SimpleNamespace(object_name=name) for name in object_names
        ]
        response = MagicMock()
        response.read.return_value = b"# doc"
        client.get_object.return_value = response
        return client

    return _make


@pytest.fixture
def loader() -> MagicMock:
    """Fake DocumentLoader returning one chunk per document by default."""
    loader = MagicMock()
    loader.load_from_bytes.return_value = [SimpleNamespace()]
    return loader
