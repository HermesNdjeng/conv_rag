import redis as redis_lib
from langchain_core.embeddings import Embeddings
from langgraph.store.redis import RedisStore


def build_redis_store(redis_url: str, embeddings: Embeddings) -> RedisStore:
    """Episodic memory: store past-episode summaries in Redis, retrievable by semantic search.

    Episodes are embedded on their `text` field so a new turn can recall relevant ones.
    """
    conn = redis_lib.from_url(redis_url)
    dims = len(embeddings.embed_query("dimension probe"))
    store = RedisStore(conn, index={"dims": dims, "embed": embeddings, "fields": ["text"]})
    store.setup()
    return store
