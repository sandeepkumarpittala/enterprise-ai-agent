
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.core.config import settings


COLLECTION_NAME = "enterprise_documents"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create and reuse the Qdrant connection."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection() -> None:
    """Create the vector collection if it does not already exist."""
    client = get_qdrant_client()

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
