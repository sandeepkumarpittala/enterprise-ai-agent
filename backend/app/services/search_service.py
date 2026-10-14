
from typing import Any

from app.services.embedding_service import embed_text
from app.services.qdrant_service import (
    COLLECTION_NAME,
    get_qdrant_client,
)


def search_documents(
    query: str,
    user_id: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find relevant document chunks belonging to one user."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if user_id <= 0:
        raise ValueError("user_id must be a positive integer.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    query_vector = embed_text(query)

    client = get_qdrant_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter={
            "must": [
                {
                    "key": "user_id",
                    "match": {"value": user_id},
                }
            ]
        },
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    matches = []

    for point in results.points:
        payload = point.payload or {}

        matches.append(
            {
                "score": point.score,
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index"),
                "filename": payload.get("filename"),
                "content": payload.get("content"),
                "chunk_metadata": payload.get("chunk_metadata", {}),
            }
        )

    return matches
