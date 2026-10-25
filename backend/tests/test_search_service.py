
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.search_service import search_documents
from app.services.qdrant_service import COLLECTION_NAME


@patch("app.services.search_service.get_qdrant_client")
@patch("app.services.search_service.embed_text")
def test_search_documents_filters_by_user_id(
    mock_embed_text,
    mock_get_client,
):
    user_id = 42
    mock_embed_text.return_value = [0.1] * 384

    point = SimpleNamespace(
        score=0.95,
        payload={
            "user_id": user_id,
            "document_id": 7,
            "chunk_id": 12,
            "chunk_index": 0,
            "filename": "example.txt",
            "content": "User-owned document content",
            "chunk_metadata": {},
        },
    )

    mock_client = Mock()
    mock_client.query_points.return_value = SimpleNamespace(
        points=[point]
    )
    mock_get_client.return_value = mock_client

    results = search_documents(
        query="example question",
        user_id=user_id,
        limit=5,
    )

    kwargs = mock_client.query_points.call_args.kwargs

    assert kwargs["collection_name"] == COLLECTION_NAME
    assert kwargs["query_filter"] == {
        "must": [
            {
                "key": "user_id",
                "match": {"value": user_id},
            }
        ]
    }
    assert kwargs["limit"] == 5
    assert results[0]["document_id"] == 7
    assert results[0]["content"] == "User-owned document content"
    mock_embed_text.assert_called_once_with("example question")
