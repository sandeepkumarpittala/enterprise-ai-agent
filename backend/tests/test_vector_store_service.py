
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.qdrant_service import COLLECTION_NAME
from app.services.vector_store_service import store_document_chunks


@patch("app.services.vector_store_service.get_qdrant_client")
@patch("app.services.vector_store_service.embed_texts")
def test_store_document_chunks_uses_document_owner(
    mock_embed_texts,
    mock_get_client,
):
    user_id = 42
    document_id = 7

    document = SimpleNamespace(
        id=document_id,
        user_id=user_id,
        filename="private.txt",
    )
    chunk = SimpleNamespace(
        id=12,
        document_id=document_id,
        chunk_index=0,
        content="Private document content",
        chunk_metadata={},
    )

    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = document
    (
        mock_db.query.return_value
        .filter.return_value
        .order_by.return_value
        .all.return_value
    ) = [chunk]

    mock_embed_texts.return_value = [[0.1] * 384]

    mock_client = Mock()
    mock_get_client.return_value = mock_client

    stored_count = store_document_chunks(
        db=mock_db,
        document_id=document_id,
    )

    assert stored_count == 1

    mock_client.upsert.assert_called_once()
    kwargs = mock_client.upsert.call_args.kwargs

    assert kwargs["collection_name"] == COLLECTION_NAME
    assert kwargs["wait"] is True
    assert len(kwargs["points"]) == 1

    payload = kwargs["points"][0].payload
    assert payload["document_id"] == document_id
    assert payload["chunk_id"] == chunk.id
    assert payload["user_id"] == user_id
    assert payload["filename"] == document.filename
    assert payload["content"] == chunk.content
