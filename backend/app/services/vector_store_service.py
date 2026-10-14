
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import embed_texts
from app.services.qdrant_service import (
    COLLECTION_NAME,
    get_qdrant_client,
)


def store_document_chunks(
    db: Session,
    document_id: int,
) -> int:
    """Embed and store a document's chunks in Qdrant."""

    document = db.query(Document).filter(
        Document.id == document_id
    ).first()

    if document is None:
        raise ValueError(f"Document {document_id} does not exist.")

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    if not chunks:
        return 0

    vectors = embed_texts([chunk.content for chunk in chunks])
    client = get_qdrant_client()

    points = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"enterprise-ai-agent:document:{document_id}:chunk:{chunk.id}",
            )
        )

        payload = {
            "document_id": document_id,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "user_id": document.user_id,
            "filename": document.filename,
            "content": chunk.content,
            "chunk_metadata": chunk.chunk_metadata or {},
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)
