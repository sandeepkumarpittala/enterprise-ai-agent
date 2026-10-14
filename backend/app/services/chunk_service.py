
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.services.document_chunker import split_text


def save_document_chunks(
    db: Session,
    document_id: int,
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Create chunk records without committing the transaction."""

    chunks = split_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        return []

    chunk_records = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=index,
            content=content,
            token_count=None,
            chunk_metadata={
                "character_count": len(content),
            },
        )
        for index, content in enumerate(chunks)
    ]

    db.add_all(chunk_records)
    db.flush()

    return chunk_records
