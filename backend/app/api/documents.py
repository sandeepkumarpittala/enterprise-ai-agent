
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, get_db
from app.models.document import Document
from app.models.user import User
from app.schemas.document_search import DocumentSearchRequest
from app.services.document_service import save_uploaded_document
from app.services.search_service import search_documents

from pathlib import Path

from qdrant_client import models

from app.services.qdrant_service import (
    COLLECTION_NAME,
    get_qdrant_client,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    document: Document = save_uploaded_document(
        file=file,
        user_id=current_user.id,
        db=db,
    )

    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "status": document.status,
        "created_at": document.created_at,
    }


@router.post("/search", response_model=dict)
def search_uploaded_documents(
    request: DocumentSearchRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        results = search_documents(
            query=request.query,
            user_id=current_user.id,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "query": request.query,
        "count": len(results),
        "results": results,
    }
@router.get("", response_model=dict)
def list_uploaded_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    statement = (
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )

    documents = db.scalars(statement).all()

    return {
        "count": len(documents),
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "status": document.status,
                "created_at": document.created_at,
            }
            for document in documents
        ],
    }

@router.delete("/{document_id}", response_model=dict)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # Only allow the owner to delete the document.
    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # Remove this user's vectors for this document from Qdrant.
    client = get_qdrant_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=current_user.id),
                    ),
                ]
            )
        ),
        wait=True,
    )

    # Delete PostgreSQL records. Chunk rows use ON DELETE CASCADE.
    storage_path = Path(document.storage_path)

    try:
        db.delete(document)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=(
                "Database deletion failed. The document record was preserved, "
                "but its vectors may need to be re-indexed."
            ),
        ) from exc

    # Remove the uploaded file after the database commit.
    try:
        storage_path.unlink(missing_ok=True)
    except OSError:
        # Database and vector deletion succeeded, but file cleanup did not.
        raise HTTPException(
            status_code=500,
            detail=(
                "Document records were deleted, but the uploaded file "
                "could not be removed from disk."
            ),
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }
