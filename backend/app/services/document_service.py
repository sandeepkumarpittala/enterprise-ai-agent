
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.chunk_service import save_document_chunks
from app.services.document_extractor import extract_text
from app.services.vector_store_service import store_document_chunks


UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def save_uploaded_document(
    file: UploadFile,
    user_id: int,
    db: Session,
) -> Document:
    original_filename = Path(file.filename or "").name.strip()

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload PDF, DOCX, or TXT/Markdown.",
        )

    content = file.file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB limit.",
        )

    content_hash = hashlib.sha256(content).hexdigest()

    duplicate = (
        db.query(Document)
        .filter(
            Document.user_id == user_id,
            Document.content_hash == content_hash,
        )
        .first()
    )

    if duplicate:
        raise HTTPException(
            status_code=409,
            detail="You have already uploaded a document with identical content.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    # Phase 1: save the file, document and chunks in PostgreSQL.
    try:
        destination.write_bytes(content)

        document = Document(
            user_id=user_id,
            filename=original_filename[:255],
            file_type=extension.lstrip("."),
            storage_path=str(destination.resolve()),
            content_hash=content_hash,
            duplicate_protection_enabled=True,
            status="processing",
        )

        db.add(document)
        db.flush()

        extracted_text = extract_text(document.storage_path)

        chunks = save_document_chunks(
            db=db,
            document_id=document.id,
            text=extracted_text,
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text chunks were generated from the document.",
            )

        document_id = document.id
        db.commit()

    except IntegrityError as exc:
        db.rollback()

        if destination.exists():
            destination.unlink()

        if "uq_documents_user_content_hash_protected" in str(exc.orig):
            raise HTTPException(
                status_code=409,
                detail="You have already uploaded a document with identical content.",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="A database integrity error occurred while saving the document.",
        ) from exc

    except Exception as exc:
        db.rollback()

        if destination.exists():
            destination.unlink()

        if isinstance(exc, HTTPException):
            raise

        raise HTTPException(
            status_code=500,
            detail="Document extraction or database saving failed.",
        ) from exc

    # Phase 2: index the committed chunks in Qdrant.
    # Keep the file and PostgreSQL records if indexing fails.
    try:
        stored_count = store_document_chunks(
            db=db,
            document_id=document_id,
        )

        if stored_count != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} vectors, stored {stored_count}."
            )

        document = db.get(Document, document_id)
        if document is None:
            raise RuntimeError("Document disappeared after saving.")

        document.status = "processed"
        db.commit()
        db.refresh(document)

        return document

    except Exception as exc:
        db.rollback()

        # Preserve the document and chunks for recovery.
        try:
            document = db.get(Document, document_id)
            if document is not None:
                document.status = "indexing_failed"
                db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=503,
            detail=(
                "The document and chunks were saved, but vector indexing failed. "
                "The document was preserved and needs indexing retry."
            ),
        ) from exc
