
from pathlib import Path

from docx import Document as DocxDocument
from fastapi import HTTPException, status
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def extract_text(file_path: str) -> str:
    """
    Extract readable text from a supported document.

    Returns cleaned text and raises an HTTPException for unsupported,
    empty, or unreadable documents.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported types: PDF, DOCX, TXT, Markdown.",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The document file was not found.",
        )

    try:
        if extension in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8-sig")

        elif extension == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        else:  # .docx
            document = DocxDocument(str(path))
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
            text = "\n".join(paragraphs)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document could not be read. Check that it is valid and not corrupted.",
        ) from exc

    cleaned_text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ).strip()

    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text was found in the document.",
        )

    return cleaned_text
