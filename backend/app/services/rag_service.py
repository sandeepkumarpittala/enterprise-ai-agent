from typing import Any

from app.services.llm_service import generate_answer
from app.services.search_service import search_documents


def answer_question(
    question: str,
    user_id: int,
    limit: int = 5,
) -> dict[str, Any]:
    """Retrieve user-owned document chunks and generate a grounded answer."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if user_id <= 0:
        raise ValueError("user_id must be a positive integer.")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    retrieved_chunks = search_documents(
        query=question,
        user_id=user_id,
        limit=limit,
    )

    if not retrieved_chunks:
        return {
            "question": question.strip(),
            "answer": (
                "I could not find relevant information in your "
                "uploaded documents."
            ),
            "sources": [],
            "retrieved_chunks": 0,
        }

    context_parts = []
    sources = []

    for chunk in retrieved_chunks:
        content = (chunk.get("content") or "").strip()

        if not content:
            continue

        filename = chunk.get("filename") or "Unknown document"
        document_id = chunk.get("document_id")
        chunk_id = chunk.get("chunk_id")

        context_parts.append(
            f"[Source: {filename}; Document ID: {document_id}; "
            f"Chunk ID: {chunk_id}]\n{content}"
        )

        sources.append(
            {
                "filename": filename,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            }
        )

    if not context_parts:
        return {
            "question": question.strip(),
            "answer": (
                "I could not find readable text in the retrieved "
                "document chunks."
            ),
            "sources": [],
            "retrieved_chunks": 0,
        }

    context = "\n\n---\n\n".join(context_parts)

    answer = generate_answer(
        question=question.strip(),
        context=context,
    )

    return {
        "question": question.strip(),
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(context_parts),
    }
