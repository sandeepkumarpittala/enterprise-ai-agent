
from typing import Any

from app.services.llm_service import generate_answer
from app.services.search_service import search_documents


def answer_question(
    question: str,
    user_id: int,
    limit: int = 5,
    conversation_history: list[dict[str, str]] | None = None,
    document_id: int | None = None,
) -> dict[str, Any]:
    """Retrieve user-owned document chunks and generate a grounded answer.

    When document_id is provided, restrict retrieval to that document.
    Otherwise, search across all documents belonging to the user.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if (
        not isinstance(user_id, int)
        or isinstance(user_id, bool)
        or user_id <= 0
    ):
        raise ValueError("user_id must be a positive integer.")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 20
    ):
        raise ValueError("limit must be between 1 and 20.")

    if document_id is not None and (
        not isinstance(document_id, int)
        or isinstance(document_id, bool)
        or document_id <= 0
    ):
        raise ValueError("document_id must be a positive integer.")

    question = question.strip()
    history = conversation_history or []

    # Keep only valid user and assistant messages.
    recent_history = [
        item
        for item in history
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        )
    ][-6:]

    # Find the most recent completed, adjacent user-assistant exchange.
    previous_user_question = None
    previous_assistant_answer = None

    for index in range(len(recent_history) - 1, 0, -1):
        previous = recent_history[index - 1]
        current = recent_history[index]

        if (
            previous["role"] == "user"
            and current["role"] == "assistant"
        ):
            previous_user_question = previous["content"].strip()
            previous_assistant_answer = current["content"].strip()
            break

    # Resolve short follow-up questions using the previous exchange.
    retrieval_query = question
    generation_question = question

    if (
        len(question.split()) <= 5
        and previous_user_question
        and previous_assistant_answer
    ):
        retrieval_query = (
            f"Previous user question: {previous_user_question}\n"
            f"Previous assistant answer: {previous_assistant_answer}\n"
            f"Current follow-up question: {question}"
        )

        generation_question = (
            f"The previous exchange was:\n"
            f"User: {previous_user_question}\n"
            f"Assistant: {previous_assistant_answer}\n\n"
            f"Current follow-up question: {question}\n\n"
            f"Answer the follow-up directly using the retrieved context. "
            f"Explain the specific purpose or process being asked about. "
            f"Do not merely repeat the previous answer."
        )

    search_kwargs: dict[str, Any] = {
        "query": retrieval_query,
        "user_id": user_id,
        "limit": limit,
    }

    if document_id is not None:
        search_kwargs["document_id"] = document_id

    retrieved_chunks = search_documents(**search_kwargs)

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": (
                "I could not find relevant information in your "
                "uploaded documents."
            ),
            "sources": [],
            "retrieved_chunks": 0,
        }

    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        raw_content = chunk.get("content")
        content = (
            raw_content.strip()
            if isinstance(raw_content, str)
            else ""
        )

        if not content:
            continue

        filename = chunk.get("filename") or "Unknown document"
        retrieved_document_id = chunk.get("document_id")
        chunk_id = chunk.get("chunk_id")

        context_parts.append(
            f"[Source: {filename}; Document ID: {retrieved_document_id}; "
            f"Chunk ID: {chunk_id}]\n{content}"
        )

        sources.append(
            {
                "filename": filename,
                "document_id": retrieved_document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            }
        )

    if not context_parts:
        return {
            "question": question,
            "answer": (
                "I could not find readable text in the retrieved "
                "document chunks."
            ),
            "sources": [],
            "retrieved_chunks": 0,
        }

    context = "\n\n---\n\n".join(context_parts)

    answer = generate_answer(
        question=generation_question,
        context=context,
        conversation_history=recent_history,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(context_parts),
    }
