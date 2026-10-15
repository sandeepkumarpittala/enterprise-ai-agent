
from typing import Any

from app.services.llm_service import generate_answer
from app.services.search_service import search_documents


ABSTENTION_ANSWER = (
    "I could not find the answer in the provided documents."
)


def answer_question(
    question: str,
    user_id: int,
    limit: int = 5,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Retrieve user-owned document chunks and generate a grounded answer."""

    # Validate inputs.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        raise ValueError("user_id must be a positive integer.")

    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")

    question = question.strip()
    history = conversation_history or []

    # Keep the six most recent valid conversation messages.
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

    # Find the immediately preceding user question and assistant answer.
    previous_user_question = None
    previous_assistant_answer = None

    for item in reversed(recent_history):
        if (
            previous_assistant_answer is None
            and item["role"] == "assistant"
        ):
            previous_assistant_answer = item["content"].strip()

        elif (
            previous_user_question is None
            and item["role"] == "user"
        ):
            previous_user_question = item["content"].strip()

        if (
            previous_user_question is not None
            and previous_assistant_answer is not None
        ):
            break

    # Expand short follow-up questions to improve document retrieval.
    retrieval_query = question

    if len(question.split()) <= 5 and previous_user_question:
        retrieval_query = (
            f"{previous_user_question}\n"
            f"Follow-up question: {question}"
        )

    # Retrieve relevant chunks belonging to this user.
    retrieved_chunks = search_documents(
        query=retrieval_query,
        user_id=user_id,
        limit=limit,
    )

    # Abstain when retrieval returns no chunks.
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

    # Build document context and source metadata.
    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        raw_content = chunk.get("content")
        content = raw_content.strip() if isinstance(raw_content, str) else ""

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

    # Abstain if the retrieved chunks contain no readable text.
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

    # Identify short follow-up questions that may refer to a previous answer.
    short_follow_up = (
        len(question.split()) <= 5
        and previous_user_question is not None
        and previous_assistant_answer is not None
        and question.lower().startswith(
            ("which ", "what ", "who ", "where ", "when ", "how ")
        )
    )

    # Reuse a previous answer only when the complete answer appears
    # verbatim in the retrieved document context.
    previous_answer_supported = (
        previous_assistant_answer is not None
        and previous_assistant_answer.casefold() in context.casefold()
    )

    if short_follow_up and previous_answer_supported:
        answer = previous_assistant_answer
    else:
        answer = generate_answer(
            question=question,
            context=context,
            conversation_history=recent_history,
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(context_parts),
    }
