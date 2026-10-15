from unittest.mock import patch

import pytest

from app.services.rag_service import answer_question


@patch("app.services.rag_service.generate_answer")
@patch("app.services.rag_service.search_documents")
def test_answer_question_returns_grounded_answer(
    mock_search_documents,
    mock_generate_answer,
):
    mock_search_documents.return_value = [
        {
            "content": "The vector database is Qdrant.",
            "filename": "rag-test.pdf",
            "document_id": 1,
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "score": 0.92,
        }
    ]
    mock_generate_answer.return_value = "The vector database is Qdrant."

    result = answer_question(
        question="Which vector database is used?",
        user_id=1,
        limit=5,
    )

    assert result["answer"] == "The vector database is Qdrant."
    assert result["retrieved_chunks"] == 1
    assert result["sources"][0]["filename"] == "rag-test.pdf"

    mock_search_documents.assert_called_once_with(
        query="Which vector database is used?",
        user_id=1,
        limit=5,
    )
    mock_generate_answer.assert_called_once()


@patch("app.services.rag_service.generate_answer")
@patch("app.services.rag_service.search_documents")
def test_no_relevant_chunks_returns_safe_answer(
    mock_search_documents,
    mock_generate_answer,
):
    mock_search_documents.return_value = []

    result = answer_question(
        question="What is the vector database?",
        user_id=1,
    )

    assert "could not find relevant information" in result["answer"].lower()
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0
    mock_generate_answer.assert_not_called()


@pytest.mark.parametrize(
    ("question", "user_id", "limit"),
    [
        ("", 1, 5),
        ("   ", 1, 5),
        ("Valid question", 0, 5),
        ("Valid question", 1, 0),
        ("Valid question", 1, 21),
    ],
)
def test_invalid_inputs_raise_value_error(question, user_id, limit):
    with pytest.raises(ValueError):
        answer_question(
            question=question,
            user_id=user_id,
            limit=limit,
        )

@patch("app.services.rag_service.generate_answer")
@patch("app.services.rag_service.search_documents")
def test_follow_up_question_uses_previous_question_for_retrieval(
    mock_search_documents,
    mock_generate_answer,
):
    mock_search_documents.return_value = [
        {
            "content": "The vector database is Qdrant.",
            "filename": "rag-test.pdf",
            "document_id": 1,
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "score": 0.92,
        }
    ]
    mock_generate_answer.return_value = "Qdrant."

    history = [
        {
            "role": "user",
            "content": "What vector database does this project use?",
        },
        {
            "role": "assistant",
            "content": "The vector database used in this project is Qdrant.",
        },
    ]

    result = answer_question(
        question="Which database was that?",
        user_id=1,
        limit=5,
        conversation_history=history,
    )

    assert result["answer"] == "Qdrant."

    mock_search_documents.assert_called_once()
    retrieval_query = mock_search_documents.call_args.kwargs["query"]
    assert "What vector database does this project use?" in retrieval_query
    assert "Which database was that?" in retrieval_query

    mock_generate_answer.assert_called_once()
    assert (
        mock_generate_answer.call_args.kwargs["conversation_history"]
        == history
    )

@patch("app.services.rag_service.generate_answer")
@patch("app.services.rag_service.search_documents")
def test_follow_up_reuses_previous_answer_when_fully_supported(
    mock_search_documents,
    mock_generate_answer,
):
    mock_search_documents.return_value = [
        {
            "content": "The vector database is Qdrant.",
            "filename": "rag-test.pdf",
            "document_id": 1,
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "score": 0.92,
        }
    ]

    history = [
        {
            "role": "user",
            "content": "What vector database does this project use?",
        },
        {
            "role": "assistant",
            "content": "The vector database is Qdrant.",
        },
    ]

    result = answer_question(
        question="Which database was that?",
        user_id=1,
        conversation_history=history,
    )

    assert result["answer"] == "The vector database is Qdrant."
    mock_generate_answer.assert_not_called()
