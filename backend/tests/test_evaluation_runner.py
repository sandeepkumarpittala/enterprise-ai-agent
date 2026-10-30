
from app.evaluation.runner import (
    answer_matches_expected_concepts,
    calculate_chunk_retrieval_metrics,
    calculate_retrieval_metrics,
    is_abstention,
    normalize,
)


def test_rag_011_accepts_correct_source_grounded_answer():
    case = {"id": "rag-011"}
    answer = (
        "RAG can reduce unsupported answers when retrieval "
        "and source-grounding work correctly."
    )

    assert answer_matches_expected_concepts(case, answer) is True


def test_answerable_case_fails_when_required_concept_is_missing():
    case = {"id": "rag-011"}
    answer = "RAG can reduce unsupported answers."

    assert answer_matches_expected_concepts(case, answer) is False


def test_unknown_answerable_case_fails_safely():
    case = {"id": "rag-unknown"}
    answer = "This answer sounds correct."

    assert answer_matches_expected_concepts(case, answer) is False


def test_abstention_marker_is_recognized():
    answer = "This information was not found in the provided documents."

    assert is_abstention(answer) is True


def test_normal_answer_is_not_an_abstention():
    answer = "Qdrant stores vector embeddings."

    assert is_abstention(answer) is False


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  RAG   Uses\nQdrant  ") == "rag uses qdrant"


def test_answer_correctness_rejects_answer_missing_required_concepts():
    case = {"id": "rag-001"}
    answer = "The application uses PostgreSQL as its relational database."

    assert answer_matches_expected_concepts(case, answer) is False


def test_answer_correctness_does_not_accept_unrelated_answer():
    case = {"id": "rag-001"}
    answer = "I could not find the answer in the provided documents."

    assert answer_matches_expected_concepts(case, answer) is False


def test_answerable_question_rejects_abstention():
    case = {"id": "rag-001"}
    answer = "I could not find the answer in the provided documents."

    assert answer_matches_expected_concepts(case, answer) is False


def test_expected_fact_must_appear_in_retrieved_content():
    expected_fact = "Qdrant is the vector database."

    relevant_chunk = (
        "The application uses Qdrant as its vector database."
    )
    irrelevant_chunk = (
        "The application uses PostgreSQL for relational data."
    )

    assert "qdrant" in relevant_chunk.lower()
    assert "qdrant" not in irrelevant_chunk.lower()


def test_precision_and_reciprocal_rank_when_expected_document_is_first():
    case = {"expected_source": "enterprise_ai_rag_test_document.pdf"}
    sources = [
        {"filename": "enterprise_ai_rag_test_document.pdf"},
        {"filename": "other_document.pdf"},
        {"filename": "another_document.pdf"},
        {"filename": "third_document.pdf"},
        {"filename": "fourth_document.pdf"},
    ]

    metrics = calculate_retrieval_metrics(case, sources, k=5)

    assert metrics == {
        "precision_at_k": 0.2,
        "reciprocal_rank": 1.0,
    }


def test_reciprocal_rank_when_expected_document_is_third():
    case = {"expected_source": "enterprise_ai_rag_test_document.pdf"}
    sources = [
        {"filename": "other_document.pdf"},
        {"filename": "another_document.pdf"},
        {"filename": "enterprise_ai_rag_test_document.pdf"},
        {"filename": "fourth_document.pdf"},
        {"filename": "fifth_document.pdf"},
    ]

    metrics = calculate_retrieval_metrics(case, sources, k=5)

    assert metrics == {
        "precision_at_k": 0.2,
        "reciprocal_rank": 0.3333,
    }


def test_metrics_are_zero_when_expected_document_is_missing():
    case = {"expected_source": "enterprise_ai_rag_test_document.pdf"}
    sources = [
        {"filename": "other_document.pdf"},
        {"filename": "another_document.pdf"},
    ]

    metrics = calculate_retrieval_metrics(case, sources, k=5)

    assert metrics == {
        "precision_at_k": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_empty_retrieval_results_return_zero_metrics():
    case = {"expected_source": "enterprise_ai_rag_test_document.pdf"}

    metrics = calculate_retrieval_metrics(case, [], k=5)

    assert metrics == {
        "precision_at_k": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_metrics_return_none_when_expected_source_is_missing():
    case = {"id": "rag-unknown"}
    sources = [{"filename": "some_document.pdf"}]

    metrics = calculate_retrieval_metrics(case, sources, k=5)

    assert metrics == {
        "precision_at_k": None,
        "reciprocal_rank": None,
    }

def test_chunk_metrics_when_relevant_chunks_are_retrieved():
    metrics = calculate_chunk_retrieval_metrics(
        relevant_chunk_ids={10, 20},
        retrieved_chunks=[
            {"chunk_id": 10},
            {"chunk_id": 30},
            {"chunk_id": 20},
            {"chunk_id": 40},
        ],
        k=4,
    )

    assert metrics == {
        "chunk_precision_at_k": 0.5,
        "chunk_recall_at_k": 1.0,
        "relevant_chunks_retrieved": 2,
    }


def test_chunk_metrics_when_no_relevant_chunks_are_retrieved():
    metrics = calculate_chunk_retrieval_metrics(
        relevant_chunk_ids={10, 20},
        retrieved_chunks=[
            {"chunk_id": 30},
            {"chunk_id": 40},
        ],
        k=2,
    )

    assert metrics == {
        "chunk_precision_at_k": 0.0,
        "chunk_recall_at_k": 0.0,
        "relevant_chunks_retrieved": 0,
    }


def test_chunk_metrics_are_none_without_relevance_labels():
    metrics = calculate_chunk_retrieval_metrics(
        relevant_chunk_ids=set(),
        retrieved_chunks=[{"chunk_id": 10}],
        k=1,
    )

    assert metrics == {
        "chunk_precision_at_k": None,
        "chunk_recall_at_k": None,
        "relevant_chunks_retrieved": 0,
    }


def test_chunk_metrics_reject_nonpositive_k():
    import pytest

    with pytest.raises(ValueError):
        calculate_chunk_retrieval_metrics(
            relevant_chunk_ids={10},
            retrieved_chunks=[{"chunk_id": 10}],
            k=0,
        )
