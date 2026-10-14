
from app.evaluation.runner import (
    answer_matches_expected_concepts,
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
