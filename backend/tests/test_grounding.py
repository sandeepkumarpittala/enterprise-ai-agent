
from app.evaluation.grounding import (
    claim_is_lexically_supported,
    evaluate_grounding,
    split_claims,
)


def test_split_claims_splits_sentences():
    claims = split_claims("Qdrant stores vectors. FastAPI serves the API.")

    assert claims == [
        "Qdrant stores vectors.",
        "FastAPI serves the API.",
    ]

def test_split_claims_ignores_numbered_and_bulleted_list_markers():
    claims = split_claims(
        "1. Qdrant stores vectors.\n"
        "2) FastAPI serves requests.\n"
        "- PostgreSQL stores records.\n"
        "* React renders the interface.\n"
        "• Vite builds the frontend."
    )

    assert claims == [
        "Qdrant stores vectors.",
        "FastAPI serves requests.",
        "PostgreSQL stores records.",
        "React renders the interface.",
        "Vite builds the frontend.",
    ]


def test_supported_claim_matches_retrieved_text():
    assert claim_is_lexically_supported(
        "Qdrant stores vector embeddings.",
        "The application uses Qdrant to store vector embeddings.",
    ) is True


def test_unrelated_claim_is_not_supported():
    assert claim_is_lexically_supported(
        "The application uses PostgreSQL for payments.",
        "Qdrant stores vector embeddings for semantic search.",
    ) is False


def test_grounding_score_is_one_when_all_claims_match():
    result = evaluate_grounding(
        "Qdrant stores vector embeddings.",
        [
            {
                "content": (
                    "The application uses Qdrant to store vector embeddings."
                )
            }
        ],
    )

    assert result["groundedness_score"] == 1.0
    assert result["claims_checked"] == 1
    assert result["supported_claims"] == 1
    assert result["unsupported_claims"] == []


def test_grounding_identifies_unsupported_claim():
    result = evaluate_grounding(
        "Qdrant stores vector embeddings. PostgreSQL processes payments.",
        [
            {
                "content": (
                    "The application uses Qdrant to store vector embeddings."
                )
            }
        ],
    )

    assert result["claims_checked"] == 2
    assert result["supported_claims"] == 1
    assert result["groundedness_score"] == 0.5
    assert result["unsupported_claims"] == [
        "PostgreSQL processes payments."
    ]


def test_empty_answer_returns_no_score():
    result = evaluate_grounding("", [])

    assert result["groundedness_score"] is None
    assert result["claims_checked"] == 0


def test_answer_with_no_evidence_has_zero_groundedness():
    result = evaluate_grounding(
        "Qdrant stores vector embeddings.",
        [],
    )

    assert result["groundedness_score"] == 0.0
    assert result["unsupported_claims"] == [
        "Qdrant stores vector embeddings."
    ]
