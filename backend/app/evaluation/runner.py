
"""Run a deterministic baseline evaluation against the real RAG pipeline."""

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.grounding import evaluate_grounding
from app.services.rag_service import answer_question
from app.services.search_service import search_documents


UNANSWERED_MARKERS = (
    "could not find",
    "not found in the provided documents",
    "not found in the document",
    "does not contain",
    "doesn't contain",
    "not available in the provided documents",
    "not specified in the provided documents",
    "cannot determine from the provided documents",
)

# Required concepts for deterministic answer checks.
# These are lexical heuristics, not semantic or human evaluation.
EXPECTED_CONCEPTS = {
    "rag-001": [("qdrant",)],
    "rag-002": [("fastapi",)],
    "rag-003": [("sentence-transformers/all-minilm-l6-v2",)],
    "rag-004": [
        ("extract", "extracted", "extraction"),
        ("chunk", "chunks", "chunking"),
        ("embedding", "embeddings"),
        ("store", "stored", "stores", "storage"),
    ],
    "rag-006": [("react",), ("vite",)],
    "rag-007": [("postgresql",)],
    "rag-008": [("ollama",)],
    "rag-009": [
        ("embedding", "embeds", "embedded"),
        ("qdrant",),
        ("retrieved context", "retrieved chunks", "relevant chunks", "context"),
        ("language model", "llm"),
        ("answer",),
    ],
      "rag-010": [
        ("extract", "extraction"),
        ("chunk", "chunks", "chunking"),
        ("embedding", "embeddings"),
        ("retrieval", "retrieve", "retrieved"),
    ],
    "rag-011": [
        ("reduce unsupported answers", "reduces unsupported answers"),
        ("retrieval", "retrieve", "retrieves"),
        ("source-grounding", "source grounding", "grounded"),
    ],
    "rag-012": [("qdrant",), ("postgresql",)],
}

# Required concepts expected in retrieved evidence.
# This checks lexical coverage, not semantic relevance or entailment.
RETRIEVAL_CONCEPTS = {
    "rag-001": [("qdrant",)],
    "rag-002": [("fastapi",)],
    "rag-003": [("sentence-transformers/all-minilm-l6-v2",)],
    "rag-004": [
        ("extract", "extracted", "extraction"),
        ("chunk", "chunks", "chunking"),
        ("embedding", "embeddings"),
        ("store", "stored", "storage"),
    ],
    "rag-006": [("react",), ("vite",)],
    "rag-007": [("postgresql",)],
    "rag-008": [("ollama",)],
    "rag-009": [
        ("embedding", "embeds", "embedded"),
        ("qdrant",),
        ("retrieval", "retrieve", "retrieved"),
        ("language model", "llm"),
    ],
    "rag-010": [
        ("extract", "extraction"),
        ("chunk", "chunks", "chunking"),
        ("embedding", "embeddings"),
        ("retrieval", "retrieve", "retrieved"),
        ("authenticated user", "document owner", "user"),
    ],
    "rag-011": [
        ("retrieval", "retrieve", "retrieves"),
        ("source-grounding", "source grounding", "grounded"),
        ("unsupported answers",),
    ],
    "rag-012": [("qdrant",), ("postgresql",)],
}


def normalize(text: str) -> str:
    """Normalize text for simple, deterministic comparisons."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_abstention(answer: str) -> bool:
    """Detect common explicit statements that the source lacks an answer."""
    normalized_answer = normalize(answer)
    return any(marker in normalized_answer for marker in UNANSWERED_MARKERS)


def answer_matches_expected_concepts(case: dict, answer: str) -> bool:
    """Check required answer concepts and reject explicit abstentions."""
    concept_groups = EXPECTED_CONCEPTS.get(case["id"])

    if not concept_groups or is_abstention(answer):
        return False

    normalized_answer = normalize(answer)

    return all(
        any(normalize(term) in normalized_answer for term in alternatives)
        for alternatives in concept_groups
    )


def retrieved_evidence_matches(
    case: dict,
    sources: list[dict[str, Any]],
) -> bool:
    """Check expected lexical evidence in chunks from the expected source."""
    expected_source = case.get("expected_source")
    concept_groups = RETRIEVAL_CONCEPTS.get(case["id"])

    if not expected_source or not concept_groups or not sources:
        return False

    expected_name = normalize(expected_source)

    matching_sources = [
        source
        for source in sources
        if expected_name in normalize(source.get("filename") or "")
    ]

    if not matching_sources:
        return False

    retrieved_text = normalize(
        " ".join(source.get("content") or "" for source in matching_sources)
    )

    return all(
        any(normalize(term) in retrieved_text for term in alternatives)
        for alternatives in concept_groups
    )


def calculate_retrieval_metrics(
    case: dict,
    sources: list[dict[str, Any]],
    k: int,
) -> dict[str, float | None]:
    """Calculate document-level Precision@K and reciprocal rank."""
    expected_source = case.get("expected_source")

    if not expected_source or k <= 0:
        return {"precision_at_k": None, "reciprocal_rank": None}

    expected_name = normalize(expected_source)
    top_k_sources = sources[:k]

    relevant = [
        expected_name in normalize(source.get("filename") or "")
        for source in top_k_sources
    ]

    precision_at_k = sum(relevant) / k

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": round(precision_at_k, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


def calculate_chunk_retrieval_metrics(
    relevant_chunk_ids: set[int | str],
    retrieved_chunks: list[dict[str, Any]],
    k: int,
) -> dict[str, float | int | None]:
    """Calculate chunk-level Precision@K and Recall@K."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if not relevant_chunk_ids:
        return {
            "chunk_precision_at_k": None,
            "chunk_recall_at_k": None,
            "relevant_chunks_retrieved": 0,
        }

    top_k = retrieved_chunks[:k]
    retrieved_ids = [
        chunk.get("chunk_id")
        for chunk in top_k
        if chunk.get("chunk_id") is not None
    ]

    relevant_retrieved = len(set(retrieved_ids) & relevant_chunk_ids)

    return {
        "chunk_precision_at_k": round(relevant_retrieved / k, 4),
        "chunk_recall_at_k": round(
            relevant_retrieved / len(relevant_chunk_ids), 4
        ),
        "relevant_chunks_retrieved": relevant_retrieved,
    }


def evaluate_case(
    case: dict,
    user_id: int,
    limit: int,
    document_id: int | None = None,
) -> dict:
    """Evaluate one question against the RAG pipeline."""
    started = time.perf_counter()

    result = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "retrieval_hit": False,
        "answer_correct": False,
        "correct_abstention": False,
        "failure": False,
        "latency_ms": None,
        "answer": None,
        "sources": [],
        "error": None,
        "precision_at_k": None,
        "reciprocal_rank": None,
        "chunk_precision_at_k": None,
        "chunk_recall_at_k": None,
        "relevant_chunks_retrieved": 0,
        "groundedness_score": None,
        "claims_checked": 0,
        "supported_claims": 0,
        "unsupported_claims": [],
        "claim_results": [],
    }

    try:
        # Preserve compatibility with callers and mocks when no document ID
        # is provided. When supplied, constrain both generation and evaluation.
        answer_kwargs: dict[str, Any] = {
            "question": case["question"],
            "user_id": user_id,
            "limit": limit,
        }
        if document_id is not None:
            answer_kwargs["document_id"] = document_id

        response = answer_question(**answer_kwargs)

        answer = response.get("answer") or ""
        sources = response.get("sources") or []

        result["answer"] = answer
        result["sources"] = sources
        retrieved_chunks = []

        if case["answerable"]:
            search_kwargs: dict[str, Any] = {
                "query": case["question"],
                "user_id": user_id,
                "limit": limit,
            }
            if document_id is not None:
                search_kwargs["document_id"] = document_id

            retrieved_chunks = search_documents(**search_kwargs)

            result["retrieval_hit"] = retrieved_evidence_matches(
                case, retrieved_chunks
            )

            result["answer_correct"] = answer_matches_expected_concepts(
                case, answer
            )

            result.update(
                calculate_retrieval_metrics(
                    case=case,
                    sources=retrieved_chunks,
                    k=limit,
                )
            )

            result.update(
                calculate_chunk_retrieval_metrics(
                    relevant_chunk_ids=set(case["relevant_chunk_ids"]),
                    retrieved_chunks=retrieved_chunks,
                    k=limit,
                )
            )
        else:
            # Unanswerable questions should be answered with an abstention.
            result["correct_abstention"] = is_abstention(answer)

        # This is a lexical-overlap heuristic, not semantic entailment.
        if case["answerable"] and not is_abstention(answer):
            grounding = evaluate_grounding(
                answer=answer,
                retrieved_chunks=retrieved_chunks,
            )
        else:
            grounding = {
                "groundedness_score": None,
                "claims_checked": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "claim_results": [],
            }

        result.update(grounding)

    except Exception as exc:
        result["failure"] = True
        result["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        result["latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )

    return result


def run_evaluation(
    user_id: int,
    limit: int = 5,
    document_id: int | None = None,
) -> dict:
    """Run all evaluation cases and aggregate their metrics."""
    results = [
        evaluate_case(case, user_id, limit, document_id)
        for case in EVALUATION_DATASET
    ]

    total = len(results)
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    precision_values = [
        r["precision_at_k"]
        for r in answerable
        if r["precision_at_k"] is not None
    ]

    reciprocal_rank_values = [
        r["reciprocal_rank"]
        for r in answerable
        if r["reciprocal_rank"] is not None
    ]

    chunk_precision_values = [
        r["chunk_precision_at_k"]
        for r in answerable
        if r["chunk_precision_at_k"] is not None
    ]

    chunk_recall_values = [
        r["chunk_recall_at_k"]
        for r in answerable
        if r["chunk_recall_at_k"] is not None
    ]

    groundedness_values = [
        r["groundedness_score"]
        for r in answerable
        if r["groundedness_score"] is not None
    ]

    return {
        "evaluation_name": "enterprise-ai-rag-expanded",
        "user_id": user_id,
        "document_id": document_id,
        "retrieval_limit": limit,
        "total_questions": total,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "retrieval_hit_rate": (
            round(
                sum(r["retrieval_hit"] for r in answerable) / len(answerable),
                4,
            )
            if answerable
            else None
        ),
        "precision_at_k": (
            round(mean(precision_values), 4) if precision_values else None
        ),
        "mean_reciprocal_rank": (
            round(mean(reciprocal_rank_values), 4)
            if reciprocal_rank_values
            else None
        ),
        "answer_correctness": (
            round(
                sum(r["answer_correct"] for r in answerable) / len(answerable),
                4,
            )
            if answerable
            else None
        ),
        "average_groundedness_score": (
            round(mean(groundedness_values), 4)
            if groundedness_values
            else None
        ),
        "grounding_method": "lexical_overlap_heuristic",
        "failure_rate": (
            round(sum(r["failure"] for r in results) / total, 4)
            if total
            else None
        ),
        "correct_abstention_rate": (
            round(
                sum(r["correct_abstention"] for r in unanswerable)
                / len(unanswerable),
                4,
            )
            if unanswerable
            else None
        ),
        "average_latency_ms": (
            round(mean(r["latency_ms"] for r in results), 2)
            if results
            else None
        ),
        "chunk_precision_at_k": (
            round(mean(chunk_precision_values), 4)
            if chunk_precision_values
            else None
        ),
        "chunk_recall_at_k": (
            round(mean(chunk_recall_values), 4)
            if chunk_recall_values
            else None
        ),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Enterprise AI RAG pipeline."
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Existing user ID who owns the indexed evaluation document.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks retrieved per question (1-20).",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=None,
        help="Optional document ID to restrict retrieval to one document.",
    )

    args = parser.parse_args()

    if args.user_id <= 0:
        parser.error("--user-id must be a positive integer.")

    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20.")

    if args.document_id is not None and args.document_id <= 0:
        parser.error("--document-id must be a positive integer.")

    report = run_evaluation(
        user_id=args.user_id,
        limit=args.limit,
        document_id=args.document_id,
    )

    report_path = (
        Path(__file__).resolve().parents[2] / "rag_evaluation_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed report saved to: {report_path}")

    print("\nPer-question results:")
    for case in report["cases"]:
        status = "ERROR" if case["failure"] else "OK"
        print(
            f"- {case['id']} [{status}] "
            f"retrieval_hit={case['retrieval_hit']} "
            f"precision_at_k={case['precision_at_k']} "
            f"reciprocal_rank={case['reciprocal_rank']} "
            f"answer_correct={case['answer_correct']} "
            f"groundedness_score={case['groundedness_score']} "
            f"claims_checked={case['claims_checked']} "
            f"supported_claims={case['supported_claims']} "
            f"correct_abstention={case['correct_abstention']} "
            f"latency_ms={case['latency_ms']}"
        )

        if case["unsupported_claims"]:
            print(f"  Unsupported claims: {case['unsupported_claims']}")

        if case["error"]:
            print(f"  Error: {case['error']}")


if __name__ == "__main__":
    main()
