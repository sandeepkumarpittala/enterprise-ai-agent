
"""Run a deterministic baseline evaluation against the real RAG pipeline."""

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

from app.evaluation.dataset import EVALUATION_DATASET
from app.services.rag_service import answer_question


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

# Each inner tuple represents one required concept.
# At least one alternative in each tuple must appear in the answer.
# These are transparent heuristic checks, not semantic or human evaluation.
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
    "rag-006": [
        ("react",),
        ("vite",),
    ],
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
        ("authenticated user", "document owner", "belongs to the user", "user"),
    ],
    "rag-011": [
    ("reduce unsupported answers", "reduces unsupported answers"),
    ("retrieval", "retrieve", "retrieves"),
    ("source-grounding", "source grounding", "grounded"),
],
    "rag-012": [
        ("qdrant",),
        ("postgresql",),
    ],
}


def normalize(text: str) -> str:
    """Normalize text for simple, deterministic comparisons."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def answer_matches_expected_concepts(case: dict, answer: str) -> bool:
    """Return whether an answer contains all required concept groups."""
    concept_groups = EXPECTED_CONCEPTS.get(case["id"])

    # Missing scoring rules must not silently fall back to exact matching.
    if not concept_groups:
        return False

    normalized_answer = normalize(answer)

    return all(
        any(normalize(term) in normalized_answer for term in alternatives)
        for alternatives in concept_groups
    )


def is_abstention(answer: str) -> bool:
    """Detect common explicit statements that the source lacks an answer."""
    normalized_answer = normalize(answer)
    return any(marker in normalized_answer for marker in UNANSWERED_MARKERS)


def evaluate_case(case: dict, user_id: int, limit: int) -> dict:
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
    }

    try:
        response = answer_question(
            question=case["question"],
            user_id=user_id,
            limit=limit,
        )

        answer = response.get("answer") or ""
        sources = response.get("sources") or []

        result["answer"] = answer
        result["sources"] = sources

        expected_source = case.get("expected_source")
        if expected_source:
            expected_name = normalize(expected_source)
            result["retrieval_hit"] = any(
                expected_name in normalize(source.get("filename") or "")
                for source in sources
            )

        if case["answerable"]:
            result["answer_correct"] = answer_matches_expected_concepts(
                case, answer
            )
        else:
            result["correct_abstention"] = is_abstention(answer)

    except Exception as exc:
        result["failure"] = True
        result["error"] = f"{type(exc).__name__}: {exc}"

    finally:
        result["latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )

    return result


def run_evaluation(user_id: int, limit: int = 5) -> dict:
    results = [
        evaluate_case(case, user_id, limit)
        for case in EVALUATION_DATASET
    ]

    total = len(results)
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    return {
        "evaluation_name": "enterprise-ai-rag-expanded",
        "user_id": user_id,
        "retrieval_limit": limit,
        "total_questions": total,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "retrieval_hit_rate": (
            round(
                sum(r["retrieval_hit"] for r in answerable)
                / len(answerable),
                4,
            )
            if answerable else None
        ),
        "answer_correctness": (
            round(
                sum(r["answer_correct"] for r in answerable)
                / len(answerable),
                4,
            )
            if answerable else None
        ),
        "failure_rate": (
            round(sum(r["failure"] for r in results) / total, 4)
            if total else None
        ),
        "correct_abstention_rate": (
            round(
                sum(r["correct_abstention"] for r in unanswerable)
                / len(unanswerable),
                4,
            )
            if unanswerable else None
        ),
        "average_latency_ms": (
            round(mean(r["latency_ms"] for r in results), 2)
            if results else None
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
    args = parser.parse_args()

    if args.user_id <= 0:
        parser.error("--user-id must be a positive integer.")
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20.")

    report = run_evaluation(args.user_id, args.limit)

    report_path = (
        Path(__file__).resolve().parents[2]
        / "rag_evaluation_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        key: value
        for key, value in report.items()
        if key != "cases"
    }
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed report saved to: {report_path}")

    print("\nPer-question results:")
    for case in report["cases"]:
        status = "ERROR" if case["failure"] else "OK"
        print(
            f"- {case['id']} [{status}] "
            f"retrieval_hit={case['retrieval_hit']} "
            f"answer_correct={case['answer_correct']} "
            f"correct_abstention={case['correct_abstention']} "
            f"latency_ms={case['latency_ms']}"
        )
        if case["error"]:
            print(f"  Error: {case['error']}")


if __name__ == "__main__":
    main()
