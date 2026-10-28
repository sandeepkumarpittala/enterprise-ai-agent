
"""Deterministic baseline checks for RAG answer grounding.

This module uses lexical overlap as a heuristic. It does not establish
semantic entailment or prove that a claim is factually correct.
"""

import re
from typing import Any


def normalize_text(text: str) -> str:
    """Lowercase text and normalize whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()



def split_claims(answer: str) -> list[str]:
    """Split answers into sentence-like claims, ignoring list numbering."""
    if not isinstance(answer, str) or not answer.strip():
        return []

    cleaned = re.sub(
        r"(?m)^\s*(?:[-*•]\s+|\d+[.)]\s*)",
        "",
        answer.strip(),
    )

    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)

    return [
        part.strip()
        for part in parts
        if part.strip() and re.search(r"[A-Za-z]{2,}", part)
    ]



def content_words(text: str) -> set[str]:
    """Return normalized words, excluding very common English words."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "been",
        "being", "by", "for", "from", "has", "have", "in",
        "into", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "which",
        "with", "can", "could", "should", "would", "will",
        "your", "you", "we", "they", "them", "these", "those",
    }

    words = re.findall(r"\b[a-z0-9][a-z0-9_.+-]*\b", text.lower())

    return {
        word
        for word in words
        if word not in stop_words and len(word) > 1
    }


def claim_is_lexically_supported(
    claim: str,
    evidence_text: str,
    minimum_overlap: float = 0.5,
) -> bool:
    """Estimate support using the proportion of claim words in evidence.

    This is a lexical heuristic only. Matching words does not guarantee
    that the evidence entails the claim.
    """
    if not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be greater than 0 and at most 1.")

    claim_words = content_words(claim)
    evidence_words = content_words(evidence_text)

    if not claim_words or not evidence_words:
        return False

    overlap = len(claim_words & evidence_words) / len(claim_words)

    return overlap >= minimum_overlap


def evaluate_grounding(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    minimum_overlap: float = 0.5,
) -> dict[str, Any]:
    """Evaluate answer claims against retrieved chunk text."""
    claims = split_claims(answer)

    evidence_parts = [
        chunk.get("content", "")
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("content"), str)
        and chunk.get("content", "").strip()
    ]

    evidence_text = " ".join(evidence_parts)

    if not claims:
        return {
            "groundedness_score": None,
            "claims_checked": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "claim_results": [],
        }

    claim_results = []

    for claim in claims:
        supported = claim_is_lexically_supported(
            claim,
            evidence_text,
            minimum_overlap=minimum_overlap,
        )

        claim_results.append({
            "claim": claim,
            "lexically_supported": supported,
        })

    supported_count = sum(
        item["lexically_supported"]
        for item in claim_results
    )

    unsupported_claims = [
        item["claim"]
        for item in claim_results
        if not item["lexically_supported"]
    ]

    return {
        "groundedness_score": round(supported_count / len(claims), 4),
        "claims_checked": len(claims),
        "supported_claims": supported_count,
        "unsupported_claims": unsupported_claims,
        "claim_results": claim_results,
    }
