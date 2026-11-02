
from app.services.document_chunker import (
    MAX_CHUNK_TOKENS,
    _token_count,
    split_text,
)


def test_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_remains_one_chunk():
    text = "The Enterprise AI Agent uses Qdrant."
    chunks = split_text(text)

    assert chunks == [text]


def test_long_text_stays_within_token_limit():
    text = (
        "The Enterprise AI Knowledge Agent uses FastAPI, "
        "PostgreSQL, Qdrant, and Groq. "
    ) * 300

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(
        _token_count(chunk) <= MAX_CHUNK_TOKENS
        for chunk in chunks
    )


def test_all_original_words_are_preserved():
    text = (
        "Document processing uses chunking, embeddings, "
        "vector search, and grounded answers. "
    ) * 100

    chunks = split_text(text)

    reconstructed_words = " ".join(chunks).split()
    original_words = text.split()

    # Overlap may duplicate words, but no original word should be lost.
    position = 0

    for word in reconstructed_words:
        if position < len(original_words) and word == original_words[position]:
            position += 1

    assert position == len(original_words)


def test_invalid_chunk_parameters_raise_errors():
    import pytest

    with pytest.raises(ValueError):
        split_text("Some text", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("Some text", chunk_size=100, chunk_overlap=100)
