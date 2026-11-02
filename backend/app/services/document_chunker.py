
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache

from tokenizers import Tokenizer


TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
MAX_CHUNK_TOKENS = 480
MAX_CONTENT_TOKENS = MAX_CHUNK_TOKENS - 2
DEFAULT_CHUNK_OVERLAP = 150


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """Load and reuse the BGE tokenizer without loading its model."""
    return Tokenizer.from_pretrained(TOKENIZER_NAME)


def _token_count(text: str) -> int:
    """Count tokens including special tokens."""
    return len(
        get_tokenizer().encode(
            text,
            add_special_tokens=True,
        ).ids
    )


def _find_split_point(text: str, start: int, max_end: int) -> int:
    """Prefer natural text boundaries without creating tiny chunks."""

    window = text[start:max_end]
    minimum_length = int(len(window) * 0.60)

    paragraph_positions = [
        match.end()
        for match in re.finditer(r"\n\s*\n", window)
        if match.end() >= minimum_length
    ]

    if paragraph_positions:
        return start + paragraph_positions[-1]

    sentence_positions = [
        match.end()
        for match in re.finditer(r"""[.!?]["')\]]?\s+""", window)
        if match.end() >= minimum_length
    ]

    if sentence_positions:
        return start + sentence_positions[-1]

    whitespace_position = window.rfind(" ")

    if whitespace_position >= minimum_length:
        return start + whitespace_position + 1

    return max_end


def _find_token_safe_end(
    text: str,
    start: int,
    token_ends: list[int],
) -> int:
    """Find a natural chunk boundary using precomputed token offsets."""

    token_index = bisect_right(token_ends, start)

    if token_index >= len(token_ends):
        return len(text)

    last_allowed_index = token_index + MAX_CONTENT_TOKENS - 1

    if last_allowed_index >= len(token_ends):
        max_end = len(text)
    else:
        max_end = token_ends[last_allowed_index]

    if max_end >= len(text):
        return len(text)

    safe_end = _find_split_point(text, start, max_end)

    while safe_end > start:
        candidate = text[start:safe_end].strip()

        if candidate and _token_count(candidate) <= MAX_CHUNK_TOKENS:
            return safe_end

        previous_index = bisect_left(token_ends, safe_end) - 1

        if previous_index < token_index:
            break

        safe_end = token_ends[previous_index]

    raise ValueError(
        "Unable to create a token-safe chunk. Check the document text."
    )


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks within the BGE token limit."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    encoding = get_tokenizer().encode(
        text,
        add_special_tokens=False,
    )

    token_ends = [
        end
        for _, end in encoding.offsets
        if end > 0
    ]

    if not token_ends:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = _find_token_safe_end(text, start, token_ends)

        if end <= start:
            raise ValueError("Chunking did not make forward progress.")

        chunk = text[start:end].strip()

        if chunk:
            if _token_count(chunk) > MAX_CHUNK_TOKENS:
                raise ValueError("Generated chunk exceeds the token limit.")

            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(start + 1, end - chunk_overlap)

        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1

        start = next_start

    return chunks
