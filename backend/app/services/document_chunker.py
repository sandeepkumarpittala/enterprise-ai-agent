
import re


def _find_split_point(
    text: str,
    start: int,
    max_end: int,
) -> int:
    """Prefer natural text boundaries without creating tiny chunks."""

    window = text[start:max_end]
    minimum_length = int(len(window) * 0.60)

    # Prefer paragraph boundaries first.
    paragraph_positions = [
        match.end()
        for match in re.finditer(r"\n\s*\n", window)
        if match.end() >= minimum_length
    ]
    if paragraph_positions:
        return start + paragraph_positions[-1]

    # Prefer sentence boundaries next.
    sentence_positions = [
        match.end()
        for match in re.finditer(r'[.!?]["\')\]]?\s+', window)
        if match.end() >= minimum_length
    ]
    if sentence_positions:
        return start + sentence_positions[-1]

    # Fall back to a word boundary.
    whitespace_position = window.rfind(" ")
    if whitespace_position >= minimum_length:
        return start + whitespace_position + 1

    # No suitable boundary: split at the maximum allowed size.
    return max_end


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)

        if max_end == text_length:
            end = text_length
        else:
            end = _find_split_point(text, start, max_end)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Retain overlap, but avoid starting in the middle of a word.
        next_start = max(start + 1, end - chunk_overlap)

        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1

        start = next_start

    return chunks
