
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Convert one text into a normalized embedding vector."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    model = get_embedding_model()

    vector = model.encode(
        text.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple texts into normalized embedding vectors."""
    if not texts:
        return []

    if any(not text or not text.strip() for text in texts):
        raise ValueError("Every text must be non-empty.")

    model = get_embedding_model()

    vectors = model.encode(
        [text.strip() for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return vectors.tolist()
