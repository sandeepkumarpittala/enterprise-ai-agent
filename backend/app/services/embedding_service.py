
import math
import time

import httpx

from app.core.config import settings
from app.services.document_chunker import _token_count


VECTOR_SIZE = 384
MAX_INPUT_TOKENS = 480
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60.0
MAX_BATCH_SIZE = 100


def _validate_text(text: str) -> str:
    """Validate text before sending it to Cloudflare."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    text = text.strip()
    token_count = _token_count(text)

    if token_count > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Text exceeds the embedding input limit of "
            f"{MAX_INPUT_TOKENS} tokens. Received {token_count} tokens. "
            "Split the text into smaller chunks."
        )

    return text


def _normalize_vector(vector: list[float]) -> list[float]:
    """Normalize an embedding vector to unit length."""

    magnitude = math.sqrt(sum(value * value for value in vector))

    if magnitude == 0:
        raise ValueError("Cloudflare returned a zero-length embedding.")

    return [value / magnitude for value in vector]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings through Cloudflare Workers AI."""

    account_id = settings.cloudflare_account_id
    api_token = settings.cloudflare_api_token
    model_name = settings.cloudflare_embedding_model

    if not account_id or not api_token:
        raise RuntimeError("Cloudflare credentials are not configured.")

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model_name}"
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    all_vectors = []

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for batch_start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[batch_start:batch_start + MAX_BATCH_SIZE]

            for attempt in range(MAX_RETRIES):
                try:
                    response = client.post(
                        url,
                        headers=headers,
                        json={"text": batch},
                    )
                    response.raise_for_status()

                    result = response.json()

                    if not result.get("success"):
                        raise RuntimeError(
                            f"Cloudflare AI request failed: "
                            f"{result.get('errors')}"
                        )

                    vectors = result["result"]["data"]

                    if len(vectors) != len(batch):
                        raise RuntimeError(
                            "Unexpected number of returned embeddings."
                        )

                    batch_vectors = []

                    for vector in vectors:
                        if len(vector) != VECTOR_SIZE:
                            raise RuntimeError(
                                f"Expected {VECTOR_SIZE} dimensions, "
                                f"received {len(vector)}."
                            )

                        batch_vectors.append(_normalize_vector(vector))

                    all_vectors.extend(batch_vectors)
                    break

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code

                    if status < 500 and status != 429:
                        raise RuntimeError(
                            f"Cloudflare rejected the request: HTTP {status}"
                        ) from exc

                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(
                            "Cloudflare API failed after retries."
                        ) from exc

                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(
                            "Cloudflare API connection failed after retries."
                        ) from exc

                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

    return all_vectors


def embed_text(text: str) -> list[float]:
    """Convert one validated text into a normalized embedding vector."""

    validated_text = _validate_text(text)

    return _embed_batch([validated_text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple validated texts into normalized embedding vectors."""

    if not texts:
        return []

    validated_texts = [_validate_text(text) for text in texts]

    return _embed_batch(validated_texts)
