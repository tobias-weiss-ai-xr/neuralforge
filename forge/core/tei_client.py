"""HuggingFace TEI client for embeddings and reranking."""

import logging
from typing import Any

import httpx

from forge.config import TEI_EMBED_URL, TEI_RERANK_URL

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None

_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


async def infer_embedding(texts: list[str]) -> list[list[float]] | None:
    if not texts:
        return []

    client = await _get_client()
    url = f"{TEI_EMBED_URL}/embed"
    payload: dict[str, Any] = {"inputs": texts}

    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

        data = resp.json()
        embedding = data.get("embedding", [])

        if isinstance(embedding[0] if embedding else None, list):
            return embedding

        if len(texts) == 1 and isinstance(embedding, list):
            return [embedding]

        logger.error("Unexpected TEI embedding response shape")
        return None

    except httpx.HTTPStatusError as exc:
        logger.error(
            "TEI embedding HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return None
    except httpx.TimeoutException:
        logger.error("TEI embedding request timed out")
        return None
    except Exception:
        logger.exception("Unexpected error during TEI embedding inference")
        return None


async def infer_rerank(query: str, documents: list[str]) -> list[float] | None:
    if not documents:
        return []
    if not query:
        logger.warning("infer_rerank called with empty query")
        return None

    client = await _get_client()
    url = f"{TEI_RERANK_URL}/rerank"
    payload: dict[str, Any] = {
        "query": query,
        "texts": documents,
        "raw_scores": False,
    }

    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        if not results:
            logger.error("TEI rerank response missing 'results': %s", data)
            return None

        return [float(r["score"]) for r in results]

    except httpx.HTTPStatusError as exc:
        logger.error(
            "TEI rerank HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return None
    except httpx.TimeoutException:
        logger.error("TEI rerank request timed out")
        return None
    except Exception:
        logger.exception("Unexpected error during TEI rerank inference")
        return None


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
