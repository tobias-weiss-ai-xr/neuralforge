"""Tests for forge.core.tei_client — embeddings, reranking, and error paths."""

import json

import httpx
import pytest

from forge.core import tei_client


class FakeResponse:
    def __init__(
        self, status_code: int = 200, data: dict | None = None, text: str = ""
    ):
        self.status_code = status_code
        self._data = data or {}
        self.text = text or json.dumps(self._data)

    def raise_for_status(self):
        if self.status_code >= 400:
            resp = httpx.Response(
                self.status_code, request=httpx.Request("POST", "http://fake")
            )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=resp.request, response=resp
            )

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict]] = []
        self.is_closed = False
        self._call_idx = 0

    async def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return FakeResponse(500, text="No canned response")

    async def aclose(self):
        self.is_closed = True


def _make_embed_response(embeddings: list[list[float]]) -> dict:
    return {"embedding": embeddings, "tokens": [len(e) for e in embeddings]}


def _make_rerank_response(scores: list[float]) -> dict:
    return {"results": [{"text": f"doc{i}", "score": s} for i, s in enumerate(scores)]}


@pytest.fixture(autouse=True)
def _reset_client():
    tei_client._client = None
    yield
    tei_client._client = None


@pytest.fixture()
def fake_client(monkeypatch):
    client = FakeAsyncClient()
    monkeypatch.setattr(tei_client, "_client", client)

    async def _patched():
        return client

    monkeypatch.setattr(tei_client, "_get_client", _patched)
    return client


class TestInferEmbedding:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, fake_client):
        result = await tei_client.infer_embedding([])
        assert result == []
        assert len(fake_client.calls) == 0

    @pytest.mark.asyncio
    async def test_single_text_embedding(self, fake_client):
        fake_client.responses = [
            FakeResponse(200, _make_embed_response([[0.1, 0.2, 0.3]]))
        ]
        result = await tei_client.infer_embedding(["hello"])
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_batch_embedding(self, fake_client):
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        fake_client.responses = [FakeResponse(200, _make_embed_response(vecs))]
        result = await tei_client.infer_embedding(["a", "b", "c"])
        assert result == vecs

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, fake_client):
        fake_client.responses = [FakeResponse(500, text="Internal Server Error")]
        result = await tei_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, monkeypatch):
        async def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        client = FakeAsyncClient()
        client.post = _raise_timeout
        monkeypatch.setattr(tei_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(tei_client, "_get_client", _patched)

        result = await tei_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_generic_exception_returns_none(self, monkeypatch):
        async def _raise(*args, **kwargs):
            raise RuntimeError("kaboom")

        client = FakeAsyncClient()
        client.post = _raise
        monkeypatch.setattr(tei_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(tei_client, "_get_client", _patched)

        result = await tei_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_url_uses_config(self, fake_client, monkeypatch):
        monkeypatch.setattr(tei_client, "TEI_EMBED_URL", "http://gpu:9000")
        fake_client.responses = [FakeResponse(200, _make_embed_response([[1.0]]))]
        await tei_client.infer_embedding(["x"])
        called_url = fake_client.calls[0][0]
        assert called_url == "http://gpu:9000/embed"

    @pytest.mark.asyncio
    async def test_payload_shape(self, fake_client):
        fake_client.responses = [FakeResponse(200, _make_embed_response([[0.1]]))]
        await tei_client.infer_embedding(["hello"])
        payload = fake_client.calls[0][1]["json"]
        assert payload == {"inputs": ["hello"]}


class TestInferRerank:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_empty_list(self, fake_client):
        result = await tei_client.infer_rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self, fake_client):
        result = await tei_client.infer_rerank("", ["doc1"])
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_rerank(self, fake_client):
        scores = [0.95, 0.42, 0.73]
        fake_client.responses = [FakeResponse(200, _make_rerank_response(scores))]
        result = await tei_client.infer_rerank("query", ["a", "b", "c"])
        assert result == scores

    @pytest.mark.asyncio
    async def test_rerank_http_error(self, fake_client):
        fake_client.responses = [FakeResponse(503, text="Service Unavailable")]
        result = await tei_client.infer_rerank("query", ["a"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_timeout(self, monkeypatch):
        async def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        client = FakeAsyncClient()
        client.post = _raise_timeout
        monkeypatch.setattr(tei_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(tei_client, "_get_client", _patched)

        result = await tei_client.infer_rerank("query", ["doc"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_missing_results(self, fake_client):
        fake_client.responses = [FakeResponse(200, {"results": []})]
        result = await tei_client.infer_rerank("query", ["doc"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_url_uses_config(self, fake_client, monkeypatch):
        monkeypatch.setattr(tei_client, "TEI_RERANK_URL", "http://rerank-box:7000")
        fake_client.responses = [FakeResponse(200, _make_rerank_response([0.5]))]
        await tei_client.infer_rerank("q", ["d"])
        called_url = fake_client.calls[0][0]
        assert called_url == "http://rerank-box:7000/rerank"

    @pytest.mark.asyncio
    async def test_rerank_payload_shape(self, fake_client):
        fake_client.responses = [FakeResponse(200, _make_rerank_response([0.9]))]
        await tei_client.infer_rerank("query", ["doc"])
        payload = fake_client.calls[0][1]["json"]
        assert payload["query"] == "query"
        assert payload["texts"] == ["doc"]
        assert payload["raw_scores"] is False


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_client(self):
        tei_client._client = FakeAsyncClient()
        await tei_client.close_client()
        assert tei_client._client is None
