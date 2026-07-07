from unittest.mock import patch
from types import SimpleNamespace

from app.models import Chunk


def test_rerank_chunks_orders_by_score_and_limits_to_top_k():
    with patch("sentence_transformers.CrossEncoder"):
        from app.services.reranker import rerank_chunks
        import app.services.reranker as reranker_module

        chunks = []
        for text in ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]:
            chunk = Chunk()
            chunk.text = text
            chunks.append(chunk)

        reranker_module._reranker = SimpleNamespace(predict=lambda pairs: [0.1, 0.4, 0.9, 0.2])
        with patch("app.services.reranker.get_reranker", return_value=reranker_module._reranker):
            result = rerank_chunks(query="test query", chunks=chunks, top_k=2)

    assert len(result) == 2
    assert result[0].text == "chunk 3"
    assert result[1].text == "chunk 2"


def test_rerank_chunks_with_empty_list_returns_empty_list():
    with patch("sentence_transformers.CrossEncoder"):
        from app.services.reranker import rerank_chunks

        result = rerank_chunks(query="test", chunks=[], top_k=5)

    assert result == []
