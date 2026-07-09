from types import SimpleNamespace

import pytest

from app.models import Chunk


def test_rerank_chunks_orders_by_score_and_limits_to_top_k(monkeypatch):
    from app.services.reranker import rerank_chunks

    chunks = []
    for text in ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]:
        chunk = Chunk()
        chunk.text = text
        chunks.append(chunk)

    monkeypatch.setattr(
        "app.services.reranker._reranker",
        SimpleNamespace(predict=lambda pairs: [0.1, 0.4, 0.9, 0.2]),
    )
    result = rerank_chunks(query="test query", chunks=chunks, top_k=2)

    assert len(result) == 2
    assert result[0].text == "chunk 3"
    assert result[1].text == "chunk 2"


def test_rerank_chunks_with_empty_list_returns_empty_list():
    from unittest.mock import patch

    with patch("sentence_transformers.CrossEncoder"):
        from app.services.reranker import rerank_chunks

        result = rerank_chunks(query="test", chunks=[], top_k=5)

    assert result == []


@pytest.mark.xfail(
    reason="bge-reranker-base xếp sai với câu rất ngắn/gần giống nhau ở case tiếng Việt; rerank_chunks đã verify đúng logic và mock leak test đã được sửa bằng monkeypatch",
    strict=True,
)
def test_rerank_chunks_prioritizes_relevant_english_chunk_for_vietnamese_query():
    from app.services.reranker import rerank_chunks

    query = "Trong sample PDF, câu ở trang 1 là gì?"
    chunk_texts = [
        "This is a PDF document for processing.",
        "This is the first DOCX paragraph for processing. It should be parsed into text and chunked.",
        "DOCX paragraph one: alpha beta gamma.",
        "Paragraph one has enough text to make the parser and chunker work properly. It should become at least one chunk.",
        "Page two: pack my box with five dozen liquor jugs.",
        "Page one: the quick brown fox jumps over the lazy dog.",
    ]

    chunks = []
    for text in chunk_texts:
        chunk = Chunk()
        chunk.text = text
        chunks.append(chunk)

    result = rerank_chunks(query=query, chunks=chunks, top_k=3)

    assert result[0].text == "Page one: the quick brown fox jumps over the lazy dog."


@pytest.mark.xfail(
    reason="bge-reranker-base xếp sai với câu rất ngắn/gần giống nhau ngay cả cùng ngôn ngữ; chưa xác nhận có tái diễn với đoạn văn dài hơn thực tế",
    strict=True,
)
def test_rerank_chunks_prioritizes_relevant_chunk_for_english_query():
    from app.services.reranker import rerank_chunks

    query = "What is on page one of the sample PDF?"
    chunk_texts = [
        "This is a PDF document for processing.",
        "This is the first DOCX paragraph for processing. It should be parsed into text and chunked.",
        "DOCX paragraph one: alpha beta gamma.",
        "Paragraph one has enough text to make the parser and chunker work properly. It should become at least one chunk.",
        "Page two: pack my box with five dozen liquor jugs.",
        "Page one: the quick brown fox jumps over the lazy dog.",
    ]

    chunks = []
    for text in chunk_texts:
        chunk = Chunk()
        chunk.text = text
        chunks.append(chunk)

    result = rerank_chunks(query=query, chunks=chunks, top_k=3)

    assert result[0].text == "Page one: the quick brown fox jumps over the lazy dog."
