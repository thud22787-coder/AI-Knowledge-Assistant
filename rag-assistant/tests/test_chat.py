from app.database import SessionLocal
from app.main import app
from app.models import Chunk, Document
from fastapi.testclient import TestClient
from unittest.mock import patch


client = TestClient(app)


def _make_embedding(base_value: float, offset_index: int | None = None, offset_value: float = 0.0) -> list[float]:
    embedding = [base_value] * 384
    if offset_index is not None:
        embedding[offset_index] = base_value + offset_value
    return embedding


def test_chat_with_no_matching_chunks_returns_fallback_answer():
    with patch("app.routers.chat.search_similar_chunks", return_value=[]) as mock_search, patch(
        "app.routers.chat.generate_answer"
    ) as mock_generate:
        response = client.post("/chat", json={"question": "Câu hỏi không có kết quả"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == []
    assert "Không tìm thấy thông tin liên quan" in payload["answer"]
    mock_generate.assert_not_called()
    mock_search.assert_called_once()


def test_chat_with_matching_chunks_returns_answer_with_sources():
    db = SessionLocal()
    document = None
    created_chunks: list[Chunk] = []

    try:
        document = Document(
            filename="chat-test.txt",
            file_path="storage/uploads/chat-test.txt",
            content_type="text/plain",
            status="ready",
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        first_chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            page_number=2,
            text="Nội dung chunk gần nhất",
            embedding=_make_embedding(1.0),
        )
        second_chunk = Chunk(
            document_id=document.id,
            chunk_index=1,
            page_number=None,
            text="Nội dung chunk phụ",
            embedding=_make_embedding(0.95),
        )

        created_chunks.extend([first_chunk, second_chunk])
        db.add_all(created_chunks)
        db.commit()
        for chunk in created_chunks:
            db.refresh(chunk)

        with patch("app.routers.chat.generate_answer", return_value="câu trả lời giả lập") as mock_generate, patch(
            "app.routers.chat.rerank_chunks", side_effect=lambda query, chunks, top_k: chunks
        ) as mock_rerank:
            response = client.post("/chat", json={"question": "Câu hỏi test"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "câu trả lời giả lập"
        assert len(payload["sources"]) >= 1
        for source in payload["sources"]:
            assert "document_id" in source
            assert "page_number" in source
            assert "text" in source
        mock_generate.assert_called_once()
        mock_rerank.assert_called_once()
    finally:
        for chunk in created_chunks:
            db.delete(chunk)
        if document is not None:
            db.delete(document)
        db.commit()
        db.close()
