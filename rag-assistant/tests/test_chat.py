import uuid

from app.database import SessionLocal
from app.main import app
from app.models import Chunk, Conversation, Document, Message
from app.models import User
from fastapi.testclient import TestClient
from unittest.mock import patch


client = TestClient(app)


def _make_embedding(base_value: float, offset_index: int | None = None, offset_value: float = 0.0) -> list[float]:
    embedding = [base_value] * 384
    if offset_index is not None:
        embedding[offset_index] = base_value + offset_value
    return embedding


def test_chat_with_no_matching_chunks_returns_fallback_answer(auth_headers):
    with patch("app.routers.chat.search_similar_chunks", return_value=[]) as mock_search, patch(
        "app.routers.chat.generate_answer"
    ) as mock_generate:
        response = client.post("/chat", json={"question": "Câu hỏi không có kết quả"}, headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == []
    assert "Không tìm thấy thông tin liên quan" in payload["answer"]
    mock_generate.assert_not_called()
    mock_search.assert_called_once()


def test_chat_with_matching_chunks_returns_answer_with_sources(auth_headers, auth_user_id):
    db = SessionLocal()
    document = None
    created_chunks: list[Chunk] = []

    try:
        document = Document(
            filename="chat-test.txt",
            file_path="storage/uploads/chat-test.txt",
            content_type="text/plain",
            status="ready",
            user_id=auth_user_id,
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
            response = client.post("/chat", json={"question": "Câu hỏi test"}, headers=auth_headers)

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


def test_chat_persists_messages_for_conversation_sequence(auth_headers, auth_user_id):
    db = SessionLocal()
    conversation = None

    try:
        conversation = Conversation(user_id=auth_user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        conversation_id = conversation.id

        with patch("app.routers.chat.search_similar_chunks", return_value=[]), patch(
            "app.routers.chat.embed_text", return_value=[0.0] * 384
        ) as mock_embed, patch("app.routers.chat.generate_answer") as mock_generate:
            first_response = client.post(
                "/chat",
                json={"question": "câu hỏi đầu tiên", "conversation_id": str(conversation_id)},
                headers=auth_headers,
            )
            second_response = client.post(
                "/chat",
                json={"question": "câu hỏi thứ hai", "conversation_id": str(conversation_id)},
                headers=auth_headers,
            )

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert mock_embed.call_count == 2
        mock_generate.assert_not_called()

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_index.asc())
            .all()
        )
        assert len(messages) == 4
        assert [message.sequence_index for message in messages] == [0, 1, 2, 3]
        assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]
        assert [message.content for message in messages] == [
            "câu hỏi đầu tiên",
            "Không tìm thấy thông tin liên quan trong tài liệu đã upload.",
            "câu hỏi thứ hai",
            "Không tìm thấy thông tin liên quan trong tài liệu đã upload.",
        ]
    finally:
        if conversation is not None:
            db.query(Message).filter(Message.conversation_id == conversation.id).delete()
            db.delete(conversation)
            db.commit()
        db.close()


def test_chat_passes_recent_history_to_generate_answer(auth_headers, auth_user_id):
    db = SessionLocal()
    conversation = None
    document = None
    created_chunks: list[Chunk] = []
    try:
        conversation = Conversation(user_id=auth_user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        prior_user = Message(
            conversation_id=conversation.id,
            role="user",
            content="câu hỏi cũ",
            sequence_index=0,
        )
        prior_assistant = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="câu trả lời cũ",
            sequence_index=1,
        )
        db.add_all([prior_user, prior_assistant])
        db.commit()
        db.refresh(prior_user)
        db.refresh(prior_assistant)

        document = Document(
            filename="history-test.txt",
            file_path="storage/uploads/history-test.txt",
            content_type="text/plain",
            status="ready",
            user_id=auth_user_id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            page_number=None,
            text="nội dung chunk giả",
            embedding=_make_embedding(1.0),
        )
        created_chunks.append(chunk)
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        captured_history = []

        def capture_generate_answer(*args, **kwargs):
            history = kwargs["history"]
            captured_history.append(
                [
                    (message.sequence_index, message.role, message.content)
                    for message in history
                ]
            )
            return "câu trả lời mới"

        with patch("app.routers.chat.embed_text", return_value=[0.0] * 384), patch(
            "app.routers.chat.search_similar_chunks", return_value=[chunk]
        ), patch("app.routers.chat.rerank_chunks", return_value=[chunk]), patch(
            "app.routers.chat.generate_answer", side_effect=capture_generate_answer
        ) as mock_generate:
            response = client.post(
                "/chat",
                json={"question": "câu hỏi mới", "conversation_id": str(conversation.id)},
                headers=auth_headers,
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "câu trả lời mới"
        assert payload["conversation_id"] == str(conversation.id)

        mock_generate.assert_called_once()
        assert captured_history == [[(0, "user", "câu hỏi cũ"), (1, "assistant", "câu trả lời cũ")]]
    finally:
        for chunk in created_chunks:
            db.delete(chunk)
        if conversation is not None:
            db.query(Message).filter(Message.conversation_id == conversation.id).delete()
            db.delete(conversation)
        if document is not None:
            db.delete(document)
        db.commit()
        db.close()


def test_user_cannot_access_another_users_conversation(auth_headers):
    email_b = f"test-{uuid.uuid4()}@example.com"
    password = "test_password_123"

    client.post("/auth/register", json={"email": email_b, "password": password})
    token_b = client.post("/auth/login", json={"email": email_b, "password": password}).json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    with patch("app.routers.chat.search_similar_chunks", return_value=[]), patch(
        "app.routers.chat.generate_answer"
    ):
        create_response = client.post("/chat", json={"question": "câu hỏi của user A"}, headers=auth_headers)

    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]

    with patch("app.routers.chat.search_similar_chunks", return_value=[]), patch(
        "app.routers.chat.generate_answer"
    ):
        response = client.post(
            "/chat",
            json={"question": "câu hỏi của user B", "conversation_id": conversation_id},
            headers=headers_b,
        )

    assert response.status_code == 404
