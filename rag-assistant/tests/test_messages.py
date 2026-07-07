from uuid import uuid4

from app.database import SessionLocal
from app.main import app
from app.models import Conversation, Message
from fastapi.testclient import TestClient


client = TestClient(app)


def test_feedback_success_persists_on_assistant_message(auth_user_id):
    db = SessionLocal()
    conversation = None
    message = None

    try:
        conversation = Conversation(user_id=auth_user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="answer text",
            sequence_index=0,
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        response = client.post(f"/messages/{message.id}/feedback", json={"feedback": "up"})

        assert response.status_code == 200
        payload = response.json()
        assert payload == {"message_id": str(message.id), "feedback": "up"}

        db.refresh(message)
        assert message.feedback == "up"
    finally:
        if message is not None:
            db.delete(message)
        if conversation is not None:
            db.delete(conversation)
        db.commit()
        db.close()


def test_feedback_returns_404_for_missing_message():
    response = client.post(f"/messages/{uuid4()}/feedback", json={"feedback": "up"})
    assert response.status_code == 404


def test_feedback_returns_400_for_user_message(auth_user_id):
    db = SessionLocal()
    conversation = None
    message = None

    try:
        conversation = Conversation(user_id=auth_user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        message = Message(
            conversation_id=conversation.id,
            role="user",
            content="user question",
            sequence_index=0,
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        response = client.post(f"/messages/{message.id}/feedback", json={"feedback": "down"})
        assert response.status_code == 400
    finally:
        if message is not None:
            db.delete(message)
        if conversation is not None:
            db.delete(conversation)
        db.commit()
        db.close()


def test_feedback_returns_422_for_invalid_feedback_value():
    response = client.post(f"/messages/{uuid4()}/feedback", json={"feedback": "maybe"})
    assert response.status_code == 422
