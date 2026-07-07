from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Message


class FeedbackRequest(BaseModel):
    feedback: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    message_id: UUID
    feedback: str


router = APIRouter(prefix="/messages", tags=["messages"])


@router.post("/{message_id}/feedback")
def set_message_feedback(message_id: UUID, request: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:
    message = db.get(Message, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")

    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Only assistant messages can receive feedback")

    message.feedback = request.feedback
    db.commit()

    return FeedbackResponse(message_id=message_id, feedback=message.feedback)
