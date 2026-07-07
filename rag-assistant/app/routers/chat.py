from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Chunk, Conversation, Message
from app.services.embedder import embed_text
from app.services.llm import generate_answer
from app.services.reranker import rerank_chunks
from app.services.retriever import search_similar_chunks


class ChatRequest(BaseModel):
    question: str
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: UUID
    sources: list[dict]


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    if request.conversation_id is None:
        conversation = Conversation()
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        conversation_id = conversation.id
    else:
        conversation = db.get(Conversation, request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = conversation.id

    current_max_sequence_index = db.query(func.max(Message.sequence_index)).filter(Message.conversation_id == conversation_id).scalar()
    user_sequence_index = 0 if current_max_sequence_index is None else current_max_sequence_index + 1
    assistant_sequence_index = user_sequence_index + 1

    db.add(
        Message(
            conversation_id=conversation_id,
            role="user",
            content=request.question,
            sequence_index=user_sequence_index,
        )
    )
    db.commit()

    history = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id, Message.sequence_index < user_sequence_index)
        .order_by(Message.sequence_index.desc())
        .limit(6)
        .all()
    )
    history.reverse()

    query_embedding = embed_text(request.question)
    context_chunks = search_similar_chunks(query_embedding=query_embedding, db=db, top_k=20)

    if not context_chunks:
        answer = "Không tìm thấy thông tin liên quan trong tài liệu đã upload."
        db.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sequence_index=assistant_sequence_index,
            )
        )
        db.commit()
        return ChatResponse(answer=answer, conversation_id=conversation_id, sources=[])

    context_chunks = rerank_chunks(query=request.question, chunks=context_chunks, top_k=5)
    answer = generate_answer(question=request.question, context_chunks=context_chunks, history=history)
    sources = [
        {
            "document_id": str(chunk.document_id),
            "page_number": chunk.page_number,
            "text": chunk.text,
        }
        for chunk in context_chunks
    ]

    db.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            sequence_index=assistant_sequence_index,
        )
    )
    db.commit()

    return ChatResponse(answer=answer, conversation_id=conversation_id, sources=sources)

