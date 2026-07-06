from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Chunk
from app.services.embedder import embed_text
from app.services.llm import generate_answer
from app.services.retriever import search_similar_chunks


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    query_embedding = embed_text(request.question)
    context_chunks = search_similar_chunks(query_embedding=query_embedding, db=db, top_k=5)

    if not context_chunks:
        return ChatResponse(
            answer="Không tìm thấy thông tin liên quan trong tài liệu đã upload.",
            sources=[],
        )

    answer = generate_answer(question=request.question, context_chunks=context_chunks)
    sources = [
        {
            "document_id": str(chunk.document_id),
            "page_number": chunk.page_number,
            "text": chunk.text,
        }
        for chunk in context_chunks
    ]

    return ChatResponse(answer=answer, sources=sources)
