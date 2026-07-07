import uuid

from sqlalchemy.orm import Session

from app.models import Chunk, Document


def search_similar_chunks(
    query_embedding: list[float], user_id: uuid.UUID, db: Session, top_k: int = 5
) -> list[Chunk]:
    distance = Chunk.embedding.cosine_distance(query_embedding)

    return (
        db.query(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.user_id == user_id)
        .filter(Chunk.embedding.isnot(None))
        .order_by(distance.asc())
        .limit(top_k)
        .all()
    )
