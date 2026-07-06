from sqlalchemy.orm import Session

from app.models import Chunk


def search_similar_chunks(query_embedding: list[float], db: Session, top_k: int = 5) -> list[Chunk]:
    distance = Chunk.embedding.cosine_distance(query_embedding)

    return (
        db.query(Chunk)
        .filter(Chunk.embedding.isnot(None))
        .order_by(distance.asc())
        .limit(top_k)
        .all()
    )
