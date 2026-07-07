import uuid

import pytest

from app.database import SessionLocal
from app.models import Chunk, Document, User
from app.services.retriever import search_similar_chunks


def _make_embedding(base_value: float, offset_index: int | None = None, offset_value: float = 0.0) -> list[float]:
    embedding = [base_value] * 384
    if offset_index is not None:
        embedding[offset_index] = base_value + offset_value
    return embedding


def test_search_similar_chunks_returns_closest_matches():
    db = SessionLocal()
    document = None
    user = None
    created_chunks: list[Chunk] = []

    try:
        user = User(
            email=f"test-{uuid.uuid4()}@example.com",
            hashed_password="test_hash",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        document = Document(
            filename="retrieval-test.txt",
            file_path="storage/uploads/retrieval-test.txt",
            content_type="text/plain",
            status="ready",
            user_id=user.id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        query_embedding = _make_embedding(1.0, offset_index=0, offset_value=0.001)
        nearest_chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            page_number=1,
            text="Nearest chunk",
            embedding=_make_embedding(1.0),
        )
        second_chunk = Chunk(
            document_id=document.id,
            chunk_index=1,
            page_number=1,
            text="Second closest chunk",
            embedding=_make_embedding(0.99),
        )
        third_chunk = Chunk(
            document_id=document.id,
            chunk_index=2,
            page_number=1,
            text="Far chunk",
            embedding=_make_embedding(0.1),
        )
        fourth_chunk = Chunk(
            document_id=document.id,
            chunk_index=3,
            page_number=1,
            text="Farthest chunk",
            embedding=_make_embedding(-1.0),
        )

        created_chunks.extend([nearest_chunk, second_chunk, third_chunk, fourth_chunk])
        db.add_all(created_chunks)
        db.commit()
        for chunk in created_chunks:
            db.refresh(chunk)

        results = search_similar_chunks(query_embedding=query_embedding, db=db, top_k=2)

        assert len(results) == 2
        assert results[0].id == nearest_chunk.id
        assert nearest_chunk.id in {chunk.id for chunk in results}
    finally:
        for chunk in created_chunks:
            db.delete(chunk)
        if document is not None:
            db.delete(document)
        if user is not None:
            db.delete(user)
        db.commit()
        db.close()
