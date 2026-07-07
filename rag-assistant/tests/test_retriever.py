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

        results = search_similar_chunks(query_embedding=query_embedding, user_id=user.id, db=db, top_k=2)

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


def test_search_similar_chunks_filters_by_user_id():
    db = SessionLocal()
    user_a = None
    user_b = None
    document_a = None
    document_b = None
    created_chunks: list[Chunk] = []

    try:
        user_a = User(
            email=f"user-a-{uuid.uuid4()}@example.com",
            hashed_password="test_hash",
        )
        user_b = User(
            email=f"user-b-{uuid.uuid4()}@example.com",
            hashed_password="test_hash",
        )
        db.add_all([user_a, user_b])
        db.commit()
        db.refresh(user_a)
        db.refresh(user_b)

        document_a = Document(
            filename="user-a-doc.txt",
            file_path="storage/uploads/user-a-doc.txt",
            content_type="text/plain",
            status="ready",
            user_id=user_a.id,
        )
        document_b = Document(
            filename="user-b-doc.txt",
            file_path="storage/uploads/user-b-doc.txt",
            content_type="text/plain",
            status="ready",
            user_id=user_b.id,
        )
        db.add_all([document_a, document_b])
        db.commit()
        db.refresh(document_a)
        db.refresh(document_b)

        query_embedding = _make_embedding(1.0)
        chunk_a = Chunk(
            document_id=document_a.id,
            chunk_index=0,
            page_number=1,
            text="Shared semantic match for user A",
            embedding=_make_embedding(1.0),
        )
        chunk_b = Chunk(
            document_id=document_b.id,
            chunk_index=0,
            page_number=1,
            text="Shared semantic match for user B",
            embedding=_make_embedding(1.0),
        )

        created_chunks.extend([chunk_a, chunk_b])
        db.add_all(created_chunks)
        db.commit()
        for chunk in created_chunks:
            db.refresh(chunk)

        results = search_similar_chunks(query_embedding=query_embedding, db=db, user_id=user_a.id, top_k=10)

        result_ids = {chunk.id for chunk in results}
        assert chunk_a.id in result_ids
        assert chunk_b.id not in result_ids
        assert all(chunk.document_id == document_a.id for chunk in results)
    finally:
        for chunk in created_chunks:
            db.delete(chunk)
        if document_a is not None:
            db.delete(document_a)
        if document_b is not None:
            db.delete(document_b)
        if user_a is not None:
            db.delete(user_a)
        if user_b is not None:
            db.delete(user_b)
        db.commit()
        db.close()
