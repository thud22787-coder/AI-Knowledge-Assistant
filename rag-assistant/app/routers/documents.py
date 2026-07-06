import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Chunk, Document
from app.services.embedder import embed_texts
from app.services.chunker import chunk_text
from app.services.parser import parse_txt


router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = Path("storage/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf, .docx, .txt files are allowed")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    document_id = uuid.uuid4()
    safe_filename = f"{document_id}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_filename

    contents = await file.read()
    file_path.write_bytes(contents)

    document = Document(
        id=document_id,
        filename=file.filename,
        file_path=str(file_path),
        content_type=file.content_type,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return {"id": str(document.id), "filename": document.filename, "status": document.status}


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": str(document.id),
            "filename": document.filename,
            "status": document.status,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "content_type": document.content_type,
        }
        for document in documents
    ]


@router.delete("/{document_id}")
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(document.file_path)
    if file_path.exists():
        file_path.unlink()

    db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    db.delete(document)
    db.commit()

    return {"id": str(document_id), "deleted": True}


@router.post("/{document_id}/process")
def process_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if Path(document.filename).suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="Only .txt supported for now")

    text = parse_txt(document.file_path)
    chunks = chunk_text(text)
    embeddings = embed_texts([chunk_data["text"] for chunk_data in chunks]) if chunks else []

    db.query(Chunk).filter(Chunk.document_id == document_id).delete()

    for chunk_data, embedding in zip(chunks, embeddings):
        db.add(
            Chunk(
                document_id=document_id,
                chunk_index=chunk_data["chunk_index"],
                text=chunk_data["text"],
                embedding=embedding,
            )
        )

    document.status = "ready"
    db.commit()

    return {"document_id": str(document_id), "status": "ready", "chunk_count": len(chunks)}
