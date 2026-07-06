# RAG Knowledge Assistant

Sprint 1 của project RAG Knowledge Assistant đã hoàn tất nền tảng cơ bản:
upload tài liệu, tách chunk, và tạo embedding local cho file `.txt`.

## Chạy project

1. Khởi động PostgreSQL:
```bash
docker-compose up -d
```

2. Chạy API:
```bash
uvicorn app.main:app --reload
```

## Endpoints hiện có

- `POST /documents/upload`
- `GET /documents`
- `DELETE /documents/{id}`
- `POST /documents/{id}/process`
- `GET /health`

## Ghi chú hiện tại

- Hiện tại chỉ hỗ trợ xử lý file `.txt`.
- PDF/DOCX sẽ làm ở giai đoạn sau.
- Embedding đang dùng model local `BAAI/bge-small-en-v1.5`.
- Model này có `384` dimensions.
- Sprint hiện tại không dùng API key cho embedding.
- Chưa có Alembic migration.
- Môi trường dev đang dùng `Base.metadata.create_all()`.
- Khi đổi schema trong dev, cần drop database thủ công rồi tạo lại để schema được cập nhật đầy đủ.

