from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(documents_router)


@app.get("/health")
def health():
    return {"status": "ok"}
