from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers.auth import router as auth_router
from app.routers.chat import router as chat_router
from app.routers.documents import router as documents_router
from app.routers.messages import router as messages_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(messages_router)
app.include_router(auth_router)


@app.get("/health")
def health():
    return {"status": "ok"}
