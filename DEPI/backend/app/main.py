import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load env files before any code reads os.environ (Groq, Pinecone, etc.).
# Order: later files override earlier. Only files that exist are loaded.
_backend_root = Path(__file__).resolve().parent.parent
_repo_root = _backend_root.parent
for _env_path in (
    _repo_root / ".env",
    Path.cwd() / ".env",
    _backend_root / ".env",
    _backend_root / ".env.local",
):
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import Base, engine
from app.models.user import User  # noqa: F401
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router

app = FastAPI(title="MedCortex API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    if not os.getenv("GROQ_API_KEY"):
        logging.getLogger("uvicorn.error").warning(
            "GROQ_API_KEY is missing. Put it in backend/.env (copy from .env.example). "
            "Keys in .env.example are NOT loaded — the file must be named .env"
        )
    Base.metadata.create_all(bind=engine)


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(chat_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
