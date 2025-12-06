import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import config
from config import DATA_DIRECTORY
from rag_engine import RagEngine
from db import engine, SessionLocal, get_db
from models import Base, User, DocumentRecord




print("Creating database tables if not exist...")
Base.metadata.create_all(bind=engine)


# -----------------------------
# FastAPI initialization
# -----------------------------

app = FastAPI(title="Multi-User LlamaIndex + Milvus RAG API")

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_file = Path("static/ui.html")
    if ui_file.exists():
        return FileResponse(str(ui_file))
    return HTMLResponse("<h2>UI not found in /static</h2>")


# -----------------------------
# Initialize Multi-user RAG Engine
# -----------------------------

rag_engine = RagEngine(
    collection_name="llamaindex_documents",
    persist_dir=config.PERSIST_DIR,
    rebuild_index=False,
)


# -----------------------------
# Pydantic Schemas
# -----------------------------

class UserCreate(BaseModel):
    username: str


class UserOut(BaseModel):
    id: int
    username: str
    created_at: Optional[str]


class Source(BaseModel):
    text: str
    source: str
    score: Optional[float]


class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]]


# -----------------------------
# Health endpoint
# -----------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# User Endpoints
# -----------------------------

@app.post("/users", response_model=UserOut)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")

    user = User(username=user_in.username)
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserOut(
        id=user.id,
        username=user.username,
        created_at=str(user.created_at),
    )


@app.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        UserOut(id=u.id, username=u.username, created_at=str(u.created_at))
        for u in users
    ]


# -----------------------------
# Document Listing (DB + Vector DB)
# -----------------------------

@app.get("/users/{user_id}/db-documents")
def list_user_documents(user_id: int, db: Session = Depends(get_db)):
    docs = db.query(DocumentRecord).filter(DocumentRecord.user_id == user_id).all()
    return [d.as_dict() for d in docs]


@app.get("/users/{user_id}/vector-sources")
def list_user_vector_sources(user_id: int):
    sources = rag_engine.list_sources(user_id=user_id)
    return {"user_id": user_id, "sources": sources}


# -----------------------------
# Upload a document for a specific user
# -----------------------------

@app.post("/users/{user_id}/upload-document")
async def upload_document_for_user(
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1. Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")

    allowed_exts = {".txt", ".pdf"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext}")

    # 2. Save file under per-user directory
    user_dir = Path(DATA_DIRECTORY) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    save_path = user_dir / file.filename

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 3. Insert DB record (embedded = False)
    try:
        doc = DocumentRecord(
            user_id=user_id,
            filename=file.filename,
            filepath=str(save_path),
            filetype=ext.lstrip(".").lower(),
            embedded=False,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    # 4. Insert into vector DB and embed
    try:
        rag_engine.add_document_from_path(
            file_path=str(save_path),
            user_id=user_id,
            doc_id=doc.id,
        )
        doc.embedded = True
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Vector index failed: {e}")

    return {
        "status": "ok",
        "message": f"Uploaded + indexed '{file.filename}' for user {user_id}.",
        "document": doc.as_dict(),
    }


# -----------------------------
# Query RAG for a specific user
# -----------------------------

@app.post("/users/{user_id}/query", response_model=QueryResponse)
async def query_for_user(user_id: int, req: QueryRequest):
    try:
        result = rag_engine.answer(
            user_id=user_id,
            question=req.question,
            include_sources=req.include_sources,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[
                Source(**s) for s in result.get("sources", [])
            ] if req.include_sources else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# -----------------------------
# Debug endpoint
# -----------------------------

@app.get("/data-files")
def list_data_files():
    root = Path(DATA_DIRECTORY)
    out = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}:
                out.append(str(p))
    return {"data_files": out}
