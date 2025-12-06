Multi-User Retrieval-Augmented Generation (RAG) System

A production-grade, multi-user RAG system built with FastAPI, LlamaIndex, Milvus (vector database), and PostgreSQL (metadata store).
The system supports independent user workspaces, document-level metadata, incremental indexing, and filtered semantic retrieval.
Ideal for personal AI assistants, knowledge-base agents, enterprise RAG, and research prototypes.

⸻

## Key Features

	•	Multi-user architecture with isolated document stores
	•	Metadata-enriched embeddings stored in Milvus / Milvus Lite
	•	PostgreSQL-backed document catalog, tracking user ownership and embedding status
	•	Incremental indexing (no re-embedding of old documents)
	•	User-based vector filtering using metadata filters
	•	FastAPI backend with clear endpoints for upload, query, and user management
	•	Simple HTML UI supporting user selection, document upload, and querying
	•	File-system persistence for both Milvus and LlamaIndex index structures

⸻

## System Architecture

### flowchart TD

    UI[Web UI] --> API[FastAPI Backend]
    API --> Users[User Management]
    API --> Docs[Document Upload]
    API --> Query[Query Endpoint]

    Docs --> PG[(PostgreSQL)]
    Users --> PG

    PG --> RagEngine[RAG Engine]

    RagEngine --> Milvus[(Milvus Vector DB)]
    RagEngine --> LlamaIndex[Index Storage]

    Query --> RagEngine


⸻

## Technology Stack

### Backend

	•	FastAPI
	•	Python 3.10
	•	LlamaIndex (VectorStoreIndex, metadata filters)
	•	Milvus / Milvus Lite (local or server mode)
	•	HuggingFace Embeddings
	•	HuggingFace LLM inference

### Database Layer

	•	PostgreSQL
	•	SQLAlchemy ORM
	•	session-based user isolation

### Vector Search

	•	Milvus Vector Database
	•	Metadata filters: user_id, doc_id
	•	Persistent index storage via LlamaIndex

⸻

### Core Concepts

## 1. User-Level Isolation

Each user gets:

	•	Their own upload directory: data/<user_id>/
	•	Metadata saved in PostgreSQL
	•	Vector embeddings filtered by metadata:
ExactMatchFilter(key="user_id", value="<id>")

No user can retrieve another user’s documents.

⸻

## 2. Document Lifecycle


| Step        | Action                                | Stored In            |
|-------------|----------------------------------------|-----------------------|
| Upload      | File saved under user's folder         | File System          |
| Metadata    | filename, user_id, embedded flag       | PostgreSQL           |
| Embedding   | Extract → vectorize → store vectors    | Milvus Vector Store  |
| Retrieval   | Filter by user metadata (user_id)      | Milvus + LlamaIndex  |


⸻


## Project Structure

```
final_chatbot/
│
├── api.py            # FastAPI endpoints (users, upload, query)
├── rag_engine.py     # Multi-user RAG engine (Milvus + LlamaIndex)
├── db.py             # PostgreSQL connection + session helper
├── models.py         # SQLAlchemy models (User, DocumentRecord)
├── config.py         # Configuration (paths, Milvus, LLM, embeddings)
│
├── static/
│   └── ui.html       # Multi-user UI
│
├── data/             # Auto-created user folders + uploaded documents
│
├── storage/          # LlamaIndex persistence (vector_store + docstore)
│
├── milvus/           # Milvus Lite vector database
│
└── requirements.txt  # Python dependencies
```


⸻

## API Endpoints

| Endpoint                               | Method | Description                                      |
|----------------------------------------|--------|--------------------------------------------------|
| /users                                 | POST   | Create a new user                                |
| /users                                 | GET    | List all users                                   |
| /users/{user_id}/upload-document       | POST   | Upload a TXT/PDF document for a specific user    |
| /users/{user_id}/db-documents          | GET    | List all documents stored in Postgres for user   |
| /users/{user_id}/vector-sources        | GET    | List all embedded document sources for user      |
| /users/{user_id}/query                 | POST   | Run a RAG query scoped to a single user          |
| /data-files                            | GET    | List all files in the filesystem                 |
| /health                                | GET    | Health check                                     |


⸻

## Setup Instructions

1. Clone Repository
```
git clone <repo-url>
cd final_chatbot
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Set Up PostgreSQL
```
Create user and database:

CREATE USER raguser WITH PASSWORD 'ragpassword';
CREATE DATABASE ragdb OWNER raguser;
```
4. SSH Port Forwarding (if on HPC)

Backend:
```
ssh -L 8001:localhost:8000 <username>@hpc
```
PostgreSQL:
```
ssh -R 5432:localhost:5432 <username>@hpc
```
5. Start Backend
```
uvicorn api:app --host 0.0.0.0 --port 8000
```
6. Open UI
```
http://localhost:8001
```

⸻

## Milvus Metadata Strategy

Each document is embedded with:

metadata = {
    "user_id": <user_id>,
    "doc_id": <doc_id>,
    "source": filename,
    "filetype": "pdf/txt"
}

Query filtering uses:

ExactMatchFilter(key="user_id", value="<id>")


⸻

### Why This Architecture?

	•	Prevents cross-user data leakage
	•	Allows incremental indexing (only new files are embedded)
	•	Ensures efficient retrieval at scale
	•	Clean separation of concerns: metadata in Postgres, vectors in Milvus
	•	Can scale horizontally with a central Milvus server

⸻

### Future Enhancements

	•	JWT-based authentication
	•	Multi-tenant UI
	•	Docker Compose setup (FastAPI + Postgres + Milvus)
	•	Async Milvus writes
	•	Background workers for embedding large files

⸻
