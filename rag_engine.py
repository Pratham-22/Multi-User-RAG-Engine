# rag_engine.py

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF for PDF text extraction
from sqlalchemy.orm import Session

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    load_index_from_storage,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

import config
from db import SessionLocal
from models import DocumentRecord


def _preview_text(text: str, max_len: int = 200) -> str:
    """
    Clean up text for preview in sources list.
    Removes weird binary chars and trims length.
    """
    cleaned = "".join(
        ch if (32 <= ord(ch) < 127) or ch in "\n\t " else " " for ch in text
    )
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    return cleaned


class RagEngine:


    def __init__(
        self,
        collection_name: str = "llamaindex_documents",
        persist_dir: Optional[str] = None,
        rebuild_index: bool = False,
    ):
        self.collection_name = collection_name
        # Make persist_dir absolute so it's stable across runs
        self.persist_dir = str(Path(persist_dir or config.PERSIST_DIR).resolve())
        self.rebuild_index = rebuild_index

        print(f"[RagEngine] Using persist_dir={self.persist_dir}")

        # 1) Embedding model
        print("[RagEngine] Loading embedding model...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            device=config.EMBEDDING_DEVICE,
        )

        # 2) LLM
        print("[RagEngine] Loading LLM...")
        self.llm = self._initialize_llm()

        # 3) Global LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP

        # 4) Vector index (Milvus + docstore)
        self.index = self._initialize_index()

        print("[RagEngine] Initialized successfully.")


    def _initialize_llm(self):
        if config.LLM_TYPE in ["llama", "olmo"]:
            return HuggingFaceLLM(
                model_name=config.LLM_MODEL_NAME,
                tokenizer_name=config.LLM_MODEL_NAME,
                context_window=config.LLM_CONTEXT_WINDOW,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                generate_kwargs={"temperature": config.LLM_TEMPERATURE},
                device_map=config.LLM_DEVICE_MAP,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {config.LLM_TYPE}")

    def _initialize_index(self) -> VectorStoreIndex:
        """
        Connect to Milvus and either load an existing index from disk
        or build a brand-new one from Postgres documents.
        """
        milvus_uri = config.MILVUS_URI

        # Milvus Lite (local sqlite-like file)
        if milvus_uri.endswith(".db"):
            milvus_path = Path(milvus_uri).resolve()
            milvus_path.parent.mkdir(parents=True, exist_ok=True)
            milvus_uri = str(milvus_path)
            print(f"[RagEngine] Using Milvus Lite at: {milvus_uri}")

        # Initialize Milvus vector store
        try:
            vector_store = MilvusVectorStore(
                uri=milvus_uri,
                token=config.MILVUS_TOKEN or None,
                collection_name=self.collection_name,
                dim=config.EMBEDDING_DIM,
                overwrite=self.rebuild_index,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus at {milvus_uri}: {e}")

        # If we are explicitly rebuilding, clear the old persist_dir
        if self.rebuild_index and os.path.exists(self.persist_dir):
            print(f"[RagEngine] Removing existing persist_dir: {self.persist_dir}")
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        # Try loading existing index from disk (if not rebuilding)
        if not self.rebuild_index and os.path.exists(self.persist_dir):
            try:
                print(f"[RagEngine] Loading existing index from {self.persist_dir}...")
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=self.persist_dir,
                )
                index = load_index_from_storage(storage_context)
                print("[RagEngine] Index loaded successfully.")
                return index
            except Exception as e:
                print(f"[RagEngine] Error loading index: {e}")
                print("[RagEngine] Falling back to full rebuild from Postgres...")
                self.rebuild_index = True

        # Build brand-new index from Postgres rows
        print("[RagEngine] Creating new index from Postgres documents...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = self._load_documents_from_db()

        if not documents:
            print("[RagEngine] No documents found in DB. Creating empty index.")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
            )
        else:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )

        os.makedirs(self.persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=self.persist_dir)
        print("[RagEngine] Index created and persisted!")
        return index

    def _load_documents_from_db(self) -> List[Document]:
        """
        Load all documents from Postgres and create LlamaIndex Document
        objects, with user_id and doc_id stored in metadata.
        """
        docs: List[Document] = []
        session: Session = SessionLocal()
        try:
            records = session.query(DocumentRecord).all()
        finally:
            session.close()

        if not records:
            print("[RagEngine] No rows in documents table.")
            return []

        print(f"[RagEngine] Loading {len(records)} document record(s) from Postgres...")

        for rec in records:
            path = Path(rec.filepath)
            if not path.exists():
                print(f"  ⚠ File not found on disk: {path}. Skipping.")
                continue

            text = self._extract_text_from_file(path)
            if not text:
                print(f"  ⚠ No text extracted from {path}. Skipping.")
                continue

            doc = Document(
                text=text,
                metadata={
                    "user_id": str(rec.user_id),
                    "doc_id": str(rec.id),
                    "source": rec.filename,
                    "file_path": str(path),
                    "filetype": rec.filetype,
                },
            )
            docs.append(doc)
            print(
                f"  ✓ Prepared doc for user_id={rec.user_id}, doc_id={rec.id}, file={rec.filename}"
            )

        print(f"[RagEngine] Prepared {len(docs)} LlamaIndex Document(s) from DB rows.")
        return docs

    def _extract_text_from_file(self, path: Path) -> str:
        """
        Extract text from a .txt or .pdf file.
        Uses PyMuPDF for PDFs to avoid the 'corrupted PDF' problem.
        """
        ext = path.suffix.lower()

        # Plain text
        if ext == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"  ⚠ Failed to read txt {path.name}: {e}")
                return ""

        # PDF via PyMuPDF
        if ext == ".pdf":
            try:
                pdf_doc = fitz.open(path)
                text_chunks = []
                for page in pdf_doc:
                    page_text = page.get_text()
                    if page_text:
                        text_chunks.append(page_text)
                pdf_doc.close()
                return "\n\n".join(text_chunks).strip()
            except Exception as e:
                print(f"  ⚠ Failed to parse PDF {path.name}: {e}")
                return ""

        print(f"  ⚠ Unsupported extension for extraction: {ext} (file={path})")
        return ""



    def answer(
        self,
        user_id: int,
        question: str,
        include_sources: bool = True,
    ) -> Dict[str, Any]:

        # Build metadata filter for user_id
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="user_id", value=str(user_id)),
            ]
        )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.TOP_K_RETRIEVAL,
            filters=filters,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=ResponseMode.COMPACT,
            verbose=config.VERBOSE,
        )

        response = query_engine.query(question)
        result: Dict[str, Any] = {"answer": str(response)}

        if include_sources:
            srcs = []
            for node in getattr(response, "source_nodes", []):
                meta = getattr(node, "metadata", {}) or {}
                srcs.append(
                    {
                        "text": _preview_text(node.text),
                        "source": meta.get("source")
                        or meta.get("file_path")
                        or "Unknown",
                        "score": getattr(node, "score", None),
                    }
                )
            result["sources"] = srcs

        return result

    def rebuild(self):

        print("[RagEngine] Rebuilding RAG index from Postgres...")
        self.rebuild_index = True
        self.index = self._initialize_index()
        print("[RagEngine] Rebuild complete.")

    def add_document_from_path(self, file_path: str, user_id: int, doc_id: int):

        path = Path(file_path)
        if not path.exists():
            print(f"[RagEngine] File not found: {file_path}")
            return

        text = self._extract_text_from_file(path)
        if not text:
            print(f"[RagEngine] No text extracted from {path}. Skipping insert.")
            return

        doc = Document(
            text=text,
            metadata={
                "user_id": str(user_id),
                "doc_id": str(doc_id),
                "source": path.name,
                "file_path": str(path),
                "filetype": path.suffix.lower().lstrip("."),
            },
        )

        print(
            f"[RagEngine] Incrementally adding doc for user_id={user_id}, doc_id={doc_id}, file={path.name}"
        )


        try:
            # VectorStoreIndex in recent LlamaIndex versions supports .insert()
            self.index.insert(doc)
        except AttributeError:
            # Fallback: re-create index view using existing storage_context.

            self.index = VectorStoreIndex.from_documents(
                [doc],
                storage_context=self.index.storage_context,
                show_progress=True,
            )

        # Persist updated index metadata
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("[RagEngine] Incremental insert complete and index persisted.")

    def list_sources(self, user_id: Optional[int] = None) -> List[str]:
        """
        Return a list of distinct sources stored in the docstore,
        optionally filtered by user_id.
        """
        sources = set()
        try:
            docstore = getattr(self.index, "docstore", None)
            if docstore is None:
                docstore = getattr(self.index.storage_context, "docstore", None)

            if docstore is None:
                print("[RagEngine] No docstore found on index or storage_context.")
                return []

            # Common SimpleDocumentStore pattern
            if hasattr(docstore, "docs"):
                iterable = docstore.docs.values()
            elif hasattr(docstore, "get_all_documents"):
                all_docs = docstore.get_all_documents()
                iterable = all_docs.values() if isinstance(all_docs, dict) else all_docs
            else:
                print("[RagEngine] Unsupported docstore structure.")
                return []

            for d in iterable:
                meta = getattr(d, "metadata", {}) or {}
                uid = meta.get("user_id")
                if user_id is not None and uid != str(user_id):
                    continue

                src = (
                    meta.get("source")
                    or meta.get("file_path")
                    or meta.get("file_name")
                )
                if src:
                    sources.add(src)

        except Exception as e:
            print(f"[RagEngine] Error listing sources: {e}")

        return sorted(sources)
