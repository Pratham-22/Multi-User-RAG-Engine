import os
from pathlib import Path


BASE_DIR = Path("").resolve()

# Directory where raw documents (.txt, .pdf) live
DATA_DIRECTORY = str(BASE_DIR / "data")

# Milvus Lite local DB (vector store)
MILVUS_URI = str(BASE_DIR / "milvus" / "milvus_lite.db")

# Directory where LlamaIndex persists index metadata
PERSIST_DIR = str(BASE_DIR / "storage")

# Ensure directories exist (they'll be created at runtime if missing)
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(MILVUS_URI), exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)


EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
EMBEDDING_DEVICE = "cpu"  # "cuda" if you have GPU

# Embedding dimension must match the model
EMBEDDING_DIM = 768


LLM_TYPE = "llama"  


LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

LLM_CONTEXT_WINDOW = 2048
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2


LLM_DEVICE_MAP = "auto"


# If you're using Milvus Lite, token is None
MILVUS_TOKEN = None


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# How many similar chunks to retrieve
TOP_K_RETRIEVAL = 5

# Verbose retrieval/response logs from LlamaIndex
VERBOSE = False


# If True, try to load PDFs via LlamaIndex PDFReader
LOAD_PDF_FILES = True
