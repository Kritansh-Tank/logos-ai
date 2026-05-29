import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

# RAG settings
TOP_K_PASSAGES = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paths — embeddings live in backend/data/ (built by rag/build_index.py)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Conversation memory
MAX_HISTORY_TURNS = 20

# Image generation
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"
