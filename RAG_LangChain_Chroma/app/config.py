import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Please add it to the .env file."
    )

GROQ_MODEL = "openai/gpt-oss-20b"

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

TOP_K = 4