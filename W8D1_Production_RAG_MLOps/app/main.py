from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    description="Containerized RAG API for W8D1 MLOps"
)


class QueryRequest(BaseModel):
    query: str


KNOWLEDGE_BASE = {
    "machine learning": (
        "Machine learning is a branch of artificial intelligence "
        "that enables systems to learn patterns from data and make predictions."
    ),
    "rag": (
        "Retrieval-Augmented Generation combines document retrieval "
        "with language generation to produce answers grounded in retrieved information."
    ),
    "docker": (
        "Docker packages applications and their dependencies into "
        "portable containers that can run consistently across environments."
    ),
}


@app.get("/")
def root():
    return {
        "message": "Production RAG API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/query")
def query(request: QueryRequest):
    query_text = request.query.lower()

    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in query_text:
            return {
                "query": request.query,
                "answer": answer,
                "source": "knowledge_base"
            }

    return {
        "query": request.query,
        "answer": "No relevant information was found in the knowledge base.",
        "source": "knowledge_base"
    }