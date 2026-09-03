"""
W8D4 - Documentation, Testing & Code Review

A simple FastAPI application demonstrating:
- API documentation
- Input validation
- Error handling
- Clean code structure
- Unit testing
- Health monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="W8D4 Documentation and Testing API",
    description=(
        "A small production-style API created for Week 8 Day 4 "
        "documentation, testing, and code review."
    ),
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request model used for knowledge-base queries."""

    question: str = Field(
        ...,
        min_length=3,
        description="Question to search in the knowledge base.",
    )


class QueryResponse(BaseModel):
    """Response model returned by the query endpoint."""

    question: str
    answer: str
    source: str


KNOWLEDGE_BASE = {
    "machine learning": (
        "Machine learning is a branch of AI that enables systems "
        "to learn patterns from data and make predictions."
    ),
    "rag": (
        "Retrieval-Augmented Generation combines document retrieval "
        "with language generation to produce context-aware answers."
    ),
    "docker": (
        "Docker packages an application and its dependencies into "
        "portable containers."
    ),
    "testing": (
        "Automated testing verifies that application components "
        "behave as expected."
    ),
}


@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "W8D4 Documentation and Testing API",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    """Return the health status of the API."""
    return {
        "status": "healthy",
        "service": "w8d4-api",
    }


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """
    Search the knowledge base using a keyword from the question.

    Raises:
        HTTPException: If no matching topic is found.
    """

    question = request.question.strip()

    for topic, answer in KNOWLEDGE_BASE.items():
        if topic in question.lower():
            return QueryResponse(
                question=question,
                answer=answer,
                source=f"knowledge_base/{topic}",
            )

    raise HTTPException(
        status_code=404,
        detail="No relevant information found for the question.",
    )