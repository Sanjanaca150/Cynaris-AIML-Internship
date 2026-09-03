"""
W8D5 - Local AI Research Assistant

A simple FastAPI prototype that retrieves relevant information
from a local knowledge base and returns a research-oriented answer.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Create the FastAPI application.
app = FastAPI(
    title="Local AI Research Assistant",
    description="A local knowledge-base research assistant prototype.",
    version="1.0.0",
)


# Locate the local knowledge base.
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.txt"


class ResearchRequest(BaseModel):
    """Request model for research questions."""

    question: str


class ResearchResponse(BaseModel):
    """Response model returned by the research endpoint."""

    question: str
    answer: str
    source: str


def load_knowledge_base() -> str:
    """Load the local knowledge base from disk."""

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError("Knowledge base file was not found.")

    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


def retrieve_answer(question: str) -> str:
    """
    Retrieve the most relevant section from the local knowledge base.

    The prototype uses simple keyword matching so that it remains
    lightweight and fully local.
    """

    knowledge_base = load_knowledge_base()

    sections = [
        section.strip()
        for section in knowledge_base.split("\n\n")
        if section.strip()
    ]

    question_lower = question.lower()

    # Match common research topics with their knowledge-base sections.
    topic_keywords = {
        "artificial intelligence": ["artificial intelligence", " ai "],
        "machine learning": ["machine learning", " ml "],
        "natural language processing": [
            "natural language processing",
            "nlp",
        ],
        "retrieval-augmented generation": [
            "retrieval-augmented generation",
            "rag",
        ],
        "large language models": [
            "large language models",
            "llm",
            "llms",
        ],
        "ai research assistants": [
            "research assistant",
            "ai research",
        ],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            for section in sections:
                if topic in section.lower():
                    return section

    raise HTTPException(
        status_code=404,
        detail="No relevant information found for the question.",
    )


@app.get("/")
def home():
    """Return a basic health message."""

    return {
        "message": "Local AI Research Assistant is running.",
        "status": "healthy",
    }


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """Answer a research question using the local knowledge base."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    answer = retrieve_answer(question)

    return ResearchResponse(
        question=question,
        answer=answer,
        source="data/knowledge_base.txt",
    )