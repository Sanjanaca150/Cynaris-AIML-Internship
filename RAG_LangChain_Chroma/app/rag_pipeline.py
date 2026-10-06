from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
)


def get_vector_store():
    """Load the existing ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_llm():
    """Create the Groq language model."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )


def format_sources(documents):
    """Return unique source document names."""
    sources = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")

        if source not in sources:
            sources.append(source)

    return sources


def answer_question(question):
    """Retrieve relevant context and generate a grounded answer."""
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=TOP_K,
    )

    if not documents:
        return {
            "answer": (
                "I could not find enough information in the "
                "provided documents to answer this question."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the information provided
in the context below.

Important rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. If the context does not contain enough information, respond with:
   "I could not find enough information in the provided documents
   to answer this question."
4. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": format_sources(documents),
    }