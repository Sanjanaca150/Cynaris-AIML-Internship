from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "healthcare_knowledge.txt"
CHROMA_DIR = BASE_DIR / "output" / "chroma_baseline"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# -----------------------------
# Load documents
# -----------------------------
print("Loading healthcare dataset...")

loader = TextLoader(str(DATA_FILE), encoding="utf-8")
documents = loader.load()

print(f"Loaded documents: {len(documents)}")


# -----------------------------
# Split documents
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# -----------------------------
# Create embeddings
# -----------------------------
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# -----------------------------
# Create ChromaDB
# -----------------------------
print("Creating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
    collection_name="healthcare_rag_baseline",
)

print("ChromaDB created successfully.")


# -----------------------------
# Create retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)


# -----------------------------
# Create Ollama LLM
# -----------------------------
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


# -----------------------------
# RAG function
# -----------------------------
def answer_question(question: str):
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content for document in retrieved_docs
    )

    prompt = f"""
Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs


# -----------------------------
# Test RAG pipeline
# -----------------------------
if __name__ == "__main__":
    question = "What are common symptoms of diabetes?"

    answer, retrieved_docs = answer_question(question)

    print("\n" + "=" * 60)
    print("RAG TEST")
    print("=" * 60)

    print(f"\nQuestion:\n{question}")

    print(f"\nAnswer:\n{answer}")

    print("\nRetrieved Context:")
    for index, document in enumerate(retrieved_docs, start=1):
        print(f"\n--- Context {index} ---")
        print(document.page_content)

    print("\n" + "=" * 60)
    print("RAG pipeline test completed.")
    print("=" * 60)