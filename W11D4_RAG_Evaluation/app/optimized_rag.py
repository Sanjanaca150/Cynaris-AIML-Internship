import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "healthcare_knowledge.txt"
CHROMA_DIR = BASE_DIR / "output" / "chroma_optimized"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"


def build_optimized_rag():
    print("Loading healthcare dataset...")

    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    # Remove old ChromaDB so repeated runs do not create duplicates
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print("Creating optimized ChromaDB...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="healthcare_rag_optimized",
        persist_directory=str(CHROMA_DIR),
    )

    print("Optimized ChromaDB created successfully.")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a healthcare information assistant.

Answer the question using ONLY the information provided in the context.

If the answer is not available in the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return retriever, llm, prompt


def answer_question(question, retriever, llm, prompt):
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content for document in retrieved_docs
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content, retrieved_docs


if __name__ == "__main__":
    retriever, llm, prompt = build_optimized_rag()

    question = "What are common symptoms of diabetes?"

    answer, retrieved_docs = answer_question(
        question,
        retriever,
        llm,
        prompt,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZED RAG TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print(f"\nNumber of retrieved chunks: {len(retrieved_docs)}")

    print("\nRetrieved Context:")

    for index, document in enumerate(retrieved_docs, start=1):
        print(f"\n--- Context {index} ---")
        print(document.page_content)

    print("\n" + "=" * 60)
    print("OPTIMIZED RAG PIPELINE TEST COMPLETED")
    print("=" * 60)