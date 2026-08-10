from pypdf import PdfReader
import chromadb
import ollama

# Read PDF
reader = PdfReader("SANJANA C A_RESUME.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted + "\n"

# Split into chunks
chunk_size = 500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Create ChromaDB client
client = chromadb.Client()

collection = client.create_collection(name="resume_collection")

# Add chunks
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[str(i)]
    )

# Query
question = "Summarize this resume."

results = collection.query(
    query_texts=[question],
    n_results=3
)

print("\nTop 3 Retrieved Chunks:\n")

for i, chunk in enumerate(results["documents"][0], start=1):
    print(f"Chunk {i}")
    print(chunk)
    print("-" * 60)

# Combine retrieved chunks
context = "\n".join(results["documents"][0])

# Ask Ollama
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": f"""
Use only the information below.

{context}

Question:
{question}
"""
        }
    ]
)

print("\nLLM Answer:\n")
print(response["message"]["content"])