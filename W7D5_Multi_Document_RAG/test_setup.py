from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

print("=" * 70)
print("W7D5 SETUP TEST")
print("=" * 70)

print("\nTesting Ollama LLM...")
llm = Ollama(
    model="llama3.2:3b",
    request_timeout=120.0
)
print("✓ Ollama LLM configured")

print("\nTesting Ollama Embedding...")
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)
print("✓ Ollama embedding configured")

print("\nTesting embedding generation...")
embedding = embed_model.get_text_embedding(
    "This is a test document for multi-document RAG."
)

print(f"✓ Embedding generated")
print(f"Embedding dimensions: {len(embedding)}")

print("\n" + "=" * 70)
print("SETUP TEST PASSED")
print("=" * 70)