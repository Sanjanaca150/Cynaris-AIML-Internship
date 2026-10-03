from datasets import Dataset

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from langchain_ollama import ChatOllama, OllamaEmbeddings


data = Dataset.from_dict(
    {
        "user_input": [
            "What are common symptoms of diabetes?"
        ],
        "response": [
            "Common symptoms of diabetes include increased thirst, "
            "frequent urination, increased hunger, fatigue, and blurred vision."
        ],
        "retrieved_contexts": [
            [
                "Common symptoms of diabetes can include increased thirst, "
                "frequent urination, increased hunger, fatigue, and blurred vision."
            ]
        ],
        "reference": [
            "Increased thirst, frequent urination, increased hunger, fatigue, "
            "and blurred vision."
        ],
    }
)


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_predict=128,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


print("=" * 60)
print("RAGAS DIAGNOSTIC TEST")
print("=" * 60)

print("\nRunning one-question Faithfulness evaluation...")

result = evaluate(
    dataset=data,
    metrics=[
        Faithfulness(llm=ragas_llm)
    ],
    batch_size=1,
)

print("\nRAW RESULT:")
print(result)

print("\nRESULT TYPE:")
print(type(result))

print("\nRESULT DICTIONARY:")
print(result.to_pandas().to_dict())

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)