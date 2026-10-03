import json
import math
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "output" / "evaluation_dataset.json"
OUTPUT_FILE = BASE_DIR / "output" / "optimized_ragas_results.json"


print("=" * 60)
print("OPTIMIZED RAGAS EVALUATION")
print("=" * 60)

print("\nLoading evaluation dataset...")

with open(DATASET_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Loaded evaluation items: {len(data)}")


# ---------------------------------------------------------
# Prepare Ragas dataset
# ---------------------------------------------------------

questions = []
answers = []
contexts = []
ground_truths = []

for item in data:
    questions.append(item["question"])
    answers.append(item["answer"])
    contexts.append(item["contexts"])
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict(
    {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
)


# ---------------------------------------------------------
# Ollama models
# ---------------------------------------------------------

print("\nLoading Ollama models...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_predict=256,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------
# Ragas metrics
# ---------------------------------------------------------

metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    ),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

print("\nRunning optimized Ragas evaluation...")
print("This may take some time with the local Ollama model.\n")

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    batch_size=1,
)


# ---------------------------------------------------------
# Extract actual scores
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OPTIMIZED RAGAS RESULTS")
print("=" * 60)

scores = {}

metric_names = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

for metric_name in metric_names:
    value = None

    try:
        value = result[metric_name]

        if hasattr(value, "mean"):
            value = value.mean()

        value = float(value)

        if math.isnan(value):
            value = None
        else:
            value = round(value, 4)

    except (KeyError, TypeError, ValueError, AttributeError):
        value = None

    scores[metric_name] = value

    print(f"{metric_name}: {value}")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

optimized_results = {
    "evaluation_type": "optimized",
    "dataset_size": len(data),
    "chunk_size": 300,
    "chunk_overlap": 50,
    "retrieval_k": 3,
    "llm": "llama3.2:3b",
    "embedding_model": "nomic-embed-text:latest",
    "scores": scores,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        optimized_results,
        file,
        indent=4,
    )

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("OPTIMIZED RAGAS EVALUATION COMPLETED")
print("=" * 60)