"""
W7D4 — LlamaIndex + Ollama — Local RAG
Cynaris Solutions Internship

Practical Tasks:
1. Verify Ollama local inference.
2. Call Ollama API with a custom system prompt.
3. Test 5 prompts.
4. Compare llama3.2:3b and qwen2.5:3b on 3 questions.
5. Document response differences.
"""

import json
import time
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/chat"

LLAMA_MODEL = "llama3.2:3b"
QWEN_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML internship mentor.

Answer questions clearly and accurately.
Keep explanations beginner-friendly.
For technical questions, use short examples where useful.
Do not invent facts.
"""


def call_ollama(model, user_prompt):
    """Send a chat request to the local Ollama API."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start_time = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        elapsed = time.perf_counter() - start_time

        answer = result.get("message", {}).get("content", "")

        return answer.strip(), elapsed

    except urllib.error.URLError as error:
        return f"ERROR: Unable to connect to Ollama: {error}", 0

    except Exception as error:
        return f"ERROR: {error}", 0


def test_five_prompts():
    """Task 2 — Test custom system prompt with 5 prompts."""

    prompts = [
        "What is an LLM?",
        "What is RAG and why is it useful?",
        "Explain vector embeddings in simple terms.",
        "What is the difference between AI and machine learning?",
        "Why can a local LLM be useful for privacy?"
    ]

    print("\n" + "=" * 70)
    print("TASK 2 — FIVE PROMPTS WITH CUSTOM SYSTEM PROMPT")
    print("=" * 70)

    for index, prompt in enumerate(prompts, start=1):

        print(f"\nPrompt {index}")
        print("-" * 70)
        print(prompt)

        answer, elapsed = call_ollama(LLAMA_MODEL, prompt)

        print("\nResponse:")
        print(answer)

        print(f"\nResponse time: {elapsed:.2f} seconds")


def compare_models():
    """Task 3 — Compare llama3.2:3b and qwen2.5:3b."""

    questions = [
        "Explain Retrieval-Augmented Generation (RAG) in simple terms.",
        "What is the difference between BM25 retrieval and dense retrieval?",
        "Why might an organization choose a local LLM instead of a cloud API?"
    ]

    print("\n" + "=" * 70)
    print("TASK 3 — LLAMA 3.2 vs QWEN 2.5 COMPARISON")
    print("=" * 70)

    for index, question in enumerate(questions, start=1):

        print(f"\nQuestion {index}")
        print("-" * 70)
        print(question)

        llama_answer, llama_time = call_ollama(
            LLAMA_MODEL,
            question
        )

        qwen_answer, qwen_time = call_ollama(
            QWEN_MODEL,
            question
        )

        print("\nLLAMA 3.2:3B")
        print("-" * 30)
        print(llama_answer)
        print(f"Response time: {llama_time:.2f} seconds")

        print("\nQWEN 2.5:3B")
        print("-" * 30)
        print(qwen_answer)
        print(f"Response time: {qwen_time:.2f} seconds")

        print("\nComparison:")
        print(
            "Both models were tested with the same question and "
            "the same custom system prompt."
        )


def main():
    print("=" * 70)
    print("W7D4 — LLAMAINDEX + OLLAMA — LOCAL RAG")
    print("=" * 70)

    print("\nOllama API:")
    print(OLLAMA_URL)

    print("\nModels:")
    print(f"- {LLAMA_MODEL}")
    print(f"- {QWEN_MODEL}")

    # First local inference
    print("\n" + "=" * 70)
    print("TASK 1 — FIRST LOCAL INFERENCE")
    print("=" * 70)

    answer, elapsed = call_ollama(
        LLAMA_MODEL,
        "Explain what a local LLM is in two sentences."
    )

    print("\nPrompt:")
    print("Explain what a local LLM is in two sentences.")

    print("\nResponse:")
    print(answer)

    print(f"\nResponse time: {elapsed:.2f} seconds")

    # Task 2
    test_five_prompts()

    # Task 3
    compare_models()

    print("\n" + "=" * 70)
    print("W7D4 PRACTICAL EXECUTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()