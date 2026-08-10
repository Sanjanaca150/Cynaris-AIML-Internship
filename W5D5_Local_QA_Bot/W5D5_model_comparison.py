import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

questions = [
    "What is machine learning?",
    "What is an API? Give a simple example.",
    "Why are vector databases useful for AI applications?"
]

system_prompt = """
You are a helpful AI/ML learning assistant.
Answer clearly and simply for a beginner.
Keep the response concise and accurate.
"""

def ask_model(model, question):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]


print("W5D5 - Model Comparison")
print("=" * 70)

for question_number, question in enumerate(questions, start=1):
    print(f"\nQUESTION {question_number}: {question}")
    print("=" * 70)

    for model in models:
        print(f"\nMODEL: {model}")
        print("-" * 70)

        answer = ask_model(model, question)
        print(answer)