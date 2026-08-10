import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

questions = [
    "What is Deep Learning?",
    "Explain Cloud Computing.",
    "Why is Python popular?"
]

for model in models:
    print("\n" + "=" * 80)
    print(f"MODEL: {model}")
    print("=" * 80)

    for i, question in enumerate(questions, start=1):
        payload = {
            "model": model,
            "prompt": question,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)

        if response.status_code == 200:
            answer = response.json()["response"]

            print(f"\nQuestion {i}: {question}")
            print("-" * 80)
            print(answer)
            print("-" * 80)
        else:
            print(f"Error: {response.text}")