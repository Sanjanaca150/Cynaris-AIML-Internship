import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Give clear, concise, and beginner-friendly answers."
)

prompts = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning in simple words.",
    "What is Python used for?",
    "Difference between AI and ML.",
    "Why should we use local LLMs?"
]

for i, prompt in enumerate(prompts, start=1):
    payload = {
        "model": "llama3.2:3b",
        "prompt": f"System: {SYSTEM_PROMPT}\nUser: {prompt}",
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        result = response.json()["response"]
        print("=" * 70)
        print(f"Prompt {i}: {prompt}")
        print("-" * 70)
        print(result)
        print()
    else:
        print(f"Error: {response.text}")