import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Answer questions clearly and simply.
Keep answers concise and suitable for a beginner.
If the question is technical, provide a short example when useful.
"""

def ask_ollama(question):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]


prompts = [
    "What is machine learning?",
    "What is the difference between AI and machine learning?",
    "What is a Python list?",
    "What is an API?",
    "Why are vector databases useful for AI applications?"
]

print("W5D5 - Ollama Local LLM Inference")
print("=" * 50)

for i, prompt in enumerate(prompts, start=1):
    print(f"\nPrompt {i}: {prompt}")
    print("-" * 50)

    answer = ask_ollama(prompt)

    print("Response:")
    print(answer)