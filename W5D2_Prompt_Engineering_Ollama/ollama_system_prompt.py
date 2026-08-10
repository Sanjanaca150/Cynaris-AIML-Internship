from ollama import chat

system_prompt = """
You are an AI tutor.
Explain every answer in very simple English.
Keep answers under 100 words.
"""

prompts = [
    "What is Machine Learning?",
    "Explain Neural Networks.",
    "What is Prompt Engineering?",
    "What is Deep Learning?",
    "Difference between AI and ML?"
]

for i, prompt in enumerate(prompts, start=1):
    response = chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    print("=" * 60)
    print(f"Prompt {i}")
    print(prompt)
    print("-" * 60)
    print(response["message"]["content"])