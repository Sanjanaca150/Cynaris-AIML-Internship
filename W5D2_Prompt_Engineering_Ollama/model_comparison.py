from ollama import chat

questions = [
    "Explain Artificial Intelligence.",
    "What is Prompt Engineering?",
    "Why is Python popular?"
]

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

for question in questions:
    print("=" * 80)
    print("QUESTION:", question)

    for model in models:
        response = chat(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ]
        )

        print("\nMODEL:", model)
        print(response["message"]["content"])