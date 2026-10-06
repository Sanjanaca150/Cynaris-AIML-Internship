from app.rag_pipeline import answer_question


def main():
    print("=" * 60)
    print("RAG Q&A CHATBOT")
    print("LangChain + ChromaDB + Groq")
    print("=" * 60)

    print("\nAsk questions about the indexed documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            result = answer_question(question)

            print("\nAssistant:")
            print(result["answer"])

            print("\nSources:")

            if result["sources"]:
                for source in result["sources"]:
                    print(f"- {source}")
            else:
                print("- No sources found.")

            print()

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()