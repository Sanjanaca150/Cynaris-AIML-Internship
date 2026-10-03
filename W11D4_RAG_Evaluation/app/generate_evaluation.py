import json
from pathlib import Path

from rag_evaluation import answer_question


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "output" / "evaluation_dataset.json"


questions = [
    {
        "question": "What are common symptoms of diabetes?",
        "ground_truth": (
            "Common symptoms of diabetes include increased thirst, "
            "frequent urination, increased hunger, fatigue, and blurred vision."
        ),
    },
    {
        "question": "What is hypertension?",
        "ground_truth": (
            "Hypertension is a condition in which the force of blood "
            "against artery walls remains consistently elevated."
        ),
    },
    {
        "question": "What are common symptoms of asthma?",
        "ground_truth": (
            "Common symptoms of asthma include wheezing, coughing, "
            "chest tightness, and shortness of breath."
        ),
    },
    {
        "question": "What is one common cause of anemia?",
        "ground_truth": (
            "Iron deficiency is one common cause of anemia."
        ),
    },
    {
        "question": "Are antibiotics effective against the common cold?",
        "ground_truth": (
            "No. Antibiotics do not treat viral infections such as "
            "the common cold."
        ),
    },
    {
        "question": "What is influenza?",
        "ground_truth": (
            "Influenza, or flu, is a contagious respiratory illness "
            "caused by influenza viruses."
        ),
    },
    {
        "question": "What are some risk factors for heart disease?",
        "ground_truth": (
            "Risk factors can include high blood pressure, high cholesterol, "
            "smoking, diabetes, physical inactivity, and family history."
        ),
    },
    {
        "question": "What do the kidneys do?",
        "ground_truth": (
            "The kidneys filter waste and excess fluid from the blood "
            "and help regulate fluid, electrolytes, and blood pressure."
        ),
    },
    {
        "question": "What does a balanced diet provide?",
        "ground_truth": (
            "A balanced diet provides carbohydrates, proteins, fats, "
            "vitamins, minerals, fiber, and water."
        ),
    },
    {
        "question": "What is preventive healthcare?",
        "ground_truth": (
            "Preventive healthcare focuses on reducing the risk of disease "
            "and detecting health problems early."
        ),
    },
]


def main():
    evaluation_data = []

    print("=" * 60)
    print("GENERATING 10-Q&A EVALUATION DATASET")
    print("=" * 60)

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"\n[{index}/10] {question}")

        answer, retrieved_docs = answer_question(question)

        contexts = [
            document.page_content
            for document in retrieved_docs
        ]

        evaluation_data.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )

        print(f"Answer: {answer}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            evaluation_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("10-Q&A DATASET GENERATED SUCCESSFULLY")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()