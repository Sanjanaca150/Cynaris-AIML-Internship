from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def validate_reviews():
    path = DATA_DIR / "aiml_product_reviews.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    print("Product Reviews Dataset")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print()


def validate_test_cases():
    path = DATA_DIR / "aiml_ecommerce_tests.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("E-Commerce Test Cases")
    print(f"Records: {len(data)}")

    if data:
        print(f"Fields: {list(data[0].keys())}")

    print()


def main():
    print("=== OmniRetail AI Data Validation ===")
    print()

    validate_reviews()
    validate_test_cases()

    clickstream = DATA_DIR / "aiml_ecommerce_clickstream.csv"

    if clickstream.exists():
        print("Clickstream Dataset: Available")
    else:
        print("Clickstream Dataset: Pending")

    print()
    print("Data validation completed successfully.")


if __name__ == "__main__":
    main()