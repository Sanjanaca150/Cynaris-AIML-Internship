import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


DATA_PATH = "data/aiml_ecommerce_clickstream.csv"
MODEL_PATH = "models/best_core_model.joblib"
OUTPUT_PATH = "docs/bias_audit.json"


df = pd.read_csv(DATA_PATH)

features = [
    "product_views",
    "add_to_cart",
    "session_duration_mins",
    "return_visitor",
    "discount_applied",
    "recommendation_clicked",
    "city",
    "device",
    "category",
]

X = df[features]
y = df["label_purchased"]

model = joblib.load(MODEL_PATH)

predictions = model.predict(X)


def evaluate_group(group_column):
    results = []

    for group_value, group_df in df.groupby(
        group_column
    ):
        indices = group_df.index

        actual = y.loc[indices]
        predicted = pd.Series(
            predictions,
            index=df.index,
        ).loc[indices]

        results.append(
            {
                "group": str(group_value),
                "samples": int(len(indices)),
                "accuracy": round(
                    float(
                        accuracy_score(
                            actual,
                            predicted,
                        )
                    ),
                    4,
                ),
                "f1_score": round(
                    float(
                        f1_score(
                            actual,
                            predicted,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }
        )

    return results


audit = {
    "purpose": (
        "Group-level performance audit for the "
        "purchase prediction model."
    ),
    "protected_attributes_available": False,
    "note": (
        "The supplied dataset does not contain "
        "protected demographic attributes such as "
        "race, gender, religion, or age. Therefore, "
        "this audit evaluates performance across "
        "available operational groups instead."
    ),
    "groups": {
        "category": evaluate_group("category"),
        "device": evaluate_group("device"),
        "city": evaluate_group("city"),
    },
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        audit,
        file,
        indent=2,
    )


print("Bias audit complete.")
print("Groups evaluated: category, device, city")
print(f"Report saved: {OUTPUT_PATH}")