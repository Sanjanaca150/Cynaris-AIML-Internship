
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "label_purchased"

NUMERIC_FEATURES = [
    "product_views",
    "add_to_cart",
    "session_duration_mins",
    "return_visitor",
    "discount_applied",
    "recommendation_clicked",
]

CATEGORICAL_FEATURES = [
    "city",
    "device",
    "category",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_clickstream(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the clickstream dataset."""

    cleaned = df.copy()

    # Convert date column to datetime
    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    # Remove duplicate sessions
    cleaned = cleaned.drop_duplicates(
        subset=["session_id"]
    )

    # Fill missing search queries
    if "search_query" in cleaned.columns:
        cleaned["search_query"] = cleaned["search_query"].fillna(
            "no_search"
        )

    return cleaned


def prepare_features(df: pd.DataFrame):
    """Prepare features and target for machine learning."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return X, y, preprocessor