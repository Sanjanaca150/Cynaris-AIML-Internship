from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

CLICKSTREAM_FILE = DATA_DIR / "aiml_ecommerce_clickstream.csv"
REVIEWS_FILE = DATA_DIR / "aiml_product_reviews.csv"
TEST_CASES_FILE = DATA_DIR / "aiml_ecommerce_tests.json"

MLFLOW_DB = BASE_DIR / "mlflow.db"