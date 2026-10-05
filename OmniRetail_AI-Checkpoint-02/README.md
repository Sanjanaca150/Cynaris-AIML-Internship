\# OmniRetail AI — Checkpoint 2



\## Data Pipeline + Baseline Model



Checkpoint 2 focuses on building the complete data ingestion and preprocessing pipeline, training a baseline machine learning model, establishing an evaluation benchmark, and integrating MLflow experiment tracking.



\---



\## Project Objective



OmniRetail AI is an AI/ML solution for an omnichannel retail environment.



The overall project aims to support:



\* Demand forecasting

\* Dynamic pricing

\* Inventory replenishment alerts

\* FastAPI backend services

\* MLflow experiment tracking

\* Manager-facing KPI dashboards



This checkpoint establishes the data and machine learning foundation for these later components.



\---



\## Checkpoint 2 Scope



The following components were completed:



\* Multi-dataset data ingestion

\* Clickstream data cleaning

\* Feature preparation

\* Train-test splitting

\* Baseline purchase prediction model

\* Model evaluation

\* MLflow experiment tracking

\* Model artifact persistence

\* Metrics persistence

\* Automated tests

\* Baseline documentation



\---



\## Dataset



\### Clickstream Dataset



File:



`data/aiml\_ecommerce\_clickstream.csv`



Statistics:



\* Records: 1,000

\* Columns: 16

\* Training records: 800

\* Testing records: 200

\* Target: `label\_purchased`



\### Product Reviews



File:



`data/aiml\_product\_reviews.csv`



\* Records: 800



\### E-commerce Test Cases



File:



`data/aiml\_ecommerce\_tests.json`



\* Test cases: 50



\---



\## Data Pipeline



The ingestion layer is implemented in:



`app/data\_loader.py`



It provides loaders for:



\* Clickstream CSV data

\* Product reviews CSV data

\* E-commerce JSON test cases



The preprocessing layer is implemented in:



`app/preprocessing.py`



\### Cleaning



The pipeline:



1\. Converts the date field to datetime.

2\. Removes duplicate sessions.

3\. Fills missing search queries with `no\_search`.

4\. Validates required columns.

5\. Separates features and target.



\---



\## Baseline Features



The baseline model uses nine features.



\### Numerical



\* `product\_views`

\* `add\_to\_cart`

\* `session\_duration\_mins`

\* `return\_visitor`

\* `discount\_applied`

\* `recommendation\_clicked`



\### Categorical



\* `city`

\* `device`

\* `category`



Potential leakage fields such as `purchase\_made` and `order\_value\_inr` are excluded from the model.



Identifiers such as `session\_id` and `user\_id` are also excluded.



\---



\## Baseline Model



Model:



\*\*Logistic Regression\*\*



The model is implemented as a scikit-learn Pipeline with:



\* Numerical imputation

\* Standard scaling

\* Categorical imputation

\* One-hot encoding

\* Logistic Regression classification



\### Training configuration



\* Train/test split: 80/20

\* Random state: 42

\* Maximum iterations: 1,000

\* Stratification: Enabled



\---



\## Baseline Results



The baseline model was evaluated on 200 test records.



| Metric    | Result |

| --------- | -----: |

| Accuracy  | 0.4900 |

| Precision | 0.5306 |

| Recall    | 0.7027 |

| F1 Score  | 0.6047 |

| ROC-AUC   | 0.4498 |



These results establish the initial benchmark for future model improvements.



The ROC-AUC result indicates that the baseline has limited discriminative performance and should not be considered production-ready.



\---



\## MLflow Tracking



MLflow experiment tracking is integrated into the training pipeline.



Experiment:



`OmniRetail\_Checkpoint\_2\_Baseline`



Tracked information includes:



\* Model parameters

\* Dataset size

\* Feature count

\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* ROC-AUC

\* Model artifact



The tracking backend uses a local SQLite database:



`mlflow.db`



\---



\## Project Structure



```text

OmniRetail\_AI-Checkpoint-02/

│

├── app/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── data\_loader.py

│   ├── preprocessing.py

│   └── baseline\_model.py

│

├── data/

│   ├── aiml\_ecommerce\_clickstream.csv

│   ├── aiml\_product\_reviews.csv

│   └── aiml\_ecommerce\_tests.json

│

├── docs/

│   ├── BASELINE\_REPORT.md

│   └── baseline\_metrics.json

│

├── models/

│   └── baseline\_purchase\_model.joblib

│

├── tests/

│   └── test\_baseline\_model.py

│

├── mlflow.db

├── requirements.txt

└── README.md

```



\---



\## Testing



Automated tests are implemented using pytest.



The test suite verifies:



\* Dataset loading

\* Product review loading

\* Test case loading

\* Preprocessing

\* Saved model

\* Saved metrics

\* MLflow database



Latest test result:



\*\*7 passed\*\*



\---



\## Model Limitations



The current Logistic Regression model is a baseline and is not intended for production deployment.



Future improvements may include:



\* Feature engineering

\* Behavioral feature extraction

\* Tree-based models

\* Hyperparameter tuning

\* Cross-validation

\* Class imbalance analysis

\* Advanced purchase prediction models



The project-level demand forecasting requirement will be evaluated separately using forecasting metrics such as MAPE.



\---



\## Next Development Stage



The baseline established in Checkpoint 2 will serve as the reference benchmark for subsequent OmniRetail AI development.



Future stages can build on this foundation to develop:



\* Improved machine learning models

\* Demand forecasting

\* Dynamic pricing

\* Inventory replenishment

\* FastAPI services

\* MLflow experiment comparison

\* Manager dashboard and KPI visualization



