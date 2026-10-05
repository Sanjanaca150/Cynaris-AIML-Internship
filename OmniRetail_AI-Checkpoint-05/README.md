\# OmniRetail AI



\## End-to-End Retail AI Platform



OmniRetail AI is an end-to-end AI/ML platform designed for retail demand forecasting, dynamic pricing, and inventory replenishment.



The system combines machine learning, time-series forecasting, experiment tracking, API serving, and manager-level visualization into a single demonstration platform.



\---



\## Project Modules



\### 1. Demand Forecasting



Uses \*\*Prophet\*\* to forecast SKU demand for the next 60 days.



Validated result:



\* SKU: SKU-001

\* Training rows: 668

\* Testing rows: 60

\* MAPE: \*\*14.4%\*\*

\* Acceptance target: MAPE < 15%

\* Result: \*\*PASSED\*\*



\### 2. Dynamic Pricing



Generates price recommendations using:



\* Base price

\* Average demand

\* Forecast demand

\* Inventory level



Validation:



\* \*\*20 SKUs tested\*\*

\* Result: \*\*PASSED\*\*



\### 3. Inventory Replenishment



Generates inventory alerts using forecast demand and inventory levels.



Current demonstration:



\* HEALTHY: 12 SKUs

\* REORDER: 4 SKUs

\* CRITICAL: 4 SKUs

\* Total analyzed: \*\*20 SKUs\*\*



\---



\## Machine Learning



\### Core Model



Random Forest Classifier



Best configuration:



\* n\_estimators: 100

\* max\_depth: 5

\* min\_samples\_split: 2

\* random\_state: 42



Evaluation:



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 53.50% |

| Precision | 54.89% |

| Recall    | 90.99% |

| F1 Score  | 68.47% |

| ROC-AUC   | 42.58% |



\---



\## MLflow Experiment Tracking



MLflow is used for experiment tracking.



Experiment:



`OmniRetail\_Checkpoint\_5\_Forecasting`



\*\*10 forecasting runs\*\* were logged with:



\* SKU

\* Model parameters

\* Training rows

\* Testing rows

\* MAPE



\---



\## Data



Supplied data:



\* `aiml\_ecommerce\_clickstream.csv` — 1,000 rows

\* `aiml\_product\_reviews.csv` — 800 rows

\* `aiml\_ecommerce\_tests.json` — 50 test cases



The supplied clickstream data does not contain historical SKU-level sales quantities or actual SKU identifiers.



Therefore, a \*\*derived/synthetic 20-SKU demand dataset\*\* was created from the supplied clickstream purchase/session activity for the end-to-end demonstration.



Generated demand data:



\* 20 SKUs

\* 14,560 daily records

\* January 2023 – December 2024

\* 5 categories



This distinction is documented because the synthetic demand data should not be interpreted as production retail sales data.



\---



\## Technology Stack



\* Python 3.12

\* Pandas

\* NumPy

\* Scikit-learn

\* Prophet

\* FastAPI

\* Uvicorn

\* Streamlit

\* Plotly

\* MLflow

\* Pytest

\* HTTPX

\* GitHub Actions



\---



\## Project Structure



```text

OmniRetail\_AI-Checkpoint-05/

│

├── app/

│   ├── \_\_init\_\_.py

│   ├── main.py

│   ├── forecasting.py

│   ├── pricing.py

│   ├── inventory.py

│   └── dashboard.py

│

├── data/

│   ├── aiml\_ecommerce\_clickstream.csv

│   ├── aiml\_product\_reviews.csv

│   ├── aiml\_ecommerce\_tests.json

│   └── demand\_data.csv

│

├── models/

│   ├── best\_core\_model.joblib

│   └── prophet\_SKU-001.json

│

├── tests/

│   ├── conftest.py

│   ├── test\_forecasting.py

│   ├── test\_pricing.py

│   ├── test\_inventory.py

│   └── test\_api.py

│

├── docs/

│   ├── MODEL\_CARD.md

│   ├── DEPLOYMENT\_GUIDE.md

│   └── bias\_audit.json

│

├── .github/

│   └── workflows/

│       └── ci.yml

│

├── run\_experiments.py

├── requirements.txt

├── mlflow.db

└── README.md

```



\---



\## Running the Application



\### 1. Activate the environment



```text

.venv\\Scripts\\activate

```



\### 2. Start FastAPI



```text

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



API documentation:



```text

http://127.0.0.1:8000/docs

```



\### 3. Start the Manager Dashboard



Open another terminal:



```text

.venv\\Scripts\\activate

streamlit run app/dashboard.py

```



\### 4. Start MLflow



Open another terminal:



```text

.venv\\Scripts\\activate

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

```



MLflow UI:



```text

http://127.0.0.1:5000

```



\---



\## API Endpoints



| Endpoint             | Purpose                         |

| -------------------- | ------------------------------- |

| `/`                  | System information              |

| `/health`            | Health check                    |

| `/forecast/{sku\_id}` | Demand forecast                 |

| `/pricing`           | Dynamic pricing recommendations |

| `/inventory`         | Inventory alerts                |

| `/dashboard-data`    | Manager dashboard KPIs          |



\---



\## Testing



Run the complete test suite:



```text

pytest -q

```



Latest local result:



```text

4 passed

```



GitHub Actions automatically runs the test suite on pushes and pull requests.



\---



\## Bias Audit



A group-level performance audit was performed across:



\* Category

\* Device

\* City



Metrics include:



\* Accuracy

\* F1 score

\* Sample count



The supplied dataset does not contain protected demographic attributes. Therefore, the audit does not make a protected-class fairness claim.



Results:



`docs/bias\_audit.json`



\---



\## Acceptance Criteria



| Requirement               | Result        |

| ------------------------- | ------------- |

| Demand forecasting        | ✅ Implemented |

| Forecast MAPE < 15%       | ✅ 14.4%       |

| Dynamic pricing           | ✅ Implemented |

| Pricing tested on 20 SKUs | ✅ 20          |

| Inventory replenishment   | ✅ Implemented |

| Manager dashboard         | ✅ Implemented |

| MLflow 10+ experiments    | ✅ 10 runs     |

| FastAPI serving           | ✅ Implemented |

| Integration testing       | ✅ Passed      |

| Bias audit                | ✅ Completed   |

| Model card                | ✅ Completed   |

| Deployment guide          | ✅ Completed   |

| CI pipeline               | ✅ Implemented |



\---



\## Limitations



This project is an end-to-end demonstration.



The main limitations are:



\* The demand dataset is derived/synthetic.

\* Real SKU-level sales history was not provided.

\* Forecast MAPE was validated on SKU-001.

\* Pricing is a heuristic demonstration rather than a causal elasticity model.

\* Inventory values are demonstration values.

\* There is no live ERP/WMS integration.

\* Production deployment would require monitoring, authentication, real-time data, and retraining.



\---



\## Documentation



\* `docs/MODEL\_CARD.md` — model information, metrics, limitations, bias audit and deployment details

\* `docs/DEPLOYMENT\_GUIDE.md` — setup and deployment instructions

\* `docs/bias\_audit.json` — group-level audit results



\---



\## Conclusion



OmniRetail AI demonstrates a complete retail AI workflow from data processing and machine learning through forecasting, pricing, inventory intelligence, API serving, experiment tracking, dashboard visualization, testing, and CI.



The project meets the demonstrated Checkpoint 5 acceptance targets, with \*\*14.4% forecast MAPE\*\*, \*\*20 SKUs tested for pricing\*\*, \*\*20 SKUs analyzed for inventory\*\*, and \*\*10 MLflow forecasting runs\*\*.



