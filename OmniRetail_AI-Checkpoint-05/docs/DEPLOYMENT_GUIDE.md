\# OmniRetail AI — Deployment Guide



\## 1. Environment



Recommended environment:



\* Python 3.12

\* FastAPI

\* Uvicorn

\* Streamlit

\* Plotly

\* Prophet

\* Scikit-learn

\* MLflow



Create and activate the virtual environment:



```text

py -3.12 -m venv .venv

.venv\\Scripts\\activate

```



Install dependencies:



```text

pip install -r requirements.txt

```



\---



\## 2. Start the FastAPI Backend



From the project root:



```text

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



Health check:



```text

http://127.0.0.1:8000/health

```



API documentation:



```text

http://127.0.0.1:8000/docs

```



\---



\## 3. API Endpoints



\### Health



```text

GET /health

```



\### Demand Forecast



```text

GET /forecast/{sku\_id}

```



Example:



```text

GET /forecast/SKU-001

```



\### Dynamic Pricing



```text

GET /pricing

```



\### Inventory



```text

GET /inventory

```



\### Dashboard Data



```text

GET /dashboard-data

```



\---



\## 4. Start the Manager Dashboard



Open a second terminal, activate the virtual environment, and run:



```text

streamlit run app/dashboard.py

```



The Streamlit dashboard provides:



\* Total SKU KPI

\* Forecast MAPE

\* Pricing KPI

\* Inventory KPI

\* Inventory status visualization

\* Dynamic pricing visualization

\* Pricing recommendations

\* Inventory replenishment alerts

\* 60-day demand forecast



\---



\## 5. MLflow Experiment Tracking



MLflow uses the local SQLite tracking database:



```text

mlflow.db

```



Start the MLflow UI:



```text

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

```



Open:



```text

http://127.0.0.1:5000

```



The Checkpoint 5 forecasting experiment contains 10 tracked runs.



\---



\## 6. Testing



Run all tests with:



```text

pytest -q

```



The test suite covers:



\* API health and endpoints

\* Forecasting functionality

\* Dynamic pricing

\* Inventory replenishment

\* Dashboard data



\---



\## 7. Production Deployment Considerations



For production deployment:



1\. Replace demonstration demand data with real SKU-level sales history.

2\. Connect inventory data to ERP/WMS systems.

3\. Add authentication and authorization.

4\. Use HTTPS.

5\. Store secrets in environment variables.

6\. Add application logging.

7\. Add model and data drift monitoring.

8\. Schedule model retraining.

9\. Validate forecasts across all production SKUs.

10\. Monitor pricing and inventory business outcomes.



\---



\## 8. Data Disclaimer



The current project uses supplied clickstream data and a derived/synthetic 20-SKU demand dataset for demonstration.



The demonstration data should not be treated as actual production retail sales or inventory data.



Production deployment requires validated historical sales, pricing, promotion, and inventory datasets.



