\# OmniRetail AI — Model Card



\## 1. Model Overview



\*\*Project:\*\* OmniRetail AI

\*\*Purpose:\*\* End-to-end AI platform for retail demand forecasting, dynamic pricing, and inventory replenishment.



The system combines:



\* Random Forest purchase prediction

\* Prophet demand forecasting

\* Dynamic pricing rules

\* Inventory replenishment alerts

\* FastAPI serving layer

\* Streamlit manager dashboard

\* MLflow experiment tracking



\---



\## 2. Data



\### Supplied Data



The project uses the provided retail data packs:



\* `aiml\_ecommerce\_clickstream.csv`



&#x20; \* 1,000 rows

&#x20; \* 16 columns

&#x20; \* Customer/session clickstream information



\* `aiml\_product\_reviews.csv`



&#x20; \* 800 rows

&#x20; \* 11 columns

&#x20; \* Product review information



\* `aiml\_ecommerce\_tests.json`



&#x20; \* 50 test cases



\### Demand Forecasting Data



The supplied clickstream dataset does not contain historical sales quantities or actual SKU identifiers.



Therefore, for the end-to-end demonstration, a \*\*derived/synthetic 20-SKU demand dataset\*\* was generated from the supplied clickstream purchase and session activity.



Generated dataset:



\* 20 SKUs

\* 5 product categories

\* 14,560 daily records

\* Date range: January 2023 to December 2024

\* Categories:



&#x20; \* Electronics

&#x20; \* Grocery

&#x20; \* Beauty

&#x20; \* Fashion

&#x20; \* Home



This synthetic/derived dataset is clearly identified as demonstration data and should not be interpreted as production sales data.



\---



\## 3. Core Purchase Prediction Model



The core purchase prediction model is a Random Forest classifier.



\### Best Configuration



\* Model: Random Forest Classifier

\* Number of estimators: 100

\* Maximum depth: 5

\* Minimum samples split: 2

\* Random state: 42



\### Evaluation Results



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 53.50% |

| Precision | 54.89% |

| Recall    | 90.99% |

| F1 Score  | 68.47% |

| ROC-AUC   | 42.58% |



The best configuration was selected using F1 score.



Three configurations were tested and tracked with MLflow.



\---



\## 4. Demand Forecasting Model



The demand forecasting module uses Facebook Prophet.



\### Configuration



\* Growth: Linear

\* Weekly seasonality: Enabled

\* Monthly seasonality: Enabled

\* Yearly seasonality: Disabled

\* Seasonality mode: Additive

\* Changepoint prior scale: 0.01

\* Seasonality prior scale: 5.0

\* Forecast horizon: 60 days



\### Validation



For SKU-001:



\* Training rows: 668

\* Testing rows: 60

\* Forecast horizon: 60 days

\* MAPE: \*\*14.4%\*\*



The target acceptance criterion is MAPE below 15%.



\*\*Result: PASSED\*\*



\---



\## 5. Dynamic Pricing



The pricing engine generates recommended prices for 20 demonstration SKUs.



Pricing considers:



\* Base SKU price

\* Average demand

\* Forecast demand

\* Current demonstration inventory

\* Demand-to-inventory relationship



Prices are constrained to a controlled range around the base price.



\### Validation



\* SKUs tested: \*\*20\*\*

\* Pricing test: \*\*PASSED\*\*



The pricing module is a rule-based demonstration engine and does not claim to estimate causal price elasticity from real transaction data.



\---



\## 6. Inventory Replenishment



The inventory module generates replenishment alerts for 20 SKUs.



\### Alert Categories



\* HEALTHY

\* REORDER

\* CRITICAL



\### Current Demonstration Results



\* HEALTHY: 12 SKUs

\* REORDER: 4 SKUs

\* CRITICAL: 4 SKUs



Total SKUs analyzed: \*\*20\*\*



The reorder quantity is calculated using forecast demand and the demonstration inventory level.



\---



\## 7. MLflow Experiment Tracking



MLflow is used to track forecasting experiments.



Experiment name:



`OmniRetail\_Checkpoint\_5\_Forecasting`



A total of \*\*10 forecasting runs\*\* were logged.



Tracked information includes:



\* SKU

\* Model type

\* Changepoint prior scale

\* Seasonality prior scale

\* Training rows

\* Testing rows

\* MAPE



This satisfies the requirement for at least 10 tracked experiments.



\---



\## 8. Bias Audit



A group-level performance audit was performed using the available operational attributes:



\* Category

\* Device

\* City



The audit calculates:



\* Accuracy

\* F1 score

\* Number of samples



The supplied dataset does not contain protected demographic attributes such as race, gender, religion, or age.



Therefore, the audit does \*\*not\*\* make a protected-class fairness claim.



The audit results are stored in:



`docs/bias\_audit.json`



Because some groups may contain relatively small numbers of samples, group-level metrics should be interpreted as diagnostic rather than definitive fairness measurements.



\---



\## 9. Limitations



The current system has the following limitations:



1\. The supplied data does not contain real SKU-level historical sales quantities.

2\. The 20-SKU demand dataset is derived/synthetic demonstration data.

3\. The forecast MAPE of 14.4% was validated on SKU-001 and should not be interpreted as the accuracy of every SKU.

4\. The pricing engine is heuristic and does not estimate causal price elasticity.

5\. Inventory levels are demonstration values rather than live ERP/WMS inventory.

6\. The system does not connect to a live retail database or ERP.

7\. The dashboard is intended for demonstration and evaluation.

8\. No protected demographic attributes were provided, so protected-class fairness cannot be fully assessed.

9\. Production deployment would require monitoring for data drift, forecast degradation, pricing effects, and inventory performance.



\---



\## 10. Deployment Guide



\### FastAPI Backend



Start the API with:



```text

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



Available endpoints:



\* `/`

\* `/health`

\* `/forecast/{sku\_id}`

\* `/pricing`

\* `/inventory`

\* `/dashboard-data`



Interactive API documentation:



`http://127.0.0.1:8000/docs`



\### Streamlit Dashboard



Start the manager dashboard with:



```text

streamlit run app/dashboard.py

```



The dashboard displays:



\* Forecast MAPE

\* Total SKUs

\* Pricing results

\* Inventory status

\* Pricing recommendations

\* Inventory alerts

\* Demand forecast charts



\### MLflow



Start MLflow using:



```text

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

```



Then open:



`http://127.0.0.1:5000`



\---



\## 11. Production Considerations



Before production deployment, the following improvements are recommended:



\* Connect to real SKU-level sales history.

\* Integrate ERP/WMS inventory data.

\* Use actual price and promotion history.

\* Estimate price elasticity using historical transactions or controlled experiments.

\* Validate forecasting across all SKUs.

\* Add automated model retraining.

\* Monitor data drift and model performance.

\* Add authentication and authorization to the API.

\* Add production logging and monitoring.

\* Conduct a formal fairness assessment when protected attributes are legally and ethically available.



\---



\## 12. Conclusion



OmniRetail AI provides a complete demonstration pipeline covering demand forecasting, dynamic pricing, inventory replenishment, model serving, experiment tracking, and manager visualization.



The demonstrated acceptance results are:



\* \*\*Demand forecasting MAPE: 14.4% — PASSED\*\*

\* \*\*Pricing tested: 20 SKUs — PASSED\*\*

\* \*\*Inventory analyzed: 20 SKUs — PASSED\*\*

\* \*\*MLflow experiments: 10 — PASSED\*\*

\* \*\*Manager dashboard: Implemented\*\*

\* \*\*FastAPI serving layer: Implemented\*\*

\* \*\*Bias audit: Completed\*\*



The system is suitable as an end-to-end demonstration and evaluation project. Production use would require real SKU-level sales, inventory, pricing, and operational data.



