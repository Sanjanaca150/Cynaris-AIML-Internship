\# W12D4 ML Model Monitoring Strategy



\## 1. Overview



The W12D4 ML API uses a Random Forest classifier trained on the Iris dataset and exposed through a FastAPI application.



The application is containerised using Docker and validated through automated CI/CD checks.



\## 2. Metrics to Monitor



\### API and System Metrics

\- Request count

\- Response time / latency

\- HTTP error rate

\- CPU usage

\- Memory usage

\- Container availability

\- API uptime



\### Model Performance Metrics

\- Prediction distribution

\- Accuracy when labelled data becomes available

\- Precision

\- Recall

\- F1-score

\- Prediction confidence where supported



\### Data Quality and Drift

\- Missing values

\- Invalid input values

\- Feature range changes

\- Feature distribution changes

\- Prediction distribution changes

\- Data drift between training and production data



\## 3. Alerts



Alerts should be configured for:



\- API availability failures

\- High HTTP error rate

\- Increased response latency

\- Excessive CPU or memory usage

\- Significant data drift

\- Significant prediction distribution changes

\- Model performance falling below the agreed threshold



\## 4. Retraining Triggers



A model retraining process should be considered when:



1\. Model accuracy or F1-score drops below the defined threshold.

2\. Significant feature or data distribution drift is detected.

3\. Production data characteristics differ substantially from training data.

4\. Prediction distributions change unexpectedly.

5\. A sufficient amount of new labelled data becomes available.



\## 5. Monitoring Workflow



Production monitoring should follow this process:



1\. Collect API, system, data, and model metrics.

2\. Store metrics for historical comparison.

3\. Compare production behaviour with baseline values.

4\. Generate alerts when thresholds are exceeded.

5\. Investigate the cause of the issue.

6\. Retrain and validate the model when retraining criteria are met.

7\. Deploy the validated model through the CI/CD pipeline.

8\. Continue monitoring the new model.



\## 6. Current Validation Evidence



The following checks were completed locally:



\- Ruff linting passed.

\- Pytest: 3 tests passed.

\- Docker container started successfully.

\- FastAPI `/docs` endpoint returned HTTP 200.

\- FastAPI `/health` endpoint returned HTTP 200.

\- FastAPI `/predict` endpoint returned HTTP 200.



\## 7. Future Improvements



Future versions can include:



\- Prometheus and Grafana monitoring

\- Centralised application logging

\- Automated data-drift detection

\- Model performance dashboards

\- Automated retraining pipelines

\- Model version tracking with MLflow

