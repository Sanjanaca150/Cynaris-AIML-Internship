# Production ML Model Monitoring Strategy



## 1. Objective



The objective of this monitoring strategy is to maintain the reliability, availability, and performance of the ML API after deployment.



The monitoring strategy covers:



- API and system health

- Model performance

- Input data quality

- Data drift

- Alerts

- Model retraining triggers



## 2. API and System Metrics



The following production metrics should be monitored:



- API uptime and availability

- Number of requests

- Request latency

- HTTP 4xx and 5xx errors

- Container health

- CPU usage

- Memory usage

- Application failures



These metrics help identify service availability and infrastructure problems.



## 3. Model Performance Metrics



The following model metrics should be tracked when labeled production data becomes available:



- Accuracy

- Precision

- Recall

- F1-score

- Prediction distribution

- Model confidence, when available



Production performance should be compared with validation performance to identify degradation.



## 4. Data Quality Monitoring



Production input data should be checked for:



- Missing values

- Invalid values

- Unexpected feature ranges

- Malformed requests

- Unexpected categorical values

- Duplicate or corrupted records



Data-quality problems should be investigated before using affected data for retraining.



## 5. Data Drift Monitoring



Data drift occurs when the distribution of production input data changes compared with the training or validation data.



The monitoring system should track:



- Feature means and standard deviations

- Feature distributions

- Categorical value frequencies

- Missing-value rates

- Significant changes in input distributions



Persistent and significant drift should trigger an investigation.



A single anomalous observation should not automatically trigger model retraining.



## 6. Alert Conditions



Alerts should be configured for important production problems such as:



- API becoming unavailable

- Increased HTTP 5xx errors

- High request latency

- Container failure

- Excessive CPU or memory usage

- High invalid-input rate

- Significant data drift

- Unexpected prediction distribution

- Model performance falling below the defined threshold



## 7. Model Retraining Triggers



Model retraining should be considered when:



- Model performance consistently falls below the defined threshold

- Significant data drift persists

- Production data distribution changes substantially

- Unexpected prediction behavior is observed

- Sufficient new labeled data becomes available

- Business or application requirements change



Retraining should not be triggered solely by one unusual prediction or one isolated data point.



## 8. Retraining Workflow



The proposed retraining workflow is:



1\. Collect new production data.

2\. Validate data quality.

3\. Check for data drift.

4\. Prepare and preprocess the data.

5\. Train a candidate model.

6\. Evaluate the candidate model.

7\. Compare it with the current production model.

8\. Track the experiment using MLflow.

9\. Register the candidate model if it satisfies the required criteria.

10\. Run automated tests.

11\. Deploy after validation.

12\. Continue monitoring the new production model.



## 9. Tools Used



The W12D3 project uses the following technologies:



- Python

- FastAPI

- Scikit-learn

- Docker

- Pytest

- Ruff

- Git

- GitHub

- GitHub Actions

- Docker Hub

- MLflow

- MLOps practices



## 10. Monitoring Review



Monitoring metrics and thresholds should be reviewed regularly.



Operational alerts should be investigated before taking corrective action, and retraining decisions should be based on sustained evidence of model or data degradation.


