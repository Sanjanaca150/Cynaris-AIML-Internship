\# OmniRetail AI — Checkpoint 4



\## Integration, API Serving, Testing \& Load Testing



This checkpoint integrates the trained OmniRetail AI Random Forest model from Checkpoint 3 into a FastAPI serving layer.



\## Model



Model artifact:



`models/best\_core\_model.joblib`



Model type:



`RandomForestClassifier`



Best configuration:



\- n\_estimators = 100

\- max\_depth = 5

\- min\_samples\_split = 2

\- random\_state = 42



\## API Endpoints



\- `GET /` — API service information

\- `GET /health` — API and model health check

\- `POST /predict` — purchase prediction and purchase probability



The API runs locally at:



`http://127.0.0.1:8000`



Interactive API documentation:



`http://127.0.0.1:8000/docs`



\## Integration Testing



Integration tests were implemented using pytest and FastAPI TestClient.



Test coverage includes:



1\. Root endpoint

2\. Health endpoint

3\. Prediction endpoint

4\. Different prediction input

5\. Invalid request validation



Result:



\*\*5 passed\*\*



\## Load Testing



A 50-concurrent-request load test was implemented using Python ThreadPoolExecutor and HTTPX.



Results:



| Metric | Result |

|---|---:|

| Total Requests | 50 |

| Concurrent Requests | 50 |

| Successful Requests | 50 |

| Failed Requests | 0 |

| Total Test Time | 25.6737 seconds |

| Average Latency | 23,418.65 ms |

| Minimum Latency | 20,248.76 ms |

| Maximum Latency | 25,645.01 ms |

| Throughput | 1.95 requests/sec |



Success rate: \*\*100%\*\*



\## How to Run



Create and activate the Python virtual environment, install dependencies, and start the API:



```powershell

pip install -r requirements.txt

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

