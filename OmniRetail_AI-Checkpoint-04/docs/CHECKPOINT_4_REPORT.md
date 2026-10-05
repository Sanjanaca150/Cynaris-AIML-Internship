# OmniRetail AI — Checkpoint 4

## Integration, API Serving, Testing & Load Testing Report

### 1. Objective

The objective of this checkpoint was to integrate the trained OmniRetail AI core model into a serving layer, implement integration tests, perform load testing with 50 concurrent requests, and document the observed latency and throughput.

The Random Forest model selected during Checkpoint 3 was reused as the prediction model for the serving layer.

---

### 2. Model Integration

The best-performing Random Forest model from Checkpoint 3 was integrated into a FastAPI application.

**Model artifact:**

`models/best_core_model.joblib`

**Model type:**

`RandomForestClassifier`

**Selected configuration:**

* `n_estimators = 100`
* `max_depth = 5`
* `min_samples_split = 2`
* `random_state = 42`

The saved Joblib model was successfully loaded by the FastAPI application and used to generate purchase predictions.

---

### 3. FastAPI Serving Layer

A FastAPI-based serving layer was implemented to expose the trained model through REST API endpoints.

The application was started using Uvicorn at:

`http://127.0.0.1:8000`

### API Endpoints

#### GET `/`

Provides basic information about the OmniRetail AI serving service.

Expected response:

```json
{
    "project": "OmniRetail AI",
    "checkpoint": "Checkpoint 4",
    "service": "Core Model Prediction API",
    "status": "running"
}
```

#### GET `/health`

Checks the health of the API and confirms that the trained model is loaded successfully.

The endpoint returned HTTP 200 successfully.

#### POST `/predict`

Accepts the model input features and returns a purchase prediction and purchase probability.

The endpoint accepts the following nine features:

* `product_views`
* `add_to_cart`
* `session_duration_mins`
* `return_visitor`
* `discount_applied`
* `recommendation_clicked`
* `city`
* `device`
* `category`

The response contains:

* `prediction`
* `purchase_probability`
* `model`

Interactive API documentation was also verified through:

`http://127.0.0.1:8000/docs`

---

### 4. Integration Testing

Integration tests were implemented using:

* Python
* Pytest
* FastAPI TestClient
* HTTPX

The test suite verifies the main API functionality, including:

1. Root endpoint response
2. Health endpoint response
3. Prediction endpoint functionality
4. Prediction with different input values
5. Invalid prediction request validation

### Test Result

**5 passed**

The complete integration test execution completed successfully.

A Starlette deprecation warning related to the current HTTPX/TestClient compatibility was displayed, but it did not affect test execution.

Final result:

```text
5 passed, 1 warning in 8.64s
```

---

### 5. Concurrent Load Testing

A concurrent load-testing script was implemented using Python `ThreadPoolExecutor` with 50 workers and HTTPX.

The test sent requests concurrently to the `/predict` endpoint.

### Load Test Configuration

* Total requests: **50**
* Concurrent requests: **50**
* Endpoint tested: **POST `/predict`**
* Request method: **HTTP POST**
* Test environment: **Local FastAPI/Uvicorn server**

---

### 6. Load Test Results

The final load test produced the following observed results:

| Metric              |   Observed Result |
| ------------------- | ----------------: |
| Total Requests      |                50 |
| Concurrent Requests |                50 |
| Successful Requests |                50 |
| Failed Requests     |                 0 |
| Total Test Time     |   24.3702 seconds |
| Average Latency     |      21,975.12 ms |
| Minimum Latency     |      18,432.96 ms |
| Maximum Latency     |      24,345.17 ms |
| Throughput          | 2.05 requests/sec |

### Load Test Result

**50-concurrent-request load test: PASSED**

The API successfully processed all 50 concurrent requests.

* Success rate: **100%**
* Failure rate: **0%**
* Average latency: **21.98 seconds**
* Minimum latency: **18.43 seconds**
* Maximum latency: **24.35 seconds**
* Throughput: **2.05 requests/sec**

The observed latency and throughput represent the performance measured in the local development environment during t
