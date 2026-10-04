\# OmniRetail AI — System Architecture



\## 1. Overview



OmniRetail AI is an end-to-end AI/ML platform for e-commerce intelligence.



The system is designed to transform retail data into predictive insights and operational recommendations across:



\* Demand forecasting

\* Purchase prediction

\* Dynamic pricing

\* Inventory and replenishment

\* Customer feedback analytics



The architecture is modular so that individual models can be developed, evaluated and deployed independently while sharing common data-processing and monitoring components.



\---



\## 2. High-Level Architecture



```text

+-------------------------------------------------------------+

|                     E-COMMERCE DATA SOURCES                 |

+-------------------------------------------------------------+

&#x20;            |                  |                  |

&#x20;            v                  v                  v

&#x20;     Clickstream          Product Reviews     Test Cases

&#x20;            |                  |                  |

&#x20;            +------------------+------------------+

&#x20;                               |

&#x20;                               v

&#x20;                   +-------------------------+

&#x20;                   |    Data Ingestion       |

&#x20;                   +-------------------------+

&#x20;                               |

&#x20;                               v

&#x20;                   +-------------------------+

&#x20;                   | Data Validation \&       |

&#x20;                   | Quality Checks           |

&#x20;                   +-------------------------+

&#x20;                               |

&#x20;                               v

&#x20;                   +-------------------------+

&#x20;                   | Preprocessing \&          |

&#x20;                   | Feature Engineering      |

&#x20;                   +-------------------------+

&#x20;                               |

&#x20;             +-----------------+-----------------+

&#x20;             |                 |                 |

&#x20;             v                 v                 v

&#x20;      Demand Forecast    Purchase Prediction  Review Analytics

&#x20;             |                 |                 |

&#x20;             +-----------------+-----------------+

&#x20;                               |

&#x20;                               v

&#x20;                   +-------------------------+

&#x20;                   | Business Intelligence   |

&#x20;                   +-------------------------+

&#x20;                        |               |

&#x20;                        v               v

&#x20;                 Dynamic Pricing   Inventory \&

&#x20;                                   Replenishment

&#x20;                        |               |

&#x20;                        +-------+-------+

&#x20;                                |

&#x20;                                v

&#x20;                   +-------------------------+

&#x20;                   | Model Evaluation         |

&#x20;                   +-------------------------+

&#x20;                                |

&#x20;                                v

&#x20;                   +-------------------------+

&#x20;                   | MLflow Experiment        |

&#x20;                   | Tracking \& Model         |

&#x20;                   | Management               |

&#x20;                   +-------------------------+

&#x20;                                |

&#x20;                                v

&#x20;                   +-------------------------+

&#x20;                   | FastAPI Service Layer    |

&#x20;                   +-------------------------+

&#x20;                                |

&#x20;                                v

&#x20;                   +-------------------------+

&#x20;                   | Business Dashboard       |

&#x20;                   +-------------------------+

```



\---



\## 3. Data Layer



\### Product Reviews



The product-review dataset contains 800 records and 11 fields.



It provides:



\* Product identifiers

\* Categories

\* Ratings

\* Review dates

\* Verified-purchase information

\* Helpful votes

\* Review text

\* Sentiment labels



This data will primarily support customer feedback and product analytics.



\### E-Commerce Test Cases



The evaluation dataset contains 50 test cases for the `purchase\_prediction` target.



The available inputs include:



\* Category

\* User device

\* Session duration

\* Prior purchases



The expected labels provide a basis for evaluating purchase-prediction behaviour.



\### Clickstream



The clickstream dataset is currently pending.



It will be integrated after its schema is inspected. The dataset is expected to become an important source for event/session-level modelling.



\---



\## 4. Data Processing Layer



The data-processing layer will contain four stages.



\### 4.1 Ingestion



Responsibilities:



\* Load CSV and JSON files

\* Validate file availability

\* Parse structured data

\* Standardize column names where necessary



\### 4.2 Validation



Checks will include:



\* Required-column validation

\* Data-type validation

\* Missing values

\* Duplicate records

\* Invalid categories

\* Invalid numerical values

\* Invalid dates

\* Target-label validation



\### 4.3 Preprocessing



Potential operations include:



\* Missing-value handling

\* Categorical encoding

\* Numerical scaling where required

\* Date feature extraction

\* Text preprocessing

\* Duplicate removal



\### 4.4 Feature Engineering



Features will be created according to the model.



Examples include:



\*\*Purchase Prediction\*\*



\* Session duration

\* Prior purchases

\* Device

\* Category



\*\*Demand Forecasting\*\*



\* Time-based features

\* Historical demand

\* Category/product trends

\* Event activity



\*\*Review Analytics\*\*



\* Rating

\* Sentiment

\* Review length

\* Helpful votes

\* Category



\---



\## 5. Machine Learning Layer



\### 5.1 Demand Forecasting



The demand forecasting component will estimate future demand at an appropriate product/category/time granularity.



Candidate approaches may include:



\* Baseline statistical forecasting

\* Random Forest / Gradient Boosting regression

\* Time-series models where appropriate



The final model will be selected based on validation performance and data characteristics.



Primary metrics:



\* MAE

\* RMSE

\* MAPE



Target:



\*\*MAPE < 15%\*\*, if supported by the available demand data.



\---



\### 5.2 Purchase Prediction



The purchase-prediction model will classify whether a user/session is likely to result in a purchase.



Candidate models include:



\* Logistic Regression

\* Decision Tree

\* Random Forest

\* Gradient Boosting



The supplied 50 test cases will be used for evaluation.



Metrics:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Confusion matrix



\---



\### 5.3 Customer Feedback Analytics



The review analytics component will analyze customer feedback.



Planned outputs:



\* Sentiment distribution

\* Average rating

\* Category-level sentiment

\* Review volume

\* Helpful-vote patterns

\* Product-level feedback



The existing sentiment field can be used for initial analytics, while the review text can support future NLP enhancements.



\---



\## 6. Business Intelligence Layer



\### Dynamic Pricing



The pricing component will use available demand and customer-behaviour signals to recommend price adjustments.



The architecture separates pricing logic from the predictive models so pricing rules can be modified without retraining every model.



Possible outputs:



```text

Product

Current Price

Demand Signal

Recommended Price

Price Change

Confidence

```



Pricing recommendations will be evaluated using business-oriented measures such as demand response and revenue impact.



\---



\### Inventory and Replenishment



Inventory recommendations will be driven primarily by demand forecasts.



Possible outputs:



```text

Product

Forecast Demand

Current Inventory

Safety Stock

Reorder Point

Recommended Reorder Quantity

Stock-out Risk

```



The final calculation will depend on the inventory-related information available in the project datasets.



\---



\## 7. Model Evaluation Layer



Every trained model will have a defined evaluation process.



The evaluation layer will:



1\. Split data appropriately.

2\. Train baseline models.

3\. Train candidate models.

4\. Compare metrics.

5\. Record experiments.

6\. Select the best validated model.

7\. Store evaluation results.



Time-dependent datasets will use time-aware validation where appropriate to reduce temporal leakage.



\---



\## 8. MLflow Layer



MLflow will be used for experiment tracking and model management.



Tracked information may include:



\* Parameters

\* Metrics

\* Model versions

\* Dataset/version information

\* Training runs

\* Artifacts



The objective is to make model experiments reproducible and comparable.



\---



\## 9. API Layer



FastAPI will provide an interface between trained models and the application/dashboard.



Potential endpoints include:



```text

GET  /health



POST /predict/purchase



POST /forecast/demand



POST /pricing/recommend



POST /inventory/recommend



GET  /analytics/reviews

```



The endpoints will be implemented as the corresponding models become available.



\---



\## 10. Dashboard Layer



The dashboard will provide business-facing views rather than exposing raw model internals.



Planned dashboard sections:



\### Executive KPIs



\* Demand forecast

\* Predicted purchases

\* Revenue-related indicators

\* Inventory risk

\* Customer sentiment



\### Demand



\* Forecast vs actual demand

\* Product/category trends

\* Forecast error



\### Pricing



\* Current price

\* Recommended price

\* Demand signal

\* Expected business impact



\### Inventory



\* Stock-out risk

\* Reorder recommendations

\* Forecast demand



\### Customer Feedback



\* Rating distribution

\* Sent



