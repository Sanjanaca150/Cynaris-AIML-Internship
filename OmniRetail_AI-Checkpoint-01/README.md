# OmniRetail AI



## AI-Powered E-Commerce Intelligence Platform



OmniRetail AI is an AI/ML-powered e-commerce intelligence platform designed to support retail decision-making through demand forecasting, dynamic pricing, inventory replenishment, purchase prediction, and customer feedback analytics.



The project follows an end-to-end machine learning architecture covering data ingestion, preprocessing, feature engineering, model training, evaluation, experiment tracking, API serving, and dashboard-based business insights.



---



## Project Objectives



The platform is designed to address five major retail intelligence use cases:



1\. **Demand Forecasting**



   * Forecast future product/category demand.

   * Evaluate forecasting performance using metrics such as MAE, RMSE and MAPE.

   * Target: achieve a MAPE below 15% when supported by the available demand data.



2\. **Dynamic Pricing**



   * Use demand and product/customer signals to recommend pricing adjustments.

   * Evaluate pricing recommendations using demand-response and business-oriented metrics.



3\. **Inventory \& Replenishment**



   * Estimate inventory requirements from demand forecasts.

   * Generate replenishment recommendations and identify potential stock-out risks.



4\. **Purchase Prediction**



   * Predict whether an e-commerce session is likely to result in a purchase.

   * Use the supplied e-commerce test cases for model validation.



5\. **Customer Feedback Analytics**



   * Analyze product reviews, ratings and sentiment.

   * Identify category/product-level customer feedback patterns.



---



## Available Datasets



### 1. Product Reviews



File:



`data/aiml_product_reviews.csv`



Current inspection:



* Records: **800**

* Columns: **11**



Available fields include:



* `review_id`

* `product_id`

* `category`

* `rating`

* `review_date`

* `verified_purchase`

* `helpful_votes`

* `review_length_chars`

* `contains_image`

* `sentiment`

* `review_text`



This dataset will support customer feedback, sentiment and product-level analytics.



### 2. E-Commerce Test Cases



File:



`data/aiml_ecommerce_tests.json`



Current inspection:



* Test cases: **50**

* Target: `purchase_prediction`



The test cases contain:



* category

* user device

* session duration

* prior purchases

* expected purchase label



These cases will be used to evaluate the purchase-prediction component.



### 3. Clickstream Dataset



File expected:



`data/aiml_ecommerce_clickstream.csv`



The clickstream dataset is pending incorporation into the repository. It will provide the event/session-level information required to build and validate the demand forecasting, pricing and inventory components.



No synthetic performance results are claimed before the clickstream data is incorporated.



---



## Proposed Architecture



```text

                    E-COMMERCE DATA SOURCES

                              |

          +-------------------+-------------------+

          |                   |                   |

     Clickstream          Reviews            Test Cases

          |                   |                   |

          +-------------------+-------------------+

                              |

                       Data Ingestion

                              |

                     Data Validation

                              |

                    Data Preprocessing

                              |

                    Feature Engineering

                              |

             +----------------+----------------+

             |                |                |

       Demand Model     Purchase Model    Review Analytics

             |                |                |

             +----------------+----------------+

                              |

                   Business Intelligence

                    +---------+---------+

                    |                   |

             Dynamic Pricing    Inventory/Replenishment

                    |                   |

                    +---------+---------+

                              |

                       Model Evaluation

                              |

                           MLflow

                              |

                         API Layer

                              |

                         Dashboard

```



---



## Technology Stack



* Python

* Pandas

* NumPy

* Scikit-learn

* FastAPI

* MLflow

* Matplotlib

* Jupyter Notebook

* Git

* GitHub

* Docker

* Streamlit/dashboard layer



Additional libraries will be selected according to the requirements of each model and pipeline component.



---



## Evaluation Strategy



### Demand Forecasting



* MAE

* RMSE

* MAPE



Primary target:



**MAPE < 15%**, subject to the quality and coverage of the demand dataset.



### Purchase Prediction



* Accuracy

* Precision

* Recall

* F1-score

* Confusion matrix



The supplied 50 e-commerce test cases will be used as an evaluation dataset.



### Review Analytics



* Sentiment distribution

* Average rating

* Category-level sentiment

* Review volume

* Helpful-vote analysis



### Inventory



* Stock-out risk

* Replenishment recommendation accuracy

* Forecast-driven inventory requirements



### Dynamic Pricing



* Demand response

* Revenue-oriented evaluation

* Price recommendation consistency



---



## Project Structure



```text

OmniRetail_AI/

|

├── app/

│   ├── __init__.py

│   ├── config.py

│   └── data_loader.py

|

├── data/

│   ├── aiml_product_reviews.csv

│   ├── aiml_ecommerce_tests.json

│   └── aiml_ecommerce_clickstream.csv

|

├── dashboard/

|

├── docs/

│   ├── ARCHITECTURE.md

│   ├── DATA_DICTIONARY.md

│   └── RESEARCH.md

|

├── models/

|

├── notebooks/

|

├── tests/

│   └── __init__.py

|

├── .gitignore

├── README.md

└── requirements.txt

```



---



## Development Approach



The project will be developed incrementally:



1\. Research and architecture

2\. Data ingestion and validation

3\. Exploratory data analysis

4\. Feature engineering

5\. Demand forecasting

6\. Purchase prediction

7\. Dynamic pricing

8\. Inventory/replenishment

9\. Model evaluation

10\. MLflow experiment tracking

11\. API integration

12\. Dashboard development

13\. Dockerisation and final integration



---



## Current Checkpoint



### Checkpoint 1 — Research + Architecture Design



Completed activities:



* Defined the OmniRetail AI business problem.

* Identified five major AI/ML use cases.

* Created the initial project structure.

* Added and inspected the available product-review dataset.

* Added and inspected the e-commerce evaluation dataset.

* Defined the proposed end-to-end architecture.

* Defined model evaluation metrics.

* Planned integration of the clickstream dataset.



### Data Status



The product reviews and e-commerce test-case datasets are currently available.



The clickstream dataset will be incorporated once available. Data-dependent model training and performance claims will be completed after the relevant data has been integrated.



---



## Expected Outcome



OmniRetail AI aims to provide a unified AI-powered retail intelligence system that converts e-commerce data into actionable recommendations for:



* Demand planning

* Pricing decisions

* Inventory management

* Purchase conversion

* Customer experience improvement



