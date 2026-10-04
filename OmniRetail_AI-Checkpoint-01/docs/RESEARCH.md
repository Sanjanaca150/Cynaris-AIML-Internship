\# OmniRetail AI — Research Review



\## 1. Purpose



The purpose of this research review is to identify established approaches relevant to the OmniRetail AI platform.



The review focuses on:



\* Demand forecasting

\* Retail/e-commerce recommendation and prediction

\* Dynamic pricing

\* Inventory optimisation

\* Machine learning evaluation and deployment



The selected research provides a foundation for the proposed architecture while allowing the final implementation to be adapted to the available project datasets.



\---



\## 2. Research Area 1 — Demand Forecasting



\### Reference



\*\*DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks\*\*



DeepAR is a probabilistic forecasting approach designed to learn forecasting patterns from multiple related time series.



\### Relevance to OmniRetail AI



E-commerce demand can contain temporal patterns across products and categories. A forecasting architecture should therefore consider:



\* Historical demand

\* Time-based patterns

\* Multiple related products

\* Uncertainty in future demand



\### Application



The project will first establish a baseline using simpler forecasting/regression approaches. More advanced approaches can be evaluated if the available clickstream/demand data contains sufficient temporal information.



\### Expected Benefit



A strong forecasting component can provide the demand signal required by the downstream:



\* Inventory module

\* Replenishment module

\* Dynamic pricing module



\---



\## 3. Research Area 2 — E-Commerce Behaviour Prediction



\### Reference



\*\*Retailrocket Recommender System Dataset / E-Commerce Behaviour Modelling\*\*



E-commerce behaviour datasets are commonly used to model interactions between users, sessions and products.



Important behavioural signals can include:



\* Sessions

\* Product interactions

\* Event sequences

\* Product/category information

\* Purchase events



\### Relevance to OmniRetail AI



The supplied purchase-prediction evaluation cases contain session-level information such as:



\* Category

\* Device

\* Session duration

\* Prior purchases



These variables provide a starting point for building a binary purchase-prediction model.



The future clickstream dataset can extend the feature set with event-level behavioural information.



\### Application



Candidate classification models will be compared using:



\* Accuracy

\* Precision

\* Recall

\* F1-score



The final choice will be based on validation performance rather than model complexity alone.



\---



\## 4. Research Area 3 — Dynamic Pricing



\### Research Direction



Dynamic pricing systems generally use demand, customer behaviour, product characteristics and market conditions to determine appropriate prices.



A practical implementation should consider both predictive performance and business constraints.



\### Relevance to OmniRetail AI



The proposed pricing component will consume signals from:



\* Demand forecasting

\* Customer behaviour

\* Product/category information

\* Inventory conditions



Instead of allowing unrestricted price changes, the system will use bounded recommendations.



\### Proposed Strategy



The initial implementation will follow a controlled recommendation workflow:



```text

Demand Signal

&#x20;     |

&#x20;     v

Pricing Model / Rule Layer

&#x20;     |

&#x20;     v

Candidate Price

&#x20;     |

&#x20;     v

Business Constraints

&#x20;     |

&#x20;     v

Recommended Price

```



Potential constraints include:



\* Minimum price

\* Maximum price

\* Maximum percentage change

\* Inventory considerations



This makes the pricing component safer and easier to evaluate.



\---



\## 5. Research Area 4 — Inventory and Replenishment



\### Research Direction



Inventory planning commonly combines demand forecasts with inventory policies such as reorder points and safety stock.



A forecasting model alone is not sufficient for operational inventory decisions.



\### Relevance to OmniRetail AI



The proposed inventory module will use forecast demand to calculate replenishment-related indicators.



Possible inputs include:



\* Forecast demand

\* Current inventory

\* Safety-stock requirements

\* Reorder point

\* Lead time



The exact implementation will depend on the inventory-related fields available in the project datasets.



\### Proposed Workflow



```text

Historical Demand

&#x20;      |

&#x20;      v

Demand Forecast

&#x20;      |

&#x20;      v

Inventory Requirement

&#x20;      |

&#x20;      v

Stock-out Risk

&#x20;      |

&#x20;      v

Replenishment Recommendation

```



\---



\## 6. Research Area 5 — Machine Learning Operations



\### Research Direction



Production ML systems require more than model training.



Important components include:



\* Experiment tracking

\* Model versioning

\* Reproducibility

\* Validation

\* Deployment

\* Monitoring



\### Relevance to OmniRetail AI



The project architecture includes MLflow so that model experiments can be tracked consistently.



The pipeline will record appropriate:



\* Parameters

\* Metrics

\* Model versions

\* Artifacts



FastAPI will provide the serving interface after model development.



\---



\## 7. Research-to-Implementation Mapping



| Research Area                         | OmniRetail AI Component | Planned Implementation                         |

| ------------------------------------- | ----------------------- | ---------------------------------------------- |

| Probabilistic/time-series forecasting | Demand Forecasting      | Baseline + candidate forecasting models        |

| E-commerce behaviour modelling        | Purchase Prediction     | Classification models                          |

| Demand-based pricing                  | Dynamic Pricing         | Predictive signal + constrained recommendation |

| Forecast-driven inventory             | Inventory               | Reorder and stock-risk logic                   |

| MLOps                                 | Experiment Management   | MLflow tracking                                |

| Model serving                         | API                     | FastAPI                                        |



\---



\## 8. Research Findings



The research supports several architectural decisions:



1\. Demand forecasting should be treated as a dedicated modelling problem rather than a simple dashboard calculation.

2\. E-commerce session behaviour provides useful signals for purchase prediction.

3\. Dynamic pricing should incorporate business constraints in addition to model predictions.

4\. Inventory decisions should use demand forecasts and operational constraints together.

5\. Production ML systems benefit from experiment tracking, model versioning and reproducible evaluation.

6\. The final model selection should be driven by validation results and business requirements.



\---



\## 9. References



1\. Salinas, D., Flunkert, V., Gasthaus, J., \& Januschowski, T.

&#x20;  \*\*DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.\*\*

&#x20;  International Journal of Forecasting, 2020.



2\. Retailrocket Recommender System Dataset.

&#x20;  Public e-commerce behavioural interaction dataset used for recommender-system and user-behaviour research.



3\. MLflow Documentation.

&#x20;  Open-source platform for experiment tracking, model management and ML lifecycle workflows.



\---



\## 10. Research Limitations



The current research review provides architectural guidance rather than claiming that any particular research model will be optimal for the project.



Final model selection will depend on:



\* Dataset size

\* Available features

\* Temporal coverage

\* Target definition

\* Data quality

\* Validation performance

\* Computational requirements



The clickstream dataset is not yet incorporated, so demand forecasting and related downstream modelling decisions will be finalized after its schema and coverage are inspected.



