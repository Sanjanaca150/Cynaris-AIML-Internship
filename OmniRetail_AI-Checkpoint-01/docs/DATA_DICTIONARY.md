\# OmniRetail AI — Data Dictionary



\## 1. Product Reviews Dataset



\*\*File:\*\* `data/aiml\_product\_reviews.csv`



\*\*Records:\*\* 800



\*\*Columns:\*\* 11



This dataset contains customer reviews and product-level feedback information.



| Column                | Description                                             | Type        |

| --------------------- | ------------------------------------------------------- | ----------- |

| `review\_id`           | Unique identifier for each review                       | String      |

| `product\_id`          | Identifier of the reviewed product                      | String      |

| `category`            | Product category                                        | Categorical |

| `rating`              | Customer rating                                         | Integer     |

| `review\_date`         | Date on which the review was submitted                  | Date        |

| `verified\_purchase`   | Indicates whether the reviewer made a verified purchase | Binary      |

| `helpful\_votes`       | Number of helpful votes received by the review          | Integer     |

| `review\_length\_chars` | Number of characters in the review                      | Integer     |

| `contains\_image`      | Indicates whether the review contains an image          | Binary      |

| `sentiment`           | Sentiment label associated with the review              | Categorical |

| `review\_text`         | Textual customer review                                 | Text        |



\### Planned Usage



The review dataset will be used for:



\* Customer feedback analytics

\* Product/category sentiment analysis

\* Rating analysis

\* Review-volume analysis

\* Product quality insights

\* Helpful-review analysis



\---



\## 2. E-Commerce Evaluation Dataset



\*\*File:\*\* `data/aiml\_ecommerce\_tests.json`



\*\*Records:\*\* 50



\*\*Model target:\*\* `purchase\_prediction`



Each test case contains an identifier, input features, an expected label, and the model target.



\### Test Case Fields



| Field                   | Description                                  | Type        |

| ----------------------- | -------------------------------------------- | ----------- |

| `test\_id`               | Unique test-case identifier                  | String      |

| `input.category`        | Product category associated with the session | Categorical |

| `input.user\_device`     | Device used by the user                      | Categorical |

| `input.session\_mins`    | Session duration in minutes                  | Numeric     |

| `input.prior\_purchases` | Number of previous purchases                 | Integer     |

| `expected\_label`        | Expected purchase outcome                    | Binary      |

| `model\_target`          | Target model/use case                        | String      |



\### Planned Usage



The dataset will be used to evaluate the purchase-prediction component.



Evaluation metrics will include:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Confusion matrix



\---



\## 3. Clickstream Dataset



\*\*Expected file:\*\* `data/aiml\_ecommerce\_clickstream.csv`



\*\*Status:\*\* Pending incorporation.



The clickstream dataset is expected to provide event/session-level information required for the demand forecasting and broader e-commerce behaviour pipeline.



After the dataset is available, its schema will be inspected before feature engineering or modelling.



\### Planned Usage



Depending on the actual available fields, the clickstream dataset may support:



\* Demand forecasting

\* Session behaviour analysis

\* Conversion modelling

\* Dynamic pricing signals

\* Inventory planning

\* Customer behaviour features



No assumptions about unavailable fields will be treated as confirmed data characteristics until the dataset is inspected.



\---



\## Data Quality Plan



Before model development, the datasets will undergo:



1\. Schema validation

2\. Missing-value analysis

3\. Duplicate detection

4\. Data-type validation

5\. Date parsing

6\. Categorical-value validation

7\. Numerical range checks

8\. Target-label validation

9\. Outlier analysis where appropriate

10\. Feature leakage checks



\---



\## Data Integration Strategy



The datasets will be processed independently where appropriate and combined through shared business entities such as:



\* Product

\* Category

\* Session

\* Date



The final integration strategy will be determined after the clickstream schema is available.



\---



\## Data Governance



The project will avoid using personally identifiable customer information.



Raw datasets will be treated as input data, while transformed datasets and model-ready features will be generated through reproducible preprocessing steps.



