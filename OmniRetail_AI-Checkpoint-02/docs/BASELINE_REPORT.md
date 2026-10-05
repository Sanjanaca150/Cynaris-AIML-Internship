\# OmniRetail AI — Checkpoint 2 Baseline Report



\## 1. Overview



This report documents the data ingestion, preprocessing pipeline, baseline machine learning model, evaluation benchmark, and experiment tracking completed for Checkpoint 2 of the OmniRetail AI project.



The objective of this checkpoint was to build a complete data pipeline, train a baseline model, establish measurable evaluation metrics, and prepare the foundation for subsequent model improvement.



\---



\## 2. Dataset



The primary dataset used for baseline model development was:



`aiml\_ecommerce\_clickstream.csv`



\### Dataset statistics



\* Total records: 1,000

\* Total columns: 16

\* Training records: 800

\* Testing records: 200

\* Target variable: `label\_purchased`

\* Missing values: `search\_query` contained missing values



Additional project datasets were also included in the pipeline:



\* `aiml\_product\_reviews.csv` — 800 records

\* `aiml\_ecommerce\_tests.json` — 50 test cases



\---



\## 3. Data Ingestion



The data ingestion layer was implemented using Python and pandas.



The pipeline provides dedicated loaders for:



\* Clickstream CSV data

\* Product review CSV data

\* E-commerce JSON test cases



The ingestion process was validated successfully.



\---



\## 4. Data Preprocessing



The clickstream data was cleaned before model training.



\### Cleaning operations



\* Converted the `date` column to datetime format.

\* Removed duplicate sessions using `session\_id`.

\* Filled missing `search\_query` values with `no\_search`.

\* Validated the required model columns.



\### Feature selection



The baseline model uses nine input features.



\#### Numerical features



\* `product\_views`

\* `add\_to\_cart`

\* `session\_duration\_mins`

\* `return\_visitor`

\* `discount\_applied`

\* `recommendation\_clicked`



\#### Categorical features



\* `city`

\* `device`

\* `category`



The following fields were intentionally excluded from the baseline model:



\* `session\_id`

\* `user\_id`

\* `purchase\_made`

\* `order\_value\_inr`

\* `search\_query`



Identifiers were excluded because they do not provide reliable predictive information.



`purchase\_made` and `order\_value\_inr` were excluded to reduce the risk of target leakage because they can contain information associated with the purchase outcome.



`search\_query` was not used as a model feature because it is free-text data with substantial missingness and requires additional text processing.



\---



\## 5. Baseline Model



A Logistic Regression classifier was selected as the baseline model.



The model was implemented as a scikit-learn pipeline containing:



1\. Numerical preprocessing

2\. Categorical preprocessing

3\. One-hot encoding

4\. Logistic Regression classifier



\### Training configuration



\* Model: Logistic Regression

\* Test size: 20%

\* Training size: 80%

\* Random state: 42

\* Maximum iterations: 1,000

\* Split strategy: Stratified train-test split



\---



\## 6. Evaluation Metrics



The baseline model was evaluated using:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* ROC-AUC



\### Baseline benchmark



| Metric    | Result |

| --------- | -----: |

| Accuracy  | 0.4900 |

| Precision | 0.5306 |

| Recall    | 0.7027 |

| F1 Score  | 0.6047 |

| ROC-AUC   | 0.4498 |



The model achieved a recall of 70.27% for the positive purchase class and an F1 score of 60.47%.



The ROC-AUC of 0.4498 indicates that the baseline model has limited discriminative performance and provides a clear benchmark for future model improvements.



\---



\## 7. Classification Results



The classification report from the 200-record test set was:



\* Class 0 precision: 0.38

\* Class 0 recall: 0.22

\* Class 0 F1 score: 0.28

\* Class 1 precision: 0.53

\* Class 1 recall: 0.70

\* Class 1 F1 score: 0.60



The baseline model performs better at identifying the positive purchase class than the negative class.



\---



\## 8. MLflow Experiment Tracking



MLflow was integrated into the baseline training pipeline.



The experiment name is:



`OmniRetail\_Checkpoint\_2\_Baseline`



The following information is tracked:



\* Model name

\* Test split

\* Random state

\* Dataset size

\* Feature count

\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* ROC-AUC

\* Trained model artifact



A local SQLite MLflow tracking database was created at:



`mlflow.db`



This provides a persistent experiment tracking foundation for subsequent model experiments.



\---



\## 9. Model Artifact



The trained baseline model was saved as:



`models/baseline\_purchase\_model.joblib`



The evaluation metrics were saved as:



`docs/baseline\_metrics.json`



Both artifacts were verified automatically using pytest.



\---



\## 10. Automated Testing



Seven automated tests were implemented to verify the Checkpoint 2 pipeline.



The tests validate:



1\. Clickstream dataset loading

2\. Product reviews dataset loading

3\. E-commerce test case loading

4\. Feature preprocessing

5\. Saved baseline model

6\. Saved baseline metrics

7\. MLflow database creation



\### Test result



`7 passed in 5.41s`



All Checkpoint 2 tests passed successfully.



\---



\## 11. Baseline Benchmark and Limitations



The Logistic Regression model establishes the initial classification benchmark for the project.



The current performance indicates that additional model development is required before production use.



Potential improvement areas include:



\* Feature engineering

\* Additional behavioral features

\* Class imbalance analysis

\* Tree-based machine learning models

\* Hyperparameter tuning

\* Cross-validation

\* Advanced purchase prediction models

\* Integration of additional relevant datasets



The final project also includes demand forecasting, dynamic pricing, and inventory replenishment components. These are separate downstream objectives and will require their own models and evaluation metrics.



The final project acceptance criterion of MAPE below 15% applies to the demand forecasting component and is therefore not used as the evaluation metric for this purchase-classification baseline.



\---



\## 12. Checkpoint 2 Outcome



Checkpoint 2 successfully established:



\* Complete dataset ingestion

\* Data cleaning and preprocessing

\* Feature preparation

\* Train-test split

\* Baseline Logistic Regression model

\* Classification evaluation benchmark

\* MLflow experiment tracking

\* Model artifact persistence

\* Metrics artifact persistence

\* Automated pipeline validation



The baseline is now ready to serve as the reference point for subsequent model improvement and development of the remaining OmniRetail AI modules.



