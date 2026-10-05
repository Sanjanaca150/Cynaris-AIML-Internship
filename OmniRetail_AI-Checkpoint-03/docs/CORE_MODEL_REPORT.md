\# OmniRetail AI — Checkpoint 3

\## Core Model Training \& Experimentation Report



\### 1. Objective



The objective of this checkpoint was to train the primary machine learning model for the OmniRetail AI project using multiple hyperparameter configurations, track the experiments with MLflow, compare model performance, and select the best configuration based on F1 score.



\### 2. Dataset



Dataset used:



\- File: `aiml\_ecommerce\_clickstream.csv`

\- Total records: 1,000

\- Training records: 800

\- Testing records: 200

\- Features used: 9

\- Target: `label\_purchased`



The features include:



\*\*Numeric features\*\*

\- product\_views

\- add\_to\_cart

\- session\_duration\_mins

\- return\_visitor

\- discount\_applied

\- recommendation\_clicked



\*\*Categorical features\*\*

\- city

\- device

\- category



The data was split using an 80/20 train-test split with stratification and `random\_state=42`.



\### 3. Primary Model



The primary model selected for experimentation was:



\*\*Random Forest Classifier\*\*



The model was implemented using a Scikit-learn Pipeline containing:



1\. Numeric feature imputation

2\. Numeric feature scaling

3\. Categorical feature imputation

4\. One-hot encoding

5\. Random Forest classification



\### 4. Hyperparameter Configurations



Three configurations were tested.



| Configuration | n\_estimators | max\_depth | min\_samples\_split |

|---|---:|---:|---:|

| RandomForest\_Config\_1 | 100 | 5 | 2 |

| RandomForest\_Config\_2 | 200 | 10 | 2 |

| RandomForest\_Config\_3 | 300 | None | 2 |



All experiments used:



\- `random\_state = 42`

\- Test size = 20%

\- Random Forest Classifier



\### 5. MLflow Experiment Tracking



MLflow was used to track the model experiments.



\*\*MLflow Experiment Name:\*\*



`OmniRetail\_Checkpoint\_3\_Core\_Model`



For each configuration, the following parameters were logged:



\- Model type

\- Number of estimators

\- Maximum depth

\- Minimum samples split

\- Random state

\- Test size



The following evaluation metrics were logged:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- ROC-AUC



\### 6. Experiment Results



| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| RandomForest\_Config\_1 | 0.5350 | 0.5489 | 0.9099 | \*\*0.6847\*\* | 0.4258 |

| RandomForest\_Config\_2 | 0.5100 | 0.5430 | 0.7387 | 0.6260 | 0.4307 |

| RandomForest\_Config\_3 | 0.4850 | 0.5290 | 0.6577 | 0.5863 | 0.4427 |



\### 7. Best Configuration



The best configuration was:



\*\*RandomForest\_Config\_1\*\*



Hyperparameters:



\- `n\_estimators = 100`

\- `max\_depth = 5`

\- `min\_samples\_split = 2`

\- `random\_state = 42`



Performance:



\- Accuracy: \*\*0.5350\*\*

\- Precision: \*\*0.5489\*\*

\- Recall: \*\*0.9099\*\*

\- F1 Score: \*\*0.6847\*\*

\- ROC-AUC: \*\*0.4258\*\*



The configuration was selected using \*\*F1 score\*\* because the experiment was focused on the classification target `label\_purchased`.



RandomForest\_Config\_1 achieved the highest F1 score among the three tested configurations.



\### 8. Model Artifact



The selected model was retrained using the best configuration and saved as:



`models/best\_core\_model.joblib`



The saved model was successfully loaded and verified for prediction.



\### 9. Experiment Results Artifact



The complete experiment results were saved as:



`docs/experiment\_results.json`



This file contains:



\- Dataset information

\- Train/test record counts

\- Feature count

\- Selection metric

\- All three experiment configurations

\- Evaluation metrics

\- Best configuration



\### 10. Automated Testing



Automated tests were added to validate the checkpoint implementation.



Test coverage includes:



\- Clickstream dataset availability

\- Data preprocessing

\- Best model artifact

\- Model prediction

\- Experiment results file

\- Multiple configurations

\- Best configuration selection

\- MLflow database



Test result:



\*\*8 passed\*\*



\### 11. Conclusion



The OmniRetail AI primary Random Forest model was successfully trained using three different hyperparameter configurations.



All experiments were tracked using MLflow and evaluated using multiple classification metrics.



`RandomForest\_Config\_1` was selected as the best configuration based on the highest F1 score of \*\*0.6847\*\*.



The trained model, experiment results, and automated tests were successfully generated and verified.



\### 12. Project Status



\*\*Checkpoint 3 — Core Model + Experimentation: COMPLETED\*\*



\- Data pipeline: Completed

\- Primary model: Completed

\- Hyperparameter experiments: 3 completed

\- MLflow tracking: Completed

\- Best configuration selection: Completed

\- Model artifact: Completed

\- Experiment results: Completed

\- Automated tests: 8/8 passed

