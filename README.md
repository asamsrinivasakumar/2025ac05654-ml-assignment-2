# Census Income Classification — ML Assignment 2

## a. Problem Statement

Predict whether an individual's annual income exceeds $50,000 based on demographic
and employment attributes collected in the 1994 US Census (age, education,
occupation, hours worked, marital status, etc.). This is a **binary classification**
problem. Five classification models are trained on the same dataset and compared
using six evaluation metrics, and the results are made explorable through an
interactive Streamlit app.

## b. Dataset Description

- **Name:** Adult / Census Income Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Instances:** 45,222 (after dropping rows with missing values), well above the
  required minimum of 500
- **Features:** 13 (age, workclass, fnlwgt, education_num, marital_status,
  occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week,
  native_country), above the required minimum of 12
  - `education` (categorical) was dropped because `education_num` already encodes
    the same information ordinally
  - `native_country` (40 raw categories, ~90% "United-States") was bucketed to the
    top 5 countries + "Other" to avoid an overly sparse encoding
- **Target:** `income` — binary, `1` if income `>50K`, else `0`
- **Split:** 80% train / 20% test, stratified on the target (`random_state=42`)
- **Preprocessing:** `StandardScaler` for numeric features, `OneHotEncoder` for
  categorical features (fit once and reused for train/test/app inference,
  see `model/preprocessor.pkl`)

## c. GitHub Repository Link

> https://github.com/asamsrinivasakumar/2025ac05654-ml-assignment-2

## d. Models Used

All 5 models were trained on the identical preprocessed dataset/split. Metrics
below are computed on the held-out 20% test set (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.901 | 0.734 | 0.590 | 0.654 | 0.561 |
| Decision Tree | 0.848 | 0.887 | 0.737 | 0.599 | 0.661 | 0.569 |
| kNN | 0.840 | 0.889 | 0.708 | 0.603 | 0.651 | 0.551 |
| Naive Bayes | 0.764 | 0.863 | 0.514 | 0.858 | 0.643 | 0.518 |
| Random Forest (Ensemble) | 0.858 | 0.914 | 0.794 | 0.576 | 0.668 | 0.592 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced baseline (AUC 0.901) given the simple linear decision boundary; slightly under-predicts the minority (`>50K`) class, keeping recall around 0.59. |
| Decision Tree | Marginally better accuracy/F1 than Logistic Regression by capturing non-linear interactions (e.g. education × hours-worked), but a lower AUC signals less reliable probability ranking and a tendency to overfit rules the ensemble later smooths out. |
| kNN | Comparable overall accuracy to the linear model, but sensitive to the mixed numeric/one-hot feature space where distance is dominated by many sparse binary dimensions; slowest to score new data among the five. |
| Naive Bayes | By far the highest recall (0.858) — it aggressively predicts `>50K` — at the cost of the lowest precision and accuracy, because its conditional-independence assumption is violated by correlated features like education and occupation. Best choice only if catching every high-income case matters more than false positives. |
| Random Forest (Ensemble) | Best accuracy, AUC, precision, F1 and MCC of all five models by averaging many decorrelated trees, which reduces the overfitting seen in the single Decision Tree. Recall is the trade-off — it is the most conservative model about predicting the minority class. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — highest Accuracy, AUC, Precision, F1 and MCC. Naive Bayes remains preferable only in a recall-critical scenario. |

## Live Streamlit App

> https://2025ac05654-ml-assignment-2-ncdlomfwnsydjedupwzhb4.streamlit.app/

The app lets you:
1. Upload a test CSV (same schema as `test_data.csv`, incl. the `income` label) — or use the bundled `test_data.csv` by default
2. Pick one of the 5 models from a dropdown
3. View its Accuracy / AUC / Precision / Recall / F1 / MCC on that data
4. View the confusion matrix and full classification report
5. See the 5-model comparison table

## Project Structure

```
ml-assignment-2/
├── app.py                     # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv              # held-out 20% test split (raw features + label)
├── data/                      # raw UCI Adult dataset (for reproducibility)
│   ├── adult.data
│   ├── adult.test
│   └── adult.names
└── model/
    ├── train_models.py        # loads data, preprocesses, trains & evaluates all 5 models
    ├── preprocessor.pkl        # fitted ColumnTransformer
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── metrics.json            # metrics for all 5 models (source for the table above)
```

## Reproducing locally

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains all 5 models, regenerates test_data.csv + metrics.json
streamlit run app.py
```
