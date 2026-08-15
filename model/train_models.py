"""
Train 5 classification models on the UCI Adult (Census Income) dataset.

Dataset: https://archive.ics.uci.edu/ml/datasets/adult
Task: predict whether a person's annual income is <=50K or >50K.

Run from the project root:
    python model/train_models.py

Produces (all under model/):
    preprocessor.pkl        - fitted ColumnTransformer (encoding + scaling)
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    random_forest.pkl
    metrics.json             - evaluation metrics for every model (for the README table)
And at the project root:
    test_data.csv            - held-out test split (raw, unprocessed) used for the Streamlit app
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
# "education" is dropped: education_num already encodes it ordinally, so keeping
# both would be a redundant, high-cardinality categorical (16 extra one-hot columns).
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

# native_country has 40 distinct values but is ~90% "United-States"; bucketing the
# long tail keeps the one-hot encoding compact (and the saved models small) without
# throwing away the signal.
TOP_COUNTRIES = ["United-States", "Mexico", "Philippines", "Germany", "Canada"]


def load_raw_data():
    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "adult.data"),
        names=COLUMNS, sep=r",\s*", engine="python", na_values="?",
    )
    test_df = pd.read_csv(
        os.path.join(DATA_DIR, "adult.test"),
        names=COLUMNS, sep=r",\s*", engine="python", na_values="?", skiprows=1,
    )
    # adult.test labels have a trailing "." (e.g. "<=50K.")
    test_df["income"] = test_df["income"].str.replace(".", "", regex=False)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    df["income"] = (df["income"] == ">50K").astype(int)
    df = df.drop(columns=["education"])
    df["native_country"] = df["native_country"].where(
        df["native_country"].isin(TOP_COUNTRIES), "Other"
    )
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_score),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")

    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save a raw (unprocessed) test split for the assignment's required test_data.csv
    # and for the Streamlit app's "upload test data" feature.
    test_out = X_test.copy()
    test_out["income"] = y_test.values
    test_out.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
    print(f"Saved test_data.csv ({test_out.shape[0]} rows)")

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_enc = preprocessor.transform(X_test).astype(np.float32)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"), compress=3)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, random_state=42, n_jobs=-1
        ),
    }

    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "kNN",
        "naive_bayes": "Naive Bayes",
        "random_forest": "Random Forest (Ensemble)",
    }

    metrics = {}
    for key, model in models.items():
        print(f"Training {display_names[key]}...")
        # KNN and Naive Bayes need dense arrays
        Xtr = X_train_enc.toarray() if hasattr(X_train_enc, "toarray") else X_train_enc
        Xte = X_test_enc.toarray() if hasattr(X_test_enc, "toarray") else X_test_enc
        model.fit(Xtr, y_train)
        metrics[key] = {"display_name": display_names[key], **evaluate(model, Xte, y_test)}
        joblib.dump(model, os.path.join(MODEL_DIR, f"{key}.pkl"), compress=3)
        print(f"  {metrics[key]}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(MODEL_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
