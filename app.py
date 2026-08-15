"""
Streamlit demo app for the ML Assignment 2 classification models.

Loads the 5 pre-trained models (see model/train_models.py), lets the user
upload a test CSV (same schema as test_data.csv), pick a model, and view
its evaluation metrics + confusion matrix / classification report on that data.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

REQUIRED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country", "income",
]


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_training_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


def evaluate_on_uploaded_data(model, preprocessor, df):
    X = df.drop(columns=["income"])
    y_true = df["income"]
    if y_true.dtype == object:
        y_true = (y_true.astype(str).str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)

    X_enc = preprocessor.transform(X).astype(np.float32)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    y_pred = model.predict(X_enc)
    y_score = model.predict_proba(X_enc)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_true, y_pred


def main():
    st.set_page_config(page_title="Income Classification Demo", layout="wide")
    st.title("Census Income Classification — Model Demo")
    st.caption(
        "BITS ML Assignment 2 — predicts whether annual income is >50K or <=50K "
        "using the UCI Adult (Census Income) dataset."
    )

    st.sidebar.header("1. Upload test data")
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV (same columns as test_data.csv, including the 'income' label)",
        type=["csv"],
    )
    st.sidebar.caption(
        "No file? The bundled test_data.csv (held-out 20% split, not used in training) "
        "is used automatically."
    )

    st.sidebar.header("2. Choose a model")
    model_choice = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read '{uploaded.name}' as CSV: {e}")
            st.stop()
        source_label = uploaded.name
        st.sidebar.success(f"Uploaded '{uploaded.name}' successfully ({len(df)} rows).")
    else:
        df = pd.read_csv(os.path.join(BASE_DIR, "test_data.csv"))
        source_label = "test_data.csv (bundled)"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
        st.stop()

    st.success(f"**Data source:** {source_label} &nbsp;|&nbsp; **Rows:** {len(df)}")
    with st.expander("Preview data"):
        st.dataframe(df.head(20))

    preprocessor = load_preprocessor()
    model = load_model(MODEL_FILES[model_choice])
    metrics, y_true, y_pred = evaluate_on_uploaded_data(model, preprocessor, df)

    st.subheader(f"Evaluation metrics — {model_choice}")
    cols = st.columns(6)
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    keys = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
    for col, label, key in zip(cols, labels, keys):
        col.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred, target_names=["<=50K", ">50K"], output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.subheader("Comparison across all 5 models (training-time held-out test set)")
    training_metrics = load_training_metrics()
    comparison_df = pd.DataFrame(
        [
            {
                "Model": v["display_name"],
                "Accuracy": v["accuracy"],
                "AUC": v["auc"],
                "Precision": v["precision"],
                "Recall": v["recall"],
                "F1": v["f1"],
                "MCC": v["mcc"],
            }
            for v in training_metrics.values()
        ]
    ).round(3)
    st.dataframe(comparison_df, use_container_width=True)


if __name__ == "__main__":
    main()
