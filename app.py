import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Classification Models")
st.write("Upload test data and select a trained model to view evaluation metrics.")

model_name = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Assume first column is target
    target_column = data.columns[0]
    y_true = data[target_column]
    X = data.drop(columns=[target_column])

    model_path_map = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl",
    }

    try:
        pipeline = joblib.load(model_path_map[model_name])

        y_pred = pipeline.predict(X)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")

        col4, col5 = st.columns(2)
        col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

    except FileNotFoundError:
        st.error("Model file not found. Please check filenames in the model folder.")
    except ValueError as e:
        st.error(f"Input data shape mismatch: {e}")

