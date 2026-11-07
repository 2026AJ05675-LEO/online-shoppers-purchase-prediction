
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree (Tuned)": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM (Tuned RBF)": "svm.pkl"
}

TARGET_COLUMN = "Revenue"

# --------------------------------------------------
# 2. Application heading
# --------------------------------------------------

st.title("🛒 Online Shoppers Purchase Prediction")

st.markdown(
    """
    ### Machine Learning Classification Dashboard

    Evaluate customer purchasing behaviour using five
    trained classification models.

    Upload a labelled CSV dataset, select a model and
    explore its classification performance.
    """
)

st.divider()

# --------------------------------------------------
# 3. Load selected model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# --------------------------------------------------
# 4. Dataset upload
# --------------------------------------------------

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing features and Revenue target",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.write("Dataset preview:")
        st.dataframe(data.head())

        st.write(
            f"Dataset dimensions: {data.shape[0]} rows "
            f"× {data.shape[1]} columns"
        )

        # --------------------------------------------------
        # 5. Validate dataset
        # --------------------------------------------------

        if TARGET_COLUMN not in data.columns:

            st.error(
                f"Required target column '{TARGET_COLUMN}' "
                "is missing from the uploaded dataset."
            )
            st.stop()

        X_test = data.drop(columns=[TARGET_COLUMN])
        y_test = data[TARGET_COLUMN]

        # Ensure target contains binary values
        if not set(y_test.dropna().unique()).issubset({0, 1, True, False}):

            st.error(
                "Revenue must contain binary values: 0 and 1."
            )
            st.stop()

        y_test = y_test.astype(int)

        if y_test.isna().any() or X_test.isna().any().any():

            st.error(
                "Missing values detected. Please upload a "
                "dataset without missing values."
            )
            st.stop()

        # --------------------------------------------------
        # 6. Model selection
        # --------------------------------------------------

        st.subheader("2. Select Classification Model")

        selected_model_name = st.selectbox(
            "Choose a model",
            list(MODEL_FILES.keys())
        )

        model_path = os.path.join(
            MODEL_DIR,
            MODEL_FILES[selected_model_name]
        )

        if not os.path.exists(model_path):

            st.error(
                f"Model artifact not found: {model_path}"
            )
            st.stop()

        model = load_model(model_path)

        # Validate feature columns and order
        expected_features = list(model.feature_names_in_)

        if set(X_test.columns) != set(expected_features):

            missing_features = sorted(
                set(expected_features) - set(X_test.columns)
            )

            extra_features = sorted(
                set(X_test.columns) - set(expected_features)
            )

            st.error("Uploaded dataset columns do not match the model.")

            if missing_features:
                st.write("Missing columns:", missing_features)

            if extra_features:
                st.write("Unexpected columns:", extra_features)

            st.stop()

        # Match the original training feature order
        X_test = X_test[expected_features]

        # --------------------------------------------------
        # 7. Generate predictions
        # --------------------------------------------------

        predictions = model.predict(X_test)

        # AUC score
        if hasattr(model, "predict_proba"):

            prediction_scores = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):

            prediction_scores = model.decision_function(X_test)

        else:

            prediction_scores = None

        # --------------------------------------------------
        # 8. Calculate evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        mcc = matthews_corrcoef(y_test, predictions)

        if prediction_scores is not None and y_test.nunique() == 2:

            auc = roc_auc_score(y_test, prediction_scores)

        else:

            auc = np.nan

        # --------------------------------------------------
        # 9. Display evaluation metrics
        # --------------------------------------------------

        st.divider()
        st.subheader("3. Model Evaluation Results")

        st.caption(f"Selected model: {selected_model_name}")

        metric_columns = st.columns(3)

        metrics = [
            ("Accuracy", accuracy),
            ("AUC", auc),
            ("Precision", precision),
            ("Recall", recall),
            ("F1-score", f1),
            ("MCC", mcc)
        ]

        for index, (metric_name, metric_value) in enumerate(metrics):

            with metric_columns[index % 3]:

                if np.isnan(metric_value):
                    st.metric(metric_name, "N/A")
                else:
                    st.metric(
                        metric_name,
                        f"{metric_value:.4f}"
                    )

        # --------------------------------------------------
        # 10. Confusion matrix
        # --------------------------------------------------

        st.divider()
        st.subheader("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Purchase", "Purchase"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # 11. Classification report
        # --------------------------------------------------

        st.divider()
        st.subheader("5. Classification Report")

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # --------------------------------------------------
        # 12. Download predictions
        # --------------------------------------------------

        st.divider()

        result_data = data.copy()
        result_data["Predicted_Revenue"] = predictions

        st.download_button(
            label="Download Prediction Results",
            data=result_data.to_csv(index=False),
            file_name="purchase_prediction_results.csv",
            mime="text/csv"
        )

        st.success("Model evaluation completed successfully!")

    except Exception as error:

        st.error(f"Unable to evaluate dataset: {error}")

else:

    st.info(
        "Please upload a labelled CSV dataset to begin model evaluation."
    )

# --------------------------------------------------
# 13. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "BITS WILP | M.Tech AIML | Machine Learning Assignment 2- 2026AJ05675"
)
