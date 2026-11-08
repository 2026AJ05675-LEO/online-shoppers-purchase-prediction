
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Purchase Intelligence | Model Evaluation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("model")
TARGET_COLUMN = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree (Tuned)": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM (Tuned RBF)": "svm.pkl",
}

# ============================================================
# 2. CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {
        --ink: #172B4D;
        --muted: #64748B;
        --brand: #2457D6;
        --brand-dark: #193F9E;
        --line: #E5EAF2;
        --surface: #FFFFFF;
        --canvas: #F5F7FB;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.035em;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted);
    }

    /* HERO */

    .hero {
        background: linear-gradient(
            115deg,
            #142D59 0%,
            #1E4CA7 62%,
            #2468C9 100%
        );
        border-radius: 22px;
        padding: 30px 34px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 14px 34px rgba(25, 63, 158, 0.14);
    }

    .hero-kicker {
        color: #C8D8FF;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        line-height: 1.16;
        font-weight: 800;
        letter-spacing: -0.045em;
        margin-bottom: 10px;
    }

    .hero-copy {
        color: #E1EAFF;
        font-size: 0.98rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 18px;
        border: 1px solid rgba(255,255,255,0.28);
        background: rgba(255,255,255,0.10);
        color: #FFFFFF;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    /* SECTION HEADINGS */

    .section-kicker {
        color: var(--brand);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 5px;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 14px;
    }

    /* PANELS */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-radius: 16px;
        border-color: var(--line);
    }

    /* METRIC CARDS */

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 15px 17px;
        min-height: 105px;
        box-shadow: 0 4px 14px rgba(23, 43, 77, 0.035);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 6px;
    }

    .status-pill {
        display: inline-block;
        background: #E8F7F2;
        color: #08785F;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* INPUTS */

    div[data-testid="stFileUploader"] {
        background: #F8FAFF;
        border: 1px dashed #A9BCE6;
        border-radius: 13px;
        padding: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.75rem;
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--brand);
        border: 1px solid var(--brand);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--brand-dark);
        border-color: var(--brand-dark);
        box-shadow: 0 5px 14px rgba(36,87,214,0.2);
    }

    .stDownloadButton > button {
        border: 1px solid #CBD7EE;
        color: var(--brand-dark);
        background: #F7F9FF;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    /* TABLES AND TABS */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stTabs"] button {
        font-weight: 700;
    }

    hr {
        border-color: var(--line);
    }

    /* FOOTER */

    .app-footer {
        color: var(--muted);
        text-align: center;
        font-size: 0.78rem;
        padding: 20px 0 4px 0;
    }

    /* RESPONSIVE STYLES */

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 23px 20px;
            border-radius: 17px;
        }

        .metric-card {
            padding: 13px;
            min-height: 94px;
        }

        .metric-value {
            font-size: 1.3rem;
        }

        .section-title {
            font-size: 1.1rem;
        }
    }

    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.6rem;
        }

        .hero-copy {
            font-size: 0.88rem;
        }

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
        
/* =========================================
   METRIC VISIBILITY AND CONTRAST
========================================= */

/* Streamlit metric labels */
[data-testid="stMetricLabel"] {
    color: #64748B !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
    color: #64748B !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Streamlit metric values */
[data-testid="stMetricValue"] {
    color: #172B4D !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

[data-testid="stMetricValue"] div {
    color: #172B4D !important;
    opacity: 1 !important;
}

/* Metric supporting text */
[data-testid="stMetricDelta"] {
    color: #64748B !important;
}

/* =========================================
   CONFUSION MATRIX BREAKDOWN
========================================= */

/* Metric labels inside breakdown */
[data-testid="stVerticalBlock"] 
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

/* Metric values inside breakdown */
[data-testid="stVerticalBlock"] 
[data-testid="stMetricValue"] {
    color: #172B4D !important;
}

/* Supporting explanation */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    opacity: 1 !important;
}


/* =========================================
   TABLE AND REPORT VISIBILITY
========================================= */

[data-testid="stDataFrame"] {
    color: #172B4D !important;
    border-color: #E2E8F0 !important;
}

/* =========================================
   FINAL TAB COLOUR OVERRIDE
========================================= */

/* Target all Streamlit tab buttons */
.stApp [role="tablist"] button,
.stApp [role="tablist"] [role="tab"] {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Target nested text elements */
.stApp [role="tablist"] button *,
.stApp [role="tablist"] [role="tab"] * {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    opacity: 1 !important;
}

/* Inactive tabs */
.stApp [role="tablist"] [role="tab"][aria-selected="false"] {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    opacity: 1 !important;
}

/* Active tab */
.stApp [role="tablist"] [role="tab"][aria-selected="true"] {
    color: #2457D6 !important;
    -webkit-text-fill-color: #2457D6 !important;
    font-weight: 700 !important;
    border-bottom-color: #2457D6 !important;
}

/* Active tab nested text */
.stApp [role="tablist"] [role="tab"][aria-selected="true"] * {
    color: #2457D6 !important;
    -webkit-text-fill-color: #2457D6 !important;
    opacity: 1 !important;
}

/* Hover should not change text to theme red */
.stApp [role="tablist"] [role="tab"]:hover,
.stApp [role="tablist"] [role="tab"]:hover * {
    color: #2457D6 !important;
    -webkit-text-fill-color: #2457D6 !important;
    opacity: 1 !important;
}


    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. REUSABLE FUNCTIONS
# ============================================================

def render_html(html_content):
    """Render HTML after removing unwanted indentation."""

    cleaned_html = "\n".join(
        line.strip()
        for line in html_content.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_path):
    """Load a saved sklearn pipeline."""
    return joblib.load(model_path)


def section_heading(kicker, title, description=""):
    """Render a consistent section heading."""

    render_html(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <p class="section-copy">{description}</p>
        """
    )


def metric_card(label, value, note):
    """Render an evaluation metric card."""

    formatted = (
        "N/A"
        if pd.isna(value)
        else f"{value:.2%}"
    )

    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{formatted}</div>
            <div class="metric-note">{note}</div>
        </div>
        """
    )


def validate_dataset(data, expected_features):
    """Validate the uploaded dataset."""

    if data.empty:
        raise ValueError("The uploaded CSV is empty.")

    if data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names were found in the uploaded file."
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    y_test = data[TARGET_COLUMN]

    if y_test.isna().any():
        raise ValueError(
            "The Revenue target contains missing values."
        )

    valid_targets = {0, 1, False, True}

    if not set(y_test.unique()).issubset(valid_targets):
        raise ValueError(
            "Revenue must contain only binary values: 0 and 1."
        )

    X_test = data.drop(columns=[TARGET_COLUMN])

    missing_features = sorted(
        set(expected_features) - set(X_test.columns)
    )

    extra_features = sorted(
        set(X_test.columns) - set(expected_features)
    )

    if missing_features or extra_features:

        messages = [
            "Uploaded feature columns do not match the selected model."
        ]

        if missing_features:
            messages.append(
                f"Missing columns: {', '.join(missing_features)}"
            )

        if extra_features:
            messages.append(
                f"Unexpected columns: {', '.join(extra_features)}"
            )

        raise ValueError(" ".join(messages))

    X_test = X_test[expected_features]

    if X_test.isna().any().any():
        raise ValueError(
            "Missing values detected in the input features."
        )

    return X_test, y_test.astype(int)


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-kicker">
            Machine Learning · Model Evaluation
        </div>

        <div class="hero-title">
            Purchase Intelligence
        </div>

        <p class="hero-copy">
            Explore online shopping purchase behaviour using
            five trained classification models. Upload a labelled
            dataset, select a model and evaluate its performance.
        </p>

        <span class="hero-tag">
            BITS WILP · M.Tech AIML · Assignment 2
        </span>
    </div>
    """
)

# ============================================================
# 5. DATASET UPLOAD AND MODEL SELECTION
# ============================================================

section_heading(
    "01 / EVALUATION SETUP",
    "Prepare your evaluation",
    "Upload a labelled CSV file and select the trained model you want to assess.",
)

setup_left, setup_right = st.columns(
    [1.25, 1],
    gap="large",
)

with setup_left:

    with st.container(border=True):

        st.markdown("**Upload test dataset**")

        st.caption(
            "CSV must contain all 17 input features and "
            "the binary Revenue target (0 or 1)."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed",
            help="Upload a labelled dataset with the original training features.",
        )

        st.caption(
            "The saved models evaluate your data without retraining."
        )


with setup_right:

    with st.container(border=True):

        st.markdown("**Choose classification model**")

        selected_model_name = st.selectbox(
            "Select a trained model",
            options=list(MODEL_FILES.keys()),
            index=4,
        )

        st.markdown(
            '<span class="status-pill">'
            '● 5 trained models available'
            '</span>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Select which saved model should perform the evaluation."
        )


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a labelled CSV dataset to begin. "
        "The file must contain the 17 input features "
        "and a Revenue column."
    )

else:

    try:

        data = pd.read_csv(uploaded_file)

        model_path = MODEL_DIR / MODEL_FILES[selected_model_name]

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        model = load_model(str(model_path))

        expected_features = list(
            model.feature_names_in_
        )

        X_test, y_test = validate_dataset(
            data,
            expected_features,
        )

        with st.spinner("Evaluating the selected model..."):

            predictions = model.predict(X_test)

            if hasattr(model, "predict_proba"):

                prediction_scores = (
                    model.predict_proba(X_test)[:, 1]
                )

            elif hasattr(model, "decision_function"):

                prediction_scores = (
                    model.decision_function(X_test)
                )

            else:

                prediction_scores = None

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            mcc = matthews_corrcoef(
                y_test,
                predictions,
            )

            if (
                prediction_scores is not None
                and y_test.nunique() == 2
            ):

                auc = roc_auc_score(
                    y_test,
                    prediction_scores,
                )

            else:

                auc = np.nan


        # ----------------------------------------------------
        # 7. DATASET SUMMARY
        # ----------------------------------------------------

        st.divider()

        section_heading(
            "02 / PERFORMANCE OVERVIEW",
            "Model evaluation results",
            f"Results for {selected_model_name} on the uploaded dataset.",
        )

        info_cols = st.columns(3)

        dataset_summary = [
            ("Evaluation records", f"{len(data):,}"),
            ("Input features", f"{X_test.shape[1]}"),
            ("Purchase rate", f"{y_test.mean():.1%}"),
        ]

        for col, (label, value) in zip(
            info_cols,
            dataset_summary,
        ):

            with col:
                st.metric(label, value)


        # ----------------------------------------------------
        # 8. PERFORMANCE METRICS
        # ----------------------------------------------------

        st.markdown("#### Classification metrics")

        metric_cols = st.columns(3, gap="medium")

        metric_items = [
            (
                "Accuracy",
                accuracy,
                "Overall correct predictions",
            ),
            (
                "AUC",
                auc,
                "Ability to rank both classes",
            ),
            (
                "Precision",
                precision,
                "Reliability of purchase predictions",
            ),
            (
                "Recall",
                recall,
                "Purchases successfully identified",
            ),
            (
                "F1-score",
                f1,
                "Balance of precision and recall",
            ),
            (
                "MCC",
                mcc,
                "Balanced correlation measure",
            ),
        ]

        for index, (label, value, note) in enumerate(metric_items):

            with metric_cols[index % 3]:

                metric_card(
                    label,
                    value,
                    note,
                )


        # ----------------------------------------------------
        # 9. MODEL DIAGNOSTICS
        # ----------------------------------------------------

        st.divider()

        section_heading(
            "03 / MODEL DIAGNOSTICS",
            "Understand the predictions",
            "Review correct classifications, prediction errors and class-level performance.",
        )

        matrix_tab, report_tab, data_tab = st.tabs(
            [
                "Confusion Matrix",
                "Classification Report",
                "Dataset Preview",
            ]
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        with matrix_tab:

            chart_col, explanation_col = st.columns(
                [1.25, 1],
                gap="large",
            )

            with chart_col:

                fig, ax = plt.subplots(
                    figsize=(6.2, 4.6)
                )

                display = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[
                        "No Purchase",
                        "Purchase",
                    ],
                )

                display.plot(
                    ax=ax,
                    cmap="Blues",
                    values_format="d",
                    colorbar=False,
                )

                ax.set_title(
                    "Actual vs Predicted",
                    fontsize=13,
                    pad=14,
                )

                ax.set_xlabel(
                    "Predicted label",
                    labelpad=10,
                )

                ax.set_ylabel(
                    "Actual label",
                    labelpad=10,
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)


            with explanation_col:

                st.markdown("**Confusion matrix breakdown**")

                tn, fp, fn, tp = cm.ravel()

                st.metric(
                    "Correctly identified non-purchases",
                    f"{tn:,}",
                )

                st.metric(
                    "Correctly identified purchases",
                    f"{tp:,}",
                )

                st.metric(
                    "False purchase predictions",
                    f"{fp:,}",
                )

                st.metric(
                    "Missed purchases",
                    f"{fn:,}",
                )

                st.caption(
                    "False negatives represent purchasing sessions "
                    "the model classified as non-purchases."
                )


        # ----------------------------------------------------
        # 10. CLASSIFICATION REPORT
        # ----------------------------------------------------

        with report_tab:

            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=[
                    "No Purchase",
                    "Purchase",
                ],
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True,
            )


        # ----------------------------------------------------
        # 11. DATASET PREVIEW
        # ----------------------------------------------------

        with data_tab:

            preview_rows = st.slider(
                "Number of preview rows",
                min_value=5,
                max_value=min(50, len(data)),
                value=min(10, len(data)),
                step=5,
            )

            st.dataframe(
                data.head(preview_rows),
                use_container_width=True,
                hide_index=True,
            )


        # ----------------------------------------------------
        # 12. DOWNLOAD RESULTS
        # ----------------------------------------------------

        st.divider()

        section_heading(
            "04 / EXPORT",
            "Download evaluation output",
            "Export the uploaded records with the model's predicted Revenue values.",
        )

        result_data = data.copy()

        result_data["Predicted_Revenue"] = predictions

        download_col, note_col = st.columns(
            [1, 2],
            gap="medium",
        )

        with download_col:

            st.download_button(
                label="Download prediction results",
                data=result_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="purchase_prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

        with note_col:

            st.success(
                f"Evaluation completed successfully for "
                f"{len(data):,} records using {selected_model_name}."
            )


    # ========================================================
    # 13. ERROR HANDLING
    # ========================================================

    except pd.errors.EmptyDataError:

        st.error(
            "The uploaded CSV does not contain readable data."
        )

    except pd.errors.ParserError:

        st.error(
            "The CSV file could not be parsed. "
            "Please check its formatting."
        )

    except (ValueError, FileNotFoundError) as error:

        st.error(str(error))

    except Exception as error:

        st.error(
            "The dataset could not be evaluated due to an "
            "unexpected issue. Please verify the file and try again."
        )

        with st.expander("Technical details"):

            st.code(str(error))


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        Purchase Intelligence · BITS WILP
        · M.Tech Artificial Intelligence & Machine Learning
        <br>
        Machine Learning Assignment 2 · 2026AJ05675
    </div>
    """,
    unsafe_allow_html=True,
)
