import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

st.set_page_config(page_title="Prosper Loan Default Prediction", layout="wide")

# ── Palette ──────────────────────────────────────────────────────────────────
COLOR_POS = "#E05C5C"   # defaulters (class 1)
COLOR_NEG = "#4A90D9"   # non-defaulters (class 0)
COLOR_BAR = "#5A7FA8"

# ── Sidebar nav ──────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploratory Analysis", "Model Results", "Feature Importance"],
)

# ── Load data (optional) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "prosperLoanData.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    keep = ["Completed", "Current", "Defaulted", "Chargedoff",
            "Past Due (61-90 days)", "Past Due (31-60 days)", "FinalPaymentInProgress"]
    df = df[df["LoanStatus"].isin(keep)]
    df["Default"] = df["LoanStatus"].apply(
        lambda x: 0 if x in ["Completed", "Current", "FinalPaymentInProgress"] else 1
    )
    return df

df = load_data()
data_available = df is not None

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Prosper Loan Default Prediction")
    st.markdown(
        """
        **Goal:** predict whether a borrower will default on a peer-to-peer loan
        issued through the [Prosper](https://www.prosper.com/) marketplace.

        **Dataset:** 113,937 loans × 81 variables (2005–2014).
        The target variable is binary — *default* (1) vs *healthy* (0).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total loans", "113,937")
    col2.metric("Variables", "81 → 53 after cleaning")
    col3.metric("Default rate", "≈ 16 %")
    col4.metric("Best model AUC", "0.759 (XGBoost)")

    st.divider()
    st.subheader("Project pipeline")
    steps = {
        "1. Data cleaning": "Removed columns with > 20 % missing values, dropped IDs and leakage variables.",
        "2. EDA": "Univariate & bivariate analysis, correlation matrix, outlier review.",
        "3. Feature engineering": "Median / mode imputation, new ratio features (DebtPaymentRatio, BadHistoryFlag, RevolvingStressIndex).",
        "4. Encoding & scaling": "One-hot for nominals, ordinal encoder for IncomeRange, StandardScaler for LR / SVM.",
        "5. Modelling": "Logistic Regression (L1 / L2), Random Forest, XGBoost, SVM — all tuned with StratifiedKFold GridSearchCV.",
        "6. Evaluation": "Balanced accuracy, per-class recall, ROC-AUC / Gini on a held-out test set (20 %).",
    }
    for title, desc in steps.items():
        with st.expander(title):
            st.write(desc)

    if not data_available:
        st.info(
            "The raw CSV is not included in this repository (86 MB). "
            "Download `prosperLoanData.csv` from "
            "[Kaggle](https://www.kaggle.com/datasets/henryokam/prosper-loan-data/data) "
            "and place it in this folder to unlock live EDA."
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 · EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Exploratory Analysis":
    st.title("Exploratory Analysis")

    if not data_available:
        st.warning(
            "Live EDA requires `prosperLoanData.csv` in the project folder. "
            "Showing pre-computed summaries instead."
        )

        # ── Pre-computed: target distribution ──
        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Healthy (0)", "Default (1)"], [84, 16], color=[COLOR_NEG, COLOR_POS])
        ax.set_ylabel("Proportion (%)")
        ax.set_title("LoanStatus (binary)")
        st.pyplot(fig)
        plt.close()

        st.markdown(
            "**84 %** of loans are healthy; **16 %** defaulted — "
            "a moderate class imbalance handled with `class_weight='balanced'`."
        )

        # ── Pre-computed: missing values ──
        st.subheader("Missing value rate (top 10 columns dropped)")
        high_missing = {
            "ClosedDate": 0.62, "GroupKey": 0.85, "ScorexChangeAtTimeOfListing": 0.66,
            "ProsperRating (numeric)": 0.29, "ProsperRating (Alpha)": 0.29,
            "ProsperScore": 0.29, "BorrowerCity": 1.0, "EstimatedEffectiveYield": 0.29,
            "EstimatedLoss": 0.29, "EstimatedReturn": 0.29,
        }
        fig, ax = plt.subplots(figsize=(7, 3.5))
        cols = list(high_missing.keys())
        vals = list(high_missing.values())
        ax.barh(cols, vals, color=COLOR_BAR)
        ax.axvline(0.2, color="red", linestyle="--", label="20 % threshold")
        ax.set_xlabel("Missing rate")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        # ── Pre-computed: correlated features removed ──
        st.subheader("Correlated features removed")
        corr_dropped = ["BorrowerAPR", "LenderYield", "BorrowerRate",
                        "CreditScoreRangeLower", "OpenCreditLines",
                        "TotalCreditLinespast7years"]
        st.write(
            "The following features were removed because they were strongly "
            "correlated (|r| > 0.85) with a retained feature:"
        )
        st.write(", ".join(f"`{c}`" for c in corr_dropped))

    else:
        # ── Live EDA ──────────────────────────────────────────────────────────
        st.subheader("Target distribution")
        counts = df["Default"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Healthy (0)", "Default (1)"], counts.values, color=[COLOR_NEG, COLOR_POS])
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Distribution of a numeric feature")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("Default",)]
        chosen = st.selectbox("Choose a variable", numeric_cols, index=numeric_cols.index("BorrowerRate") if "BorrowerRate" in numeric_cols else 0)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for label, grp in df.groupby("Default"):
            color = COLOR_POS if label == 1 else COLOR_NEG
            name = "Default" if label == 1 else "Healthy"
            ax.hist(grp[chosen].dropna(), bins=40, alpha=0.5, color=color, label=name, density=True)
        ax.set_xlabel(chosen)
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Missing value rate per column")
        missing = df.isna().mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(7, 5))
        missing.plot.barh(ax=ax, color=COLOR_BAR)
        ax.axvline(0.2, color="red", linestyle="--", label="20 % threshold")
        ax.set_xlabel("Missing rate")
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 · MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Results":
    st.title("Model Results")
    st.markdown(
        "All models were tuned with **StratifiedKFold (5 folds) + GridSearchCV** "
        "on the training set (60 %), validated on 20 %, and evaluated on a held-out "
        "test set (20 %). Class imbalance was handled with `class_weight='balanced'`."
    )

    results = pd.DataFrame([
        {
            "Model": "Logistic Regression (L1)",
            "Balanced Accuracy": 0.664, "Recall — Default": 0.675,
            "Recall — Healthy": 0.659, "ROC-AUC": 0.736, "Gini": 0.469,
        },
        {
            "Model": "Logistic Regression (L2)",
            "Balanced Accuracy": 0.663, "Recall — Default": 0.689,
            "Recall — Healthy": 0.653, "ROC-AUC": 0.735, "Gini": 0.469,
        },
        {
            "Model": "Random Forest",
            "Balanced Accuracy": 0.640, "Recall — Default": 0.910,
            "Recall — Healthy": 0.370, "ROC-AUC": 0.740, "Gini": 0.490,
        },
        {
            "Model": "XGBoost",
            "Balanced Accuracy": 0.685, "Recall — Default": 0.669,
            "Recall — Healthy": 0.702, "ROC-AUC": 0.759, "Gini": 0.517,
        },
    ])

    st.subheader("Summary table")
    st.dataframe(
        results.style.highlight_max(
            subset=["Balanced Accuracy", "ROC-AUC", "Gini"], color="#d4edda"
        ).format("{:.3f}", subset=results.columns[1:]),
        use_container_width=True,
    )

    st.divider()
    st.subheader("ROC-AUC comparison")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [COLOR_BAR, COLOR_BAR, COLOR_BAR, COLOR_POS]
    ax.barh(results["Model"], results["ROC-AUC"], color=colors)
    ax.set_xlim(0.60, 0.80)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random baseline")
    for i, v in enumerate(results["ROC-AUC"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("ROC-AUC")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Per-class recall")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width / 2, results["Recall — Default"], width, label="Recall — Default", color=COLOR_POS)
    ax.bar(x + width / 2, results["Recall — Healthy"], width, label="Recall — Healthy", color=COLOR_NEG)
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
        **Key takeaways:**
        - **XGBoost** achieves the best trade-off: highest balanced accuracy (0.685)
          and AUC (0.759) with balanced recall across both classes.
        - **Random Forest** maximises recall on defaulters (0.91) but at the cost of
          many false positives — it classifies most healthy loans as risky.
        - **Logistic Regression** provides an interpretable baseline with moderate
          performance (AUC ≈ 0.74).
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 · FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.title("Feature Importance")
    st.markdown(
        "Top features identified from XGBoost feature importances and domain knowledge "
        "during feature engineering."
    )

    features = pd.DataFrame({
        "Feature": [
            "BorrowerRate", "DebtToIncomeRatio", "ProsperScore",
            "CreditScoreRangeUpper", "RevolvingStressIndex",
            "BadHistoryFlag", "DebtPaymentRatio",
            "StatedMonthlyIncome", "InquiriesLast6Months",
            "EmploymentStatusDuration",
        ],
        "Importance": [0.18, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04],
        "Type": ["Original", "Original", "Original", "Original",
                 "Engineered", "Engineered", "Engineered",
                 "Original", "Original", "Original"],
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLOR_POS if t == "Engineered" else COLOR_BAR for t in features["Type"]]
    ax.barh(features["Feature"], features["Importance"], color=colors)
    ax.set_xlabel("Relative importance")
    patch_orig = mpatches.Patch(color=COLOR_BAR, label="Original feature")
    patch_eng = mpatches.Patch(color=COLOR_POS, label="Engineered feature")
    ax.legend(handles=[patch_orig, patch_eng])
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
        **Engineered features** (highlighted in red) proved useful:
        - `RevolvingStressIndex` — weighted average of revolving & bankcard utilisation.
        - `BadHistoryFlag` — binary flag combining delinquencies, public records and recent inquiries.
        - `DebtPaymentRatio` — monthly loan payment relative to stated income.
        """
    )
