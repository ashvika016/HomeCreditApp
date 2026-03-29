import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #f0f4ff;}
h1 {color: #1a3c6e;}
h2, h3 {color: #2b5ea7;}
.metric-card {
    background: #e8f0fe;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Home Credit Default Risk – Analysis Dashboard")
st.caption("Upload your data files, explore EDA insights, engineer features, and run an XGBoost model end-to-end.")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – File uploads
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Files")
    train_file    = st.file_uploader("application_train.csv",    type="csv", key="train")
    test_file     = st.file_uploader("application_test.csv",     type="csv", key="test")
    bureau_file   = st.file_uploader("bureau.csv",               type="csv", key="bureau")
    #prev_app_file = st.file_uploader("previous_application.csv", type="csv", key="prev")

    st.markdown("---")
    st.info("Upload at least **application_train.csv** to start exploring.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper – cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_obj):
    return pd.read_csv(file_obj)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔍 EDA",
    "⚙️ Feature Engineering",
    "🤖 Model Training",
    "📁 Submission",
])

# ===========================================================================
# TAB 0 – OVERVIEW
# ===========================================================================
with tabs[0]:
    st.header("Dataset Overview")

    if train_file is None:
        st.warning("Upload **application_train.csv** in the sidebar to begin.")
        st.stop()

    app_train = load_csv(train_file)
    app_train["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{app_train.shape[0]:,}")
    col2.metric("Features", f"{app_train.shape[1]:,}")
    col3.metric("Default Rate", f"{app_train['TARGET'].mean()*100:.1f}%")
    col4.metric("Missing Cells", f"{app_train.isnull().sum().sum():,}")

    with st.expander("Preview (first 5 rows)"):
        st.dataframe(app_train.head(), use_container_width=True)

    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = app_train["TARGET"].value_counts().sort_index()
    ax.bar(["Repaid (0)", "Defaulted (1)"], counts.values, color=["#4c9be8", "#e8624c"])
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=9)
    st.pyplot(fig, use_container_width=False)

    st.subheader("Top 20 Columns with Missing Values")
    missing_pct = (app_train.isnull().mean() * 100).sort_values(ascending=False).head(20)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=missing_pct.index, y=missing_pct.values, ax=ax2, color="#4c9be8")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=8)
    ax2.set_ylabel("% Missing")
    ax2.set_title("Top 20 Missing Value Columns")
    st.pyplot(fig2)

# ===========================================================================
# TAB 1 – EDA
# ===========================================================================
with tabs[1]:
    st.header("Exploratory Data Analysis")

    if train_file is None:
        st.warning("Upload **application_train.csv** in the sidebar.")
        st.stop()

    app_train = load_csv(train_file)
    app_train["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    edu_map = {
        "Lower secondary": 1,
        "Secondary / secondary special": 2,
        "Incomplete higher": 3,
        "Higher education": 4,
        "Academic degree": 5,
    }
    app_train["education_encoded"] = app_train["NAME_EDUCATION_TYPE"].map(edu_map)

    # ── External Sources heatmap ──────────────────────────────────────────
    st.subheader("External Source Scores vs Target")
    ext_cols = [c for c in ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in app_train.columns]
    if len(ext_cols) > 1:
        corr = app_train[ext_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
        ax.set_title("Correlation: External Sources vs Target")
        st.pyplot(fig)

    # ── KDE plots ─────────────────────────────────────────────────────────
    st.subheader("Feature Distribution by Target Class")
    kde_features = [c for c in [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DAYS_BIRTH", "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_EMPLOYED"
    ] if c in app_train.columns]

    selected_feat = st.selectbox("Choose a feature to plot:", kde_features)

    fig, ax = plt.subplots(figsize=(9, 4))
    for target_val, label, color in [(0, "Repaid (0)", "#4c9be8"), (1, "Defaulted (1)", "#e8624c")]:
        subset = app_train.loc[app_train["TARGET"] == target_val, selected_feat].dropna()
        # clip extreme values for readability
        p1, p99 = np.percentile(subset, [1, 99])
        subset = subset.clip(p1, p99)
        sns.kdeplot(subset, label=label, fill=True, ax=ax, color=color, alpha=0.5)
    ax.set_title(f"Distribution of {selected_feat} by Target")
    ax.set_xlabel(selected_feat)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # ── Categorical default rates ─────────────────────────────────────────
    st.subheader("Default Rate by Category")
    cat_cols = [c for c in ["NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                             "NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE"] if c in app_train.columns]
    cat_choice = st.selectbox("Choose a categorical feature:", cat_cols)

    temp = app_train[[cat_choice, "TARGET"]].groupby(cat_choice, as_index=False).mean()
    temp = temp.sort_values("TARGET", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=cat_choice, y="TARGET", data=temp, palette="Reds_d", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"Default Rate by {cat_choice}")
    ax.set_ylabel("Probability of Default")
    st.pyplot(fig)

    # ── Important features heatmap ────────────────────────────────────────
    st.subheader("Correlation Heatmap – Key Features")
    key_cols = [c for c in ["TARGET", "DAYS_BIRTH", "AMT_CREDIT",
                             "AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "education_encoded"] if c in app_train.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(app_train[key_cols].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation: Key Features vs Target")
    st.pyplot(fig)

# ===========================================================================
# TAB 2 – FEATURE ENGINEERING
# ===========================================================================
with tabs[2]:
    st.header("Feature Engineering")

    if train_file is None:
        st.warning("Upload **application_train.csv** in the sidebar.")
        st.stop()

    app_train = load_csv(train_file)
    app_train["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # ── Application-level features ────────────────────────────────────────
    st.subheader("Application-Level Ratios")
    app_train["CREDIT_INCOME_PERCENT"] = app_train["AMT_CREDIT"] / app_train["AMT_INCOME_TOTAL"]
    app_train["ANNUITY_INCOME_PERCENT"] = app_train["AMT_ANNUITY"] / app_train["AMT_INCOME_TOTAL"]
    app_train["CREDIT_TERM"] = app_train["AMT_ANNUITY"] / app_train["AMT_CREDIT"]

    st.dataframe(
        app_train[["CREDIT_INCOME_PERCENT", "ANNUITY_INCOME_PERCENT", "CREDIT_TERM", "TARGET"]].head(8),
        use_container_width=True,
    )

    # ── Bureau aggregation ────────────────────────────────────────────────
    st.subheader("Bureau Aggregation")
    if bureau_file:
        bureau = load_csv(bureau_file)
        bureau_agg = bureau.groupby("SK_ID_CURR").agg(
            BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
            BUREAU_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
            BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
            BUREAU_ACTIVE_LOANS=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        ).reset_index()

        st.success(f"Bureau aggregated → {bureau_agg.shape[0]:,} clients, {bureau_agg.shape[1]} features")
        st.dataframe(bureau_agg.head(6), use_container_width=True)

        app_train = app_train.merge(bureau_agg, on="SK_ID_CURR", how="left")
        app_train[["BUREAU_LOAN_COUNT", "BUREAU_TOTAL_DEBT", "BUREAU_ACTIVE_LOANS"]] = \
            app_train[["BUREAU_LOAN_COUNT", "BUREAU_TOTAL_DEBT", "BUREAU_ACTIVE_LOANS"]].fillna(0)

        app_train["TOTAL_DEBT_RATIO"] = app_train["BUREAU_TOTAL_DEBT"] / (app_train["AMT_INCOME_TOTAL"] + 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        for tv, label, color in [(0, "Repaid", "#4c9be8"), (1, "Defaulted", "#e8624c")]:
            subset = app_train.loc[(app_train["TARGET"] == tv) & (app_train["TOTAL_DEBT_RATIO"] < 5), "TOTAL_DEBT_RATIO"]
            sns.kdeplot(subset, label=label, fill=True, ax=ax, color=color, alpha=0.5)
        ax.set_title("External Debt-to-Income Ratio by Default Status")
        ax.set_xlabel("External Debt / Income")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Upload **bureau.csv** to see bureau aggregation.")

    # ── Previous applications ─────────────────────────────────────────────
    # st.subheader("Previous Applications Aggregation")
    # if prev_app_file:
    #     prev_app = load_csv(prev_app_file)
    #     prev_cats = pd.get_dummies(prev_app["NAME_CONTRACT_STATUS"], prefix="PREV")
    #     prev_app = pd.concat([prev_app, prev_cats], axis=1)
    #     prev_app["APP_CREDIT_PERC"] = prev_app["AMT_CREDIT"] / prev_app["AMT_APPLICATION"].replace(0, np.nan)

    #     agg_dict = {
    #         "SK_ID_PREV": "count",
    #         "AMT_CREDIT": "mean",
    #         "AMT_APPLICATION": "mean",
    #         "APP_CREDIT_PERC": "mean",
    #         "AMT_DOWN_PAYMENT": "mean",
    #     }
    #     for col in ["PREV_Approved", "PREV_Refused", "PREV_Canceled", "PREV_Unused offer"]:
    #         if col in prev_app.columns:
    #             agg_dict[col] = "sum"

    #     prev_agg = prev_app.groupby("SK_ID_CURR").agg(agg_dict).reset_index()
    #     prev_agg.columns = (["SK_ID_CURR", "PREV_COUNT", "PREV_AVG_CREDIT", "PREV_AVG_AMT_APP",
    #                           "PREV_AVG_CREDIT_PERC", "PREV_AVG_DOWN_PAYMENT"] +
    #                          [c for c in ["PREV_CNT_APPROVED", "PREV_CNT_REFUSED",
    #                                       "PREV_CNT_CANCELED", "PREV_CNT_UNUSED"]
    #                           if ("PREV_" + c.split("_CNT_")[1].capitalize()) in prev_app.columns or
    #                              c.replace("CNT_", "") in prev_app.columns
    #                           ][:len(agg_dict) - 6])

    #     st.success(f"Previous applications aggregated → {prev_agg.shape[0]:,} clients")
    #     st.dataframe(prev_agg.head(6), use_container_width=True)
    # else:
    #     st.info("Upload **previous_application.csv** to see previous application features.")

# ===========================================================================
# TAB 3 – MODEL TRAINING
# ===========================================================================
with tabs[3]:
    st.header("Model Training — XGBoost")

    if train_file is None:
        st.warning("Upload **application_train.csv** in the sidebar.")
        st.stop()

    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    except ImportError:
        st.error("XGBoost / scikit-learn not installed in this environment.")
        st.stop()

    st.info("This tab trains an XGBoost model on the uploaded data. Training may take a minute.")

    # ── Hyperparameter controls ───────────────────────────────────────────
    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_est      = c1.slider("n_estimators",   100, 1000, 300, 50)
        lr         = c2.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        max_depth  = c3.slider("max_depth", 3, 10, 6)
        subsample  = c1.slider("subsample",      0.5, 1.0, 0.8, 0.05)
        col_bt     = c2.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
        test_size  = c3.slider("Validation split", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Train Model"):
        with st.spinner("Preparing data & training…"):
            # Load & preprocess
            df = load_csv(train_file).copy()
            df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

            # Bureau merge
            if bureau_file:
                bureau = load_csv(bureau_file)
                bureau_counts = bureau.groupby("SK_ID_CURR", as_index=False)["SK_ID_BUREAU"].count()
                bureau_counts.rename(columns={"SK_ID_BUREAU": "BUREAU_LOAN_COUNT"}, inplace=True)
                df = df.merge(bureau_counts, on="SK_ID_CURR", how="left")
                df["BUREAU_LOAN_COUNT"].fillna(0, inplace=True)

            # Previous apps merge
            # if prev_app_file:
            #     prev_app = load_csv(prev_app_file)
            #     prev_agg = prev_app.groupby("SK_ID_CURR", as_index=False)["AMT_CREDIT"].mean()
            #     prev_agg.columns = ["SK_ID_CURR", "PREV_APP_CREDIT_MEAN"]
            #     df = df.merge(prev_agg, on="SK_ID_CURR", how="left")

            # Drop high-missing columns
            miss = df.isnull().mean()
            df.drop(columns=miss[miss > 0.5].index.tolist(), inplace=True)

            # Impute numeric
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

            # One-hot encode
            df = pd.get_dummies(df)

            X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
            y = df["TARGET"]

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            ratio = (y_train == 0).sum() / (y_train == 1).sum()

            model = XGBClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=col_bt,
                scale_pos_weight=ratio,
                n_jobs=-1,
                random_state=42,
                eval_metric="auc",
                use_label_encoder=False,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # ── Results ───────────────────────────────────────────────────────
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred  = model.predict(X_val)
        auc     = roc_auc_score(y_val, y_proba)

        st.success(f"✅ Training complete! Validation ROC-AUC: **{auc:.4f}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("ROC-AUC", f"{auc:.4f}")
        report = classification_report(y_val, y_pred, output_dict=True)
        m2.metric("Precision (Default class)", f"{report['1']['precision']:.3f}")
        m3.metric("Recall (Default class)", f"{report['1']['recall']:.3f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Repaid", "Default"],
                    yticklabels=["Repaid", "Default"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Feature importance
        st.subheader("Top 20 Features by Importance")
        imp_df = pd.DataFrame({"feature": X_train.columns,
                                "importance": model.feature_importances_}
                               ).sort_values("importance", ascending=False).head(20)
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4c9be8")
        ax2.set_title("Top 20 Feature Importances")
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)

        # Store model & columns in session state for submission tab
        st.session_state["model"] = model
        st.session_state["train_cols"] = list(X_train.columns)
        st.session_state["train_medians"] = df[num_cols].median().to_dict()

# ===========================================================================
# TAB 4 – SUBMISSION
# ===========================================================================
with tabs[4]:
    st.header("Generate Submission File")

    if test_file is None:
        st.warning("Upload **application_test.csv** in the sidebar.")
        st.stop()

    if "model" not in st.session_state:
        st.warning("Train a model first in the **Model Training** tab.")
        st.stop()

    if st.button("📥 Generate Predictions"):
        with st.spinner("Processing test set…"):
            model      = st.session_state["model"]
            train_cols = st.session_state["train_cols"]
            medians    = st.session_state["train_medians"]

            app_test = load_csv(test_file).copy()
            app_test["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

            if bureau_file:
                bureau = load_csv(bureau_file)
                bureau_counts = bureau.groupby("SK_ID_CURR", as_index=False)["SK_ID_BUREAU"].count()
                bureau_counts.rename(columns={"SK_ID_BUREAU": "BUREAU_LOAN_COUNT"}, inplace=True)
                app_test = app_test.merge(bureau_counts, on="SK_ID_CURR", how="left")
                app_test["BUREAU_LOAN_COUNT"].fillna(0, inplace=True)

            if prev_app_file:
                prev_app = load_csv(prev_app_file)
                prev_agg = prev_app.groupby("SK_ID_CURR", as_index=False)["AMT_CREDIT"].mean()
                prev_agg.columns = ["SK_ID_CURR", "PREV_APP_CREDIT_MEAN"]
                app_test = app_test.merge(prev_agg, on="SK_ID_CURR", how="left")

            ids = app_test["SK_ID_CURR"].copy()
            app_test = pd.get_dummies(app_test.drop(columns=["SK_ID_CURR"], errors="ignore"))

            # Align to training columns (add missing cols as 0, drop extras)
            for col in train_cols:
                if col not in app_test.columns:
                    app_test[col] = 0
            app_test = app_test[train_cols]

            # Fill remaining NaNs with training medians
            for col in app_test.columns:
                if app_test[col].isnull().any():
                    app_test[col].fillna(medians.get(col, 0), inplace=True)

            probs = model.predict_proba(app_test)[:, 1]

        submission = pd.DataFrame({"SK_ID_CURR": ids.values, "TARGET": probs})
        st.success(f"✅ Predictions generated for {len(submission):,} applicants.")
        st.dataframe(submission.head(10), use_container_width=True)

        csv_bytes = submission.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="home_credit_submission.csv",
            mime="text/csv",
        )
