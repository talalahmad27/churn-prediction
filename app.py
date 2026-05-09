"""Streamlit dashboard: churn EDA and interactive prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import analysis
from src.modeling import (
    DEFAULT_DATA_PATH,
    MODEL_PATH,
    TARGET,
    aggregated_model_importance,
    fill_complaint_type_na,
    load_model_artifact,
    permutation_importance_by_feature,
    predict_churn,
)

st.set_page_config(page_title="Customer churn", layout="wide")


@st.cache_data
def load_dataset(path_str: str) -> pd.DataFrame:
    """Purpose: Load churn CSV once per session/path so Streamlit reruns stay fast.

    Steps:
        1. Receive the dataset path as a string (hashable cache key).
        2. Delegate to `analysis.load_data` to read the CSV into a DataFrame.
        3. Streamlit caches the result across reruns until the cache is cleared.
    """
    return analysis.load_data(path_str)


@st.cache_resource
def cached_model():
    """Purpose: Load the serialized sklearn pipeline and churn threshold once per app lifetime.

    Steps:
        1. Call `load_model_artifact` against the default `MODEL_PATH`.
        2. Cache the loaded objects so repeated predictions avoid repeated disk IO / deserialization.
    """
    return load_model_artifact()


def model_available() -> bool:
    """Purpose: Tell the UI whether prediction features should be enabled or training instructions shown.

    Steps:
        1. Check whether the expected joblib file exists at `MODEL_PATH`.
        2. Return True if present, False otherwise.
    """
    return MODEL_PATH.exists()


def prediction_form(df_ref: pd.DataFrame) -> pd.DataFrame:
    """Purpose: Collect every model input via Streamlit widgets and assemble one aligned inference row.

    Steps:
        1. Use reference data (`df_ref`) for dropdown categories and sensible default medians.
        2. Lay out three columns of widgets (demographics/account, engagement/revenue, payments/support/marketing).
        3. Read widget values into a single-row dict matching training column names (excluding id/target).
        4. Wrap as a DataFrame and apply the same `complaint_type` NA handling as training.
        5. Return the one-row frame ready for `predict_churn`.
    """
    ref = df_ref
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", sorted(ref["gender"].dropna().unique()))
        age = st.number_input("Age", min_value=18, max_value=100, value=int(ref["age"].median()))
        country = st.selectbox("Country", sorted(ref["country"].dropna().unique()))
        city = st.selectbox("City", sorted(ref["city"].dropna().unique()))
        segment = st.selectbox(
            "Customer segment",
            sorted(ref["customer_segment"].dropna().unique()),
        )
        tenure_months = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            value=int(ref["tenure_months"].median()),
        )
        signup_channel = st.selectbox(
            "Signup channel",
            sorted(ref["signup_channel"].dropna().unique()),
        )
        contract_type = st.selectbox(
            "Contract type",
            sorted(ref["contract_type"].dropna().unique()),
        )
    with c2:
        monthly_logins = st.number_input(
            "Monthly logins",
            min_value=0,
            max_value=200,
            value=int(ref["monthly_logins"].median()),
        )
        weekly_active_days = st.number_input(
            "Weekly active days",
            min_value=0,
            max_value=7,
            value=int(ref["weekly_active_days"].median()),
        )
        avg_session_time = st.number_input(
            "Avg session time (min)",
            min_value=0.0,
            max_value=120.0,
            value=float(ref["avg_session_time"].median()),
            step=0.5,
        )
        features_used = st.number_input(
            "Features used",
            min_value=0,
            max_value=30,
            value=int(ref["features_used"].median()),
        )
        usage_growth_rate = st.number_input(
            "Usage growth rate",
            min_value=-1.0,
            max_value=1.0,
            value=float(ref["usage_growth_rate"].median()),
            step=0.01,
        )
        last_login_days_ago = st.number_input(
            "Last login (days ago)",
            min_value=0,
            max_value=90,
            value=int(ref["last_login_days_ago"].median()),
        )
        monthly_fee = st.number_input(
            "Monthly fee",
            min_value=0,
            max_value=500,
            value=int(ref["monthly_fee"].median()),
        )
        total_revenue = st.number_input(
            "Total revenue",
            min_value=0,
            max_value=500_000,
            value=int(ref["total_revenue"].median()),
            step=10,
        )
    with c3:
        payment_method = st.selectbox(
            "Payment method",
            sorted(ref["payment_method"].dropna().unique()),
        )
        payment_failures = st.number_input(
            "Payment failures",
            min_value=0,
            max_value=20,
            value=int(ref["payment_failures"].median()),
        )
        discount_applied = st.selectbox(
            "Discount applied",
            sorted(ref["discount_applied"].dropna().unique()),
        )
        price_increase_last_3m = st.selectbox(
            "Price increase (last 3m)",
            sorted(ref["price_increase_last_3m"].dropna().unique()),
        )
        support_tickets = st.number_input(
            "Support tickets",
            min_value=0,
            max_value=50,
            value=int(ref["support_tickets"].median()),
        )
        avg_resolution_time = st.number_input(
            "Avg resolution time",
            min_value=0.0,
            max_value=120.0,
            value=float(ref["avg_resolution_time"].median()),
            step=0.5,
        )
        complaint_opts = sorted(ref["complaint_type"].fillna("Unknown").astype(str).unique())
        complaint_type = st.selectbox("Complaint type", complaint_opts)
        csat_default = float(ref["csat_score"].median())
        if pd.isna(csat_default):
            csat_default = 3.0
        csat_score = st.number_input(
            "CSAT score",
            min_value=1.0,
            max_value=5.0,
            value=csat_default,
            step=0.5,
        )
        escalations = st.number_input(
            "Escalations",
            min_value=0,
            max_value=20,
            value=int(ref["escalations"].median()),
        )
        email_open_rate = st.slider(
            "Email open rate",
            min_value=0.0,
            max_value=1.0,
            value=float(ref["email_open_rate"].median()),
            step=0.01,
        )
        marketing_click_rate = st.slider(
            "Marketing click rate",
            min_value=0.0,
            max_value=1.0,
            value=float(ref["marketing_click_rate"].median()),
            step=0.01,
        )
        nps_score = st.number_input(
            "NPS score",
            min_value=-100,
            max_value=100,
            value=int(ref["nps_score"].median()),
        )
        survey_response = st.selectbox(
            "Survey response",
            sorted(ref["survey_response"].dropna().unique()),
        )
        referral_count = st.number_input(
            "Referral count",
            min_value=0,
            max_value=50,
            value=int(ref["referral_count"].median()),
        )

    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "country": country,
                "city": city,
                "customer_segment": segment,
                "tenure_months": tenure_months,
                "signup_channel": signup_channel,
                "contract_type": contract_type,
                "monthly_logins": monthly_logins,
                "weekly_active_days": weekly_active_days,
                "avg_session_time": avg_session_time,
                "features_used": features_used,
                "usage_growth_rate": usage_growth_rate,
                "last_login_days_ago": last_login_days_ago,
                "monthly_fee": monthly_fee,
                "total_revenue": total_revenue,
                "payment_method": payment_method,
                "payment_failures": payment_failures,
                "discount_applied": discount_applied,
                "price_increase_last_3m": price_increase_last_3m,
                "support_tickets": support_tickets,
                "avg_resolution_time": avg_resolution_time,
                "complaint_type": complaint_type,
                "csat_score": csat_score,
                "escalations": escalations,
                "email_open_rate": email_open_rate,
                "marketing_click_rate": marketing_click_rate,
                "nps_score": nps_score,
                "survey_response": survey_response,
                "referral_count": referral_count,
            }
        ]
    )
    return fill_complaint_type_na(row)


def main() -> None:
    """Purpose: Render the full Streamlit dashboard: sidebar stats, EDA tabs, and conditional churn prediction.

    Steps:
        1. Load the dataset through the cached loader and show row count / churn rate in the sidebar (warn if no model file).
        2. Overview tab: numeric describe table + churn class bar chart.
        3. Explore tab: segment and payment-method aggregates, plus a selectable numeric histogram.
        4. Predict tab: if artifact exists, load cached model, render `prediction_form`, and on button click run `predict_churn` and display label + probability + threshold; otherwise show training instructions.
    """
    st.title("Customer churn dashboard")
    path = str(DEFAULT_DATA_PATH)
    df = load_dataset(path)

    with st.sidebar:
        st.header("Dataset")
        st.caption(str(DEFAULT_DATA_PATH.name))
        st.metric("Rows", f"{len(df):,}")
        churn_rate = df["churn"].mean()
        st.metric("Churn rate", f"{churn_rate:.1%}")
        if not model_available():
            st.warning(
                f"No trained model at `{MODEL_PATH.name}`. Run `python scripts/train_model.py` first."
            )

    tab_overview, tab_explore, tab_impact, tab_predict = st.tabs(
        ["Overview", "Explore", "Feature impact", "Predict"]
    )

    with tab_overview:
        st.subheader("Numeric summary")
        st.dataframe(analysis.analyze_data(df), use_container_width=True)
        st.subheader("Churn mix")
        mix = df["churn"].value_counts().rename(index={0: "Stay", 1: "Churn"})
        st.bar_chart(mix)

    with tab_explore:
        c1, c2 = st.columns(2)
        seg_counts = (
            df.groupby(["customer_segment", "churn"], observed=False)
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "stay", 1: "churn"})
        )
        with c1:
            st.markdown("**Churn by segment**")
            st.bar_chart(seg_counts)
        pay_churn = df.groupby("payment_method", observed=False)["churn"].mean().sort_values(ascending=False)
        with c2:
            st.markdown("**Churn rate by payment method**")
            st.bar_chart(pay_churn)
        num_cols = df.select_dtypes(include=["number"]).columns.drop("churn", errors="ignore")
        pick = st.selectbox("Distribution", options=list(num_cols))
        st.markdown(f"**Histogram: {pick}**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df[pick].dropna(), bins=40, edgecolor="white", alpha=0.85)
        ax.set_xlabel(pick)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with tab_impact:
        st.subheader("Feature impact")
        st.caption(
            "Permutation importance reflects how much each input column affects the model score when shuffled. "
            "Model-based importance comes from the random forest and is aggregated across one-hot encoded categories."
        )

        if not model_available():
            st.info(
                "Train the model from the project root: `python scripts/train_model.py`, then refresh this page."
            )
        else:
            pipe, _threshold = cached_model()

            df_clean = fill_complaint_type_na(df)
            feature_cols = [c for c in df_clean.columns if c not in ("customer_id", TARGET)]
            X = df_clean[feature_cols]
            y = df_clean[TARGET]

            # Match the same test split used in training for a comparable view.
            _X_train, X_test, _y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Permutation importance on original input columns (interpretable).
            with st.spinner("Computing permutation importance (this may take ~10-30s)..."):
                perm = permutation_importance_by_feature(
                    pipe, X_test, y_test, scoring="f1", n_repeats=5, random_state=42
                )

            top_n = st.slider("Show top N features", min_value=5, max_value=30, value=15, step=1)
            st.markdown("**Permutation importance (F1 impact, higher = more important)**")
            st.dataframe(perm.head(top_n), use_container_width=True)
            st.bar_chart(perm.head(top_n).set_index("feature")["importance_mean"])

            st.markdown("**Model-based importance (random forest, aggregated to original columns)**")
            rf_imp = aggregated_model_importance(pipe)
            st.dataframe(rf_imp.head(top_n), use_container_width=True)
            st.bar_chart(rf_imp.head(top_n).set_index("feature")["importance"])

            st.markdown("**Simple feature selection suggestion**")
            st.write(
                "A simple starting point is to keep the top features by permutation importance. "
                "This is not retraining yet—just a recommendation list."
            )
            st.code(", ".join(perm["feature"].head(10).tolist()))

    with tab_predict:
        if not model_available():
            st.info(
                "Train the model from the project root: `python scripts/train_model.py`, then refresh this page."
            )
        else:
            pipe, threshold = cached_model()
            features = prediction_form(df)
            if st.button("Predict churn", type="primary"):
                pred, proba = predict_churn(pipe, features, threshold=threshold)
                label = "Likely to churn" if pred == 1 else "Likely to stay"
                st.success(
                    f"**{label}** (estimated churn probability: **{proba:.1%}**; decision threshold **{threshold:.2f}**)"
                )


if __name__ == "__main__":
    main()
