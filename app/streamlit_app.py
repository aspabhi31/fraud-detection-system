import os
import streamlit as st
import requests
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
# Uses environment variable in production, falls back to local API for development.
# Local default:
#   http://127.0.0.1:8000
#
# Production example:
#   export API_BASE_URL="http://YOUR_PUBLIC_IP:8000"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

API_URL_SINGLE = f"{API_BASE_URL}/predict"
API_URL_BATCH = f"{API_BASE_URL}/predict_batch"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection System")
st.write("Fast batch inference + explainable AI")

# =========================
# LOAD MODEL (FOR SHAP)
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer

model, feature_names, explainer = load_artifacts()

# =========================
# SINGLE PREDICTION
# =========================
st.header("🔍 Single Transaction Prediction")

features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

payload = {}

with st.form("single_form"):
    for feature in features:
        default = 0.0 if feature != "Amount" else 100.0
        payload[feature] = st.number_input(
            feature,
            value=float(default),
            format="%.6f"
        )

    submit = st.form_submit_button("Predict")

if submit:
    try:
        response = requests.post(API_URL_SINGLE, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Fraud Probability",
            f"{result['fraud_probability']:.4f}"
        )

        col2.metric(
            "Prediction",
            "Fraud" if result["prediction"] == 1 else "Legitimate"
        )

        col3.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.info(result["recommended_action"])

        # Store payload for SHAP explanation
        st.session_state["last_payload"] = payload.copy()

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

# =========================
# SHAP EXPLANATION
# =========================
if "last_payload" in st.session_state:
    if st.button("Explain Prediction (SHAP)"):
        input_df = pd.DataFrame([st.session_state["last_payload"]])
        input_df = input_df[feature_names]

        shap_values = explainer.shap_values(input_df)

        st.subheader("Feature Importance")

        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values,
            input_df,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig1, clear_figure=True)

        st.subheader("Detailed Explanation")

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        )

        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig2, clear_figure=True)

# =========================
# BULK PREDICTION
# =========================
st.header("📂 Bulk Fraud Detection (Batch API)")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        try:
            payload = df.to_dict(orient="records")

            with st.spinner("Running batch prediction..."):
                response = requests.post(
                    API_URL_BATCH,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()

            results = response.json()["results"]
            results_df = pd.DataFrame(results)

            output_df = pd.concat(
                [df.reset_index(drop=True), results_df],
                axis=1
            )

            st.subheader("Results")
            st.dataframe(output_df)

            # Summary metrics
            st.subheader("Summary")

            total = len(output_df)
            fraud_count = int(output_df["prediction"].sum())
            legitimate_count = total - fraud_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraud Detected", fraud_count)
            col3.metric("Legitimate", legitimate_count)

            # Probability chart
            st.subheader("Fraud Probability Distribution")
            st.bar_chart(output_df["fraud_probability"])

            # Download button
            csv_data = output_df.to_csv(index=False)

            st.download_button(
                label="Download Predictions",
                data=csv_data,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Batch API request failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")