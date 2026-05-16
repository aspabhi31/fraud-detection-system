import streamlit as st
import requests
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"

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
model = joblib.load("models/xgboost_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
explainer = shap.TreeExplainer(model)

# =========================
# SINGLE PREDICTION
# =========================
st.header("🔍 Single Transaction Prediction")

features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

payload = {}

with st.form("single_form"):
    for f in features:
        default = 0.0 if f != "Amount" else 100.0
        payload[f] = st.number_input(f, value=float(default))

    submit = st.form_submit_button("Predict")

if submit:
    response = requests.post(API_URL_SINGLE, json=payload)
    result = response.json()

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
    col2.metric("Prediction", "Fraud" if result["prediction"] == 1 else "Legit")
    col3.metric("Risk Level", result["risk_level"])

    st.info(result["recommended_action"])

    # SHAP explanation
    if st.button("Explain Prediction (SHAP)"):
        input_df = pd.DataFrame([payload])[feature_names]
        shap_values = explainer.shap_values(input_df)

        st.subheader("Feature Importance")

        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig1)

        st.subheader("Detailed Explanation")

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        )

        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig2)

# =========================
# BULK PREDICTION (OPTIMIZED)
# =========================
st.header("📂 Bulk Fraud Detection (FAST Batch API)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head())

    # Convert to batch format
    payload = df.to_dict(orient="records")

    st.write("Running batch prediction...")

    response = requests.post(API_URL_BATCH, json=payload)

    results = response.json()["results"]

    results_df = pd.DataFrame(results)

    output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    st.subheader("Results")
    st.dataframe(output_df)

    # =========================
    # SUMMARY
    # =========================
    st.subheader("Summary")

    total = len(output_df)
    fraud_count = output_df["prediction"].sum()
    legit_count = total - fraud_count

    col1, col2, col3 = st.columns(3)

    col1.metric("Total", total)
    col2.metric("Fraud", fraud_count)
    col3.metric("Legit", legit_count)

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("Fraud Probability Distribution")
    st.bar_chart(output_df["fraud_probability"])

    # =========================
    # DOWNLOAD
    # =========================
    csv = output_df.to_csv(index=False)

    st.download_button(
        "Download Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )