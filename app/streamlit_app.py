# `app/streamlit_app.py` (Standalone Version for Streamlit Community Cloud)
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection System")
st.write("Real-time and batch fraud detection with SHAP explainability")

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    # Optional artifacts
    scaler = None
    try:
        scaler = joblib.load("models/scaler.pkl")
    except:
        pass

    explainer = shap.TreeExplainer(model)

    return model, feature_names, scaler, explainer


model, feature_names, scaler, explainer = load_artifacts()


# =========================
# PREDICTION FUNCTION
# =========================
def predict_transaction(input_df):
    X = input_df[feature_names]

    if scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    fraud_probability = float(model.predict_proba(X_processed)[:, 1][0])
    prediction = int(fraud_probability >= 0.5)

    # Risk level logic
    if fraud_probability < 0.10:
        risk_level = "Low"
        action = "Approve transaction"
    elif fraud_probability < 0.50:
        risk_level = "Medium"
        action = "Review transaction"
    elif fraud_probability < 0.90:
        risk_level = "High"
        action = "Flag for investigation"
    else:
        risk_level = "Critical"
        action = "Block transaction immediately"

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "recommended_action": action,
        "X_processed": X_processed,
        "X_original": X,
    }


# =========================
# SINGLE PREDICTION
# =========================
st.header("🔍 Single Transaction Prediction")

features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

payload = {}

with st.form("single_prediction_form"):
    col1, col2, col3 = st.columns(3)

    for idx, feature in enumerate(features):
        default_value = 100.0 if feature == "Amount" else 0.0

        if idx % 3 == 0:
            with col1:
                payload[feature] = st.number_input(
                    feature,
                    value=float(default_value),
                    format="%.6f"
                )
        elif idx % 3 == 1:
            with col2:
                payload[feature] = st.number_input(
                    feature,
                    value=float(default_value),
                    format="%.6f"
                )
        else:
            with col3:
                payload[feature] = st.number_input(
                    feature,
                    value=float(default_value),
                    format="%.6f"
                )

    submit = st.form_submit_button("Predict Fraud")


if submit:
    try:
        input_df = pd.DataFrame([payload])
        result = predict_transaction(input_df)

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

        st.info(f"Recommended Action: {result['recommended_action']}")

        # Save for SHAP explanation
        st.session_state["last_result"] = result

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================
# SHAP EXPLANATION
# =========================
if "last_result" in st.session_state:
    st.header("🧠 SHAP Explainability")

    if st.button("Explain Prediction"):
        try:
            result = st.session_state["last_result"]
            X_original = result["X_original"]

            shap_values = explainer.shap_values(X_original)

            # Bar plot
            st.subheader("Feature Importance")
            fig1 = plt.figure()
            shap.summary_plot(
                shap_values,
                X_original,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig1, clear_figure=True)

            # Waterfall plot
            st.subheader("Detailed Contribution")
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_original.iloc[0],
                feature_names=feature_names
            )

            fig2 = plt.figure()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig2, clear_figure=True)

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")


# =========================
# BULK PREDICTION
# =========================
st.header("📂 Bulk Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing transactions",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df.head())

        missing_cols = [col for col in feature_names if col not in df.columns]

        if missing_cols:
            st.error(
                "Missing required columns: " + ", ".join(missing_cols)
            )
        else:
            if st.button("Run Batch Prediction"):
                with st.spinner("Scoring transactions..."):
                    results = []

                    for _, row in df.iterrows():
                        row_df = pd.DataFrame([row[feature_names]])
                        result = predict_transaction(row_df)

                        results.append({
                            "prediction": result["prediction"],
                            "fraud_probability": result["fraud_probability"],
                            "risk_level": result["risk_level"],
                            "recommended_action": result["recommended_action"],
                        })

                    results_df = pd.DataFrame(results)
                    output_df = pd.concat(
                        [df.reset_index(drop=True), results_df],
                        axis=1
                    )

                st.subheader("Results")
                st.dataframe(output_df)

                # Summary metrics
                total = len(output_df)
                fraud_count = int(output_df["prediction"].sum())
                legitimate_count = total - fraud_count

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", total)
                col2.metric("Fraud Detected", fraud_count)
                col3.metric("Legitimate", legitimate_count)

                # Chart
                st.subheader("Fraud Probability Distribution")
                st.bar_chart(output_df["fraud_probability"])

                # Download
                csv_data = output_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Failed to process file: {e}")


