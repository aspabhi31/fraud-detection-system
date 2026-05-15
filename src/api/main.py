from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load artifacts at startup
model = joblib.load("models/xgboost_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")
feature_names = joblib.load("models/feature_names.pkl")

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

# Input schema
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: TransactionInput):
    data = pd.DataFrame([transaction.model_dump()])
    data = data[feature_names]

    probability = model.predict_proba(data)[:, 1][0]
    prediction = int(probability >= threshold)

    if probability >= 0.90:
        risk_level = "High"
        action = "Block transaction"
    elif probability >= threshold:
        risk_level = "Medium"
        action = "Manual review"
    else:
        risk_level = "Low"
        action = "Approve transaction"

    return {
        "fraud_probability": round(float(probability), 6),
        "prediction": prediction,
        "threshold": round(float(threshold), 6),
        "risk_level": risk_level,
        "recommended_action": action
    }