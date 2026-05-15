# Fraud Detection System

A production-grade machine learning system for detecting fraudulent credit card transactions using Python, XGBoost, SHAP, and FastAPI.

---

## Overview

This project demonstrates an end-to-end data science workflow:

* Exploratory Data Analysis (EDA)
* Baseline modeling with Logistic Regression
* Advanced modeling with XGBoost
* Threshold optimization
* Model explainability with SHAP
* Model artifact persistence
* REST API development with FastAPI
* Version control with Git and GitHub

The goal is to build a deployable fraud detection platform that can score transactions in real time and explain why a transaction was flagged.

---

## Business Problem

Financial institutions process millions of transactions daily. A small percentage are fraudulent, but the financial impact can be significant.

This project predicts whether a credit card transaction is fraudulent and returns:

* Fraud probability
* Predicted class (fraud or legitimate)
* Risk level
* Recommended action

---

## Dataset

This project uses the publicly available Credit Card Fraud Detection dataset.

**Dataset Source:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Dataset Characteristics

* 284,807 transactions
* 31 columns
* 492 fraudulent transactions
* Fraud rate: ~0.17%

### Columns

* `Time`
* `V1` to `V28` (PCA-transformed anonymized features)
* `Amount`
* `Class` (target)

### Note

The dataset is not included in this repository due to GitHub file size limits.

Download `creditcard.csv` from Kaggle and place it in:

```text
data/raw/creditcard.csv
```

---

## Project Structure

```text
fraud-detection-system/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── models/
│   ├── xgboost_model.pkl
│   ├── best_threshold.pkl
│   ├── feature_names.pkl
│   └── metadata.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_xgboost_model.ipynb
├── reports/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── utils/
├── tests/
├── .github/workflows/
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Technology Stack

### Programming Language

* Python

### Data Science Libraries

* pandas
* numpy
* scikit-learn
* xgboost
* shap
* matplotlib

### Backend

* FastAPI
* Uvicorn

### Model Persistence

* joblib

### Experiment Tracking (planned)

* MLflow

### Monitoring (planned)

* Evidently

### Deployment (planned)

* Docker
* AWS / Render / Railway

### Version Control

* Git
* GitHub

---

# Work Completed So Far

## Phase 1: Project Setup

Completed:

* Created project folder structure
* Set up Python virtual environment
* Installed dependencies
* Initialized Git repository

### Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Phase 2: Data Acquisition

Completed:

* Downloaded dataset from Kaggle
* Stored it under `data/raw/`

---

## Phase 3: Exploratory Data Analysis (EDA)

Notebook: `notebooks/01_eda.ipynb`

Completed:

* Loaded dataset
* Checked shape and data types
* Verified missing values
* Analyzed class imbalance
* Visualized class distribution
* Studied transaction amount distributions
* Calculated feature correlations

### Key Findings

* No missing values
* Fraud rate is approximately 0.17%
* Data is highly imbalanced
* `V1`–`V28` are PCA-transformed features
* Transaction amounts are right-skewed

---

## Phase 4: Baseline Model (Logistic Regression)

Notebook: `notebooks/02_baseline_model.ipynb`

Completed:

* Train/test split using stratification
* Feature scaling with StandardScaler
* Logistic Regression with `class_weight='balanced'`
* Probability predictions
* Evaluation using:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * PR-AUC

### Purpose

Established a benchmark for comparing more advanced models.

---

## Phase 5: Advanced Model (XGBoost)

Notebook: `notebooks/03_xgboost_model.ipynb`

Completed:

* Calculated `scale_pos_weight`
* Trained XGBoost classifier
* Generated fraud probabilities
* Evaluated performance
* Compared with baseline model
* Visualized feature importance

### Why XGBoost?

* Excellent performance on tabular data
* Handles non-linear relationships
* Supports severe class imbalance

---

## Phase 5B: Threshold Optimization

Completed:

* Computed precision-recall curve
* Calculated F1-score across thresholds
* Selected best threshold
* Generated custom predictions

### Outcome

Improved operational balance between precision and recall.

---

## Phase 5C: SHAP Explainability

Completed:

* Generated global feature importance
* Created summary plots
* Built per-transaction explanations

### Purpose

Explains why a transaction is predicted as fraud.

---

## Phase 6: Model Artifact Persistence

Completed:

Saved the following artifacts:

* `models/xgboost_model.pkl`
* `models/best_threshold.pkl`
* `models/feature_names.pkl`
* `models/metadata.pkl`

### Metadata Includes

* ROC-AUC
* PR-AUC
* Threshold
* Number of features

---

## Phase 7: FastAPI Prediction API

File: `src/api/main.py`

Completed:

* Loaded saved model artifacts
* Defined input schema using Pydantic
* Created `GET /health`
* Created `POST /predict`
* Returned:

  * Fraud probability
  * Prediction
  * Risk level
  * Recommended action

### Run the API

```bash
uvicorn src.api.main:app --reload
```

### Interactive Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Git & GitHub Setup

Completed:

* Created `.gitignore`
* Excluded `data/` and `.venv/`
* Removed large dataset files from Git tracking
* Pushed repository to GitHub

Repository URL:

```text
https://github.com/aspabhi31/fraud-detection-system
```

---

# Machine Learning Metrics Used

## Precision

Measures how many flagged transactions are actually fraud.

## Recall

Measures how many fraud cases are detected.

## F1-Score

Balances precision and recall.

## ROC-AUC

Measures ranking quality across thresholds.

## PR-AUC

Primary metric for highly imbalanced fraud detection problems.

---

# API Example Response

```json
{
  "fraud_probability": 0.923451,
  "prediction": 1,
  "threshold": 0.231004,
  "risk_level": "High",
  "recommended_action": "Block transaction"
}
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/aspabhi31/fraud-detection-system.git
cd fraud-detection-system
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the Dataset

Download from Kaggle and place in:

```text
data/raw/creditcard.csv
```

## 5. Run the API

```bash
uvicorn src.api.main:app --reload
```

## 6. Open Swagger Docs

```text
http://127.0.0.1:8000/docs
```

---

# Current Project Status

## Completed

* Project setup
* EDA
* Baseline model
* XGBoost model
* Threshold tuning
* SHAP explainability
* Model saving
* FastAPI backend
* GitHub repository

## In Progress

* Documentation refinement

## Planned

* Streamlit frontend
* MLflow experiment tracking
* Monitoring with Evidently
* Dockerization
* Cloud deployment
* CI/CD with GitHub Actions

---

# Planned Architecture

```text
Transaction Data
      ↓
Feature Engineering
      ↓
XGBoost Model
      ↓
Threshold Optimization
      ↓
SHAP Explainability
      ↓
FastAPI Prediction Service
      ↓
Streamlit Dashboard (Planned)
      ↓
Docker & Cloud Deployment (Planned)
```

---

# Resume Bullet

Built a production-grade fraud detection system using XGBoost, SHAP, and FastAPI to detect credit card fraud on a highly imbalanced dataset, with threshold optimization, explainable predictions, and deployable REST API endpoints.

---

# Key Skills Demonstrated

* Exploratory Data Analysis
* Imbalanced Classification
* Logistic Regression
* XGBoost
* Threshold Optimization
* Explainable AI (SHAP)
* Model Persistence
* REST API Development
* Git and GitHub
* Production-Oriented Machine Learning

---

# Future Enhancements

* Interactive Streamlit dashboard
* Batch prediction upload
* MLflow experiment tracking
* Data drift monitoring
* Dockerized deployment
* AWS hosting
* Automated CI/CD

---

# Author

Abhijeet Singh Pawar

* GitHub: [https://github.com/aspabhi31](https://github.com/aspabhi31)
* LinkedIn: https://www.linkedin.com/in/abhijeet-singh-pawar-482576149/

---

# License

This project is for educational and portfolio purposes.
