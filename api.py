# ============================================================
# api.py
# FastAPI backend — IndusCredit Finance Credit Risk System
# Endpoints: /predict, /batch_predict, /segment_analysis,
#            /model_info, /health
# ============================================================

import os
import io
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# ============================================================
# LOAD ARTIFACTS AT STARTUP
# ============================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model         = load_pkl("best_model.pkl")
model_name    = load_pkl("best_model_name.pkl")
target_enc    = load_pkl("target_encoder.pkl")
train_columns = load_pkl("train_columns.pkl")
train_medians = load_pkl("train_medians.pkl")
threshold     = load_pkl("best_threshold.pkl")
scaler        = load_pkl("scaler.pkl") if (model_name == "LR" and os.path.exists("scaler.pkl")) else None

# ============================================================
# PREPROCESSING FUNCTION (mirrors training pipeline exactly)
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop loan_id if present
    if "loan_id" in df.columns:
        df.drop(columns=["loan_id"], inplace=True)

    # Drop label if accidentally included
    if "default_flag" in df.columns:
        df.drop(columns=["default_flag"], inplace=True)

    # Date features
    df["application_date"]  = pd.to_datetime(df["application_date"])
    df["application_month"] = df["application_date"].dt.month
    df["application_year"]  = df["application_date"].dt.year
    df.drop(columns=["application_date"], inplace=True)

    # LTV
    df["ltv_missing_flag"] = df["ltv_ratio"].isnull().astype(int)
    df["ltv_ratio"]        = df["ltv_ratio"].fillna(-1)

    # Numeric nulls → training medians
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if col in train_medians.index:
            df[col] = df[col].fillna(train_medians[col])

    # Categorical nulls
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Feature engineering
    df["loan_to_income_ratio"]     = df["loan_amount_inr"] / df["annual_income_inr"]
    df["dti_credit_risk"]          = df["dti_ratio"] / (df["credit_score"] / 700)
    df["income_per_year_employed"] = df["annual_income_inr"] / (df["employment_years"] + 1)
    df["emi_estimate"]             = df["loan_amount_inr"] / df["loan_tenure_months"]
    df["loan_per_credit_score"]    = df["loan_amount_inr"] / df["credit_score"]
    df["high_dti"]                 = (df["dti_ratio"] > 0.4).astype(int)
    df["is_new_employee"]          = (df["employment_years"] < 2).astype(int)
    df["savings_to_loan_ratio"]    = df["savings_account_balance_inr"] / (df["loan_amount_inr"] + 1)
    df["credit_enquiry_risk"]      = df["bureau_enquiries_6m"] * df["dti_ratio"]
    df["repayment_risk_score"]     = df["missed_payments_2y"] * (1 / (df["credit_score"] / 700))
    df["loan_burden"]              = df["num_existing_loans"] * df["dti_ratio"]
    df["interest_x_tenure"]        = df["interest_rate_pct"] * df["loan_tenure_months"]

    # Encoding
    education_map = {
        "No_Formal": 0, "Diploma": 1, "Undergraduate": 2,
        "Graduate": 3, "Post_Graduate": 4, "Unknown": -1
    }
    df["education"] = df["education"].map(education_map).fillna(-1).astype(int)

    one_hot_cols = ["gender", "urban_rural", "employment_type", "loan_type", "loan_purpose"]
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Target encode state
    df["state"] = target_enc.transform(df[["state"]])

    # Align to training columns
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[train_columns]

    # Scale if LR
    if scaler is not None:
        num_cols_s = df.select_dtypes(include=np.number).columns
        df[num_cols_s] = scaler.transform(df[num_cols_s])

    return df


def risk_band(prob: float) -> str:
    if prob < 0.3:   return "Low"
    elif prob < 0.5: return "Medium"
    elif prob < 0.7: return "High"
    else:            return "Very High"


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="IndusCredit Finance — Credit Risk API",
    description="ML-based loan default risk scoring API. Built for credit officers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ───────────────────────────────
class ApplicantInput(BaseModel):
    loan_id:                      Optional[str] = "N/A"
    application_date:             str           = Field(..., example="2024-01-15")
    age:                          int           = Field(..., ge=21, le=65)
    gender:                       str           = Field(..., example="Male")
    education:                    str           = Field(..., example="Graduate")
    state:                        str           = Field(..., example="MH")
    urban_rural:                  str           = Field(..., example="Urban")
    employment_type:              str           = Field(..., example="Salaried")
    employment_years:             int           = Field(..., ge=0)
    annual_income_inr:            float
    loan_type:                    str           = Field(..., example="Home_Loan")
    loan_purpose:                 str           = Field(..., example="Purchase")
    loan_amount_inr:              float
    loan_tenure_months:           int
    interest_rate_pct:            float
    credit_score:                 int           = Field(..., ge=550, le=900)
    num_existing_loans:           int           = Field(..., ge=0)
    dti_ratio:                    float         = Field(..., ge=0.0, le=0.65)
    ltv_ratio:                    Optional[float] = None
    has_collateral:               int           = Field(..., ge=0, le=1)
    bureau_enquiries_6m:          int           = Field(..., ge=0)
    missed_payments_2y:           int           = Field(..., ge=0)
    savings_account_balance_inr:  float


class PredictionResponse(BaseModel):
    loan_id:              str
    default_probability:  float
    predicted_default:    int
    risk_band:            str
    decision:             str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "threshold": round(threshold, 4)}


@app.get("/model_info")
def model_info():
    return {
        "model_name":  model_name,
        "threshold":   round(threshold, 4),
        "n_features":  len(train_columns),
        "top_features": [
            "repayment_risk_score",
            "missed_payments_2y",
            "credit_score",
            "dti_ratio",
            "bureau_enquiries_6m",
            "dti_credit_risk",
            "num_existing_loans",
            "has_collateral",
            "loan_burden",
            "credit_enquiry_risk",
        ],
        "description": (
            f"{model_name} model trained on 8,000 loan records. "
            f"AUC-ROC: 0.9011. Threshold optimized via PR curve."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantInput):
    try:
        df = pd.DataFrame([applicant.dict()])
        X  = preprocess(df)
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= threshold)
        return PredictionResponse(
            loan_id             = applicant.loan_id,
            default_probability = round(prob, 4),
            predicted_default   = pred,
            risk_band           = risk_band(prob),
            decision            = "REJECT" if pred == 1 else "APPROVE",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Upload a CSV file (same schema as loan_test.csv) for bulk scoring."""
    try:
        contents = await file.read()
        df_raw   = pd.read_csv(io.BytesIO(contents))
        loan_ids = df_raw["loan_id"].tolist() if "loan_id" in df_raw.columns else list(range(len(df_raw)))

        X    = preprocess(df_raw)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)

        results = []
        for i, (lid, p, d) in enumerate(zip(loan_ids, prob, pred)):
            results.append({
                "loan_id":             str(lid),
                "default_probability": round(float(p), 4),
                "predicted_default":   int(d),
                "risk_band":           risk_band(float(p)),
                "decision":            "REJECT" if d == 1 else "APPROVE",
            })

        approve_count = sum(1 for r in results if r["decision"] == "APPROVE")
        reject_count  = len(results) - approve_count

        return {
            "total":          len(results),
            "approved":       approve_count,
            "rejected":       reject_count,
            "approval_rate":  round(approve_count / len(results) * 100, 1),
            "predictions":    results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segment_analysis")
def segment_analysis():
    """
    Returns pre-computed default rates by segment from the training set.
    Used by the dashboard for segment-wise risk charts.
    """
    try:
        df = pd.read_csv("loan_train.csv")
        result = {}
        for col in ["loan_type", "employment_type", "urban_rural", "gender", "education"]:
            rates = df.groupby(col)["default_flag"].mean().round(4)
            result[col] = {str(k): float(v) for k, v in rates.items()}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
