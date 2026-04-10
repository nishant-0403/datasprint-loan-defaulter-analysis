# ============================================================
# api.py
# FastAPI backend — IndusCredit Finance Credit Risk System
# Endpoints: /predict, /explain, /batch_predict,
#            /segment_analysis, /model_info, /health
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
import shap
import google.generativeai as genai

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

# ── SHAP Explainer (loaded once at startup for performance) ──
if model_name == "LR":
    # LinearExplainer requires background data; load a sample from training set
    _bg = pd.read_csv("loan_train.csv").head(200)
    # We need preprocessed background — build a minimal version
    # (full preprocessing requires the label column stripped first)
    _bg_X = _bg.drop(columns=["default_flag", "loan_id"], errors="ignore")
    shap_explainer = None          # lazy-init on first /explain call (needs preprocessed bg)
    _bg_raw = _bg                  # store raw for lazy init
else:
    shap_explainer = shap.TreeExplainer(model)

# ── Gemini client ────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDrblMHudVePQR3IB4g7IDAcbd-2g67igM"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_client = None

# ── Human-readable feature labels ───────────────────────────
FEATURE_LABELS = {
    "repayment_risk_score":      "Repayment Risk Score",
    "missed_payments_2y":        "Missed Payments (last 2 years)",
    "credit_score":              "Credit Score",
    "dti_ratio":                 "Debt-to-Income Ratio",
    "dti_credit_risk":           "DTI adjusted for Credit Score",
    "bureau_enquiries_6m":       "Credit Bureau Enquiries (6 months)",
    "num_existing_loans":        "Number of Existing Loans",
    "has_collateral":            "Has Collateral",
    "loan_burden":               "Loan Burden (existing loans × DTI)",
    "credit_enquiry_risk":       "Credit Enquiry Risk",
    "loan_to_income_ratio":      "Loan-to-Income Ratio",
    "loan_amount_inr":           "Loan Amount",
    "annual_income_inr":         "Annual Income",
    "savings_to_loan_ratio":     "Savings-to-Loan Ratio",
    "interest_rate_pct":         "Interest Rate",
    "interest_x_tenure":         "Interest × Loan Tenure",
    "employment_years":          "Years of Employment",
    "income_per_year_employed":  "Income per Year Employed",
    "loan_tenure_months":        "Loan Tenure (months)",
    "emi_estimate":              "Estimated Monthly EMI",
    "loan_per_credit_score":     "Loan Amount per Credit Score Point",
    "ltv_ratio":                 "Loan-to-Value Ratio",
    "ltv_missing_flag":          "LTV Not Applicable (no collateral)",
    "high_dti":                  "High DTI Flag (>40%)",
    "is_new_employee":           "New Employee Flag (<2 years)",
    "age":                       "Applicant Age",
    "education":                 "Education Level",
    "application_month":         "Application Month",
    "application_year":          "Application Year",
}

def feature_label(col: str) -> str:
    """Return a human-readable label for a feature column."""
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    # One-hot columns like gender_Male, employment_type_Salaried
    for prefix in ["gender_", "urban_rural_", "employment_type_",
                   "loan_type_", "loan_purpose_"]:
        if col.startswith(prefix):
            category = col[len(prefix):].replace("_", " ")
            group    = prefix.rstrip("_").replace("_", " ").title()
            return f"{group}: {category}"
    return col.replace("_", " ").title()


# ============================================================
# PREPROCESSING FUNCTION (mirrors training pipeline exactly)
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "loan_id" in df.columns:
        df.drop(columns=["loan_id"], inplace=True)
    if "default_flag" in df.columns:
        df.drop(columns=["default_flag"], inplace=True)

    df["application_date"]  = pd.to_datetime(df["application_date"])
    df["application_month"] = df["application_date"].dt.month
    df["application_year"]  = df["application_date"].dt.year
    df.drop(columns=["application_date"], inplace=True)

    df["ltv_missing_flag"] = df["ltv_ratio"].isnull().astype(int)
    df["ltv_ratio"]        = df["ltv_ratio"].fillna(-1)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if col in train_medians.index:
            df[col] = df[col].fillna(train_medians[col])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

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

    education_map = {
        "No_Formal": 0, "Diploma": 1, "Undergraduate": 2,
        "Graduate": 3, "Post_Graduate": 4, "Unknown": -1
    }
    df["education"] = df["education"].map(education_map).fillna(-1).astype(int)

    one_hot_cols = ["gender", "urban_rural", "employment_type", "loan_type", "loan_purpose"]
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    df["state"] = target_enc.transform(df[["state"]])

    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[train_columns]

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
# GEMINI — PLAIN LANGUAGE EXPLANATION
# ============================================================
def gemini_explain(
    default_probability: float,
    risk_band_label: str,
    top_factors: list,
    applicant_context: dict,
) -> str:
    """
    Call Gemini to generate a plain-language explanation for a loan rejection.
    Falls back to a template-based explanation if Gemini is unavailable.
    """
    # ── Build factor summary for the prompt ──────────────────
    factor_lines = []
    for i, f in enumerate(top_factors, 1):
        direction = "increases" if f["direction"] == "increases_risk" else "reduces"
        factor_lines.append(
            f"  {i}. {f['feature_label']} — {direction} default risk "
            f"(contribution: {f['shap_value']:+.3f})"
        )
    factors_text = "\n".join(factor_lines)

    prompt = f"""You are a credit officer assistant at IndusCredit Finance in India.
A loan application has been flagged as HIGH RISK by our ML model.

Applicant summary:
- Default probability: {default_probability*100:.1f}%
- Risk band: {risk_band_label}
- Loan type: {applicant_context.get('loan_type', 'N/A')}
- Loan amount: ₹{applicant_context.get('loan_amount_inr', 0):,.0f}
- Annual income: ₹{applicant_context.get('annual_income_inr', 0):,.0f}

Top 5 risk factors identified by our model:
{factors_text}

Write a clear, sizeable explanation for the credit officer explaining WHY this
application was rejected. Use plain English — no ML jargon, no mention of SHAP or model internals.
Focus on what the numbers mean for the applicant's ability to repay. Do not suggest remedies unless
they are obvious (e.g. "reducing the loan amount"). Be professional and neutral in tone."""

    if gemini_client:
        try:
            response = gemini_client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            import traceback
            print(f"[Gemini ERROR] {e}\n{traceback.format_exc()}")
            # Fall through to template

    # ── Template fallback (no Gemini key or API error) ───────
    top2 = top_factors[:2]
    lines = [
        f"This application carries a {default_probability*100:.1f}% probability of default, "
        f"placing it in the '{risk_band_label}' risk category."
    ]
    for f in top2:
        if f["direction"] == "increases_risk":
            lines.append(
                f"The applicant's {f['feature_label'].lower()} is a significant concern "
                f"that raises the likelihood of non-repayment."
            )
        else:
            lines.append(
                f"While the applicant's {f['feature_label'].lower()} is a positive indicator, "
                f"it was outweighed by other risk factors."
            )
    lines.append(
        "Based on the overall risk profile, the application does not meet our current "
        "credit acceptance criteria."
    )
    return " ".join(lines)


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="IndusCredit Finance — Credit Risk API",
    description="ML-based loan default risk scoring API. Built for credit officers.",
    version="1.1.0",
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


class RiskFactor(BaseModel):
    rank:           int
    feature:        str
    feature_label:  str
    shap_value:     float
    direction:      str   # "increases_risk" | "reduces_risk"


class ExplainResponse(BaseModel):
    loan_id:              str
    default_probability:  float
    predicted_default:    int
    risk_band:            str
    decision:             str
    top_factors:          List[RiskFactor]
    plain_explanation:    str
    gemini_used:          bool


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model":          model_name,
        "threshold":      round(threshold, 4),
        "gemini_enabled": bool(gemini_client),
    }


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
        df   = pd.DataFrame([applicant.dict()])
        X    = preprocess(df)
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


@app.post("/explain", response_model=ExplainResponse)
def explain(applicant: ApplicantInput):
    """
    Returns the prediction PLUS SHAP-based top-5 risk factors and a
    Gemini-generated plain-language explanation of the decision.
    Works for both APPROVE and REJECT decisions — always useful to know why.
    """
    try:
        global shap_explainer

        df   = pd.DataFrame([applicant.dict()])
        X    = preprocess(df)
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= threshold)
        band = risk_band(prob)

        # ── Lazy-init LinearExplainer (needs preprocessed background) ──
        if shap_explainer is None and model_name == "LR":
            bg_processed = preprocess(_bg_raw)
            shap_explainer = shap.LinearExplainer(
                model, bg_processed, feature_perturbation="interventional"
            )

        # ── Compute SHAP values for this single applicant ───────────
        shap_raw = shap_explainer.shap_values(X)

        # Normalise to 1-D array (one value per feature)
        if isinstance(shap_raw, list):
            # Binary classifier returns [neg_class, pos_class]
            sv = np.array(shap_raw[1], dtype=np.float64).flatten()
        else:
            sv = np.array(shap_raw, dtype=np.float64).flatten()

        feature_names = list(train_columns)

        # ── Rank by absolute SHAP value, pick top 5 ─────────────────
        ranked_idx = np.argsort(np.abs(sv))[::-1][:5]

        top_factors = []
        for rank, idx in enumerate(ranked_idx, 1):
            col   = feature_names[idx]
            val   = float(sv[idx])
            top_factors.append(
                RiskFactor(
                    rank          = rank,
                    feature       = col,
                    feature_label = feature_label(col),
                    shap_value    = round(val, 4),
                    direction     = "increases_risk" if val > 0 else "reduces_risk",
                )
            )

        # ── Gemini plain-language explanation ───────────────────────
        applicant_ctx = {
            "loan_type":         applicant.loan_type,
            "loan_amount_inr":   applicant.loan_amount_inr,
            "annual_income_inr": applicant.annual_income_inr,
        }
        explanation = gemini_explain(
            default_probability = prob,
            risk_band_label     = band,
            top_factors         = [f.dict() for f in top_factors],
            applicant_context   = applicant_ctx,
        )

        return ExplainResponse(
            loan_id             = applicant.loan_id,
            default_probability = round(prob, 4),
            predicted_default   = pred,
            risk_band           = band,
            decision            = "REJECT" if pred == 1 else "APPROVE",
            top_factors         = top_factors,
            plain_explanation   = explanation,
            gemini_used         = bool(gemini_client),
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
            "total":         len(results),
            "approved":      approve_count,
            "rejected":      reject_count,
            "approval_rate": round(approve_count / len(results) * 100, 1),
            "predictions":   results,
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
