# ============================================================
# pipeline.py  —  shared preprocessing + scoring
# Mirrors training pipeline exactly. Used by both apps.
# ============================================================
import pickle, os
import numpy as np
import pandas as pd

def _load(p):
    with open(p, "rb") as f: return pickle.load(f)

MODEL         = _load("best_model.pkl")
MODEL_NAME    = _load("best_model_name.pkl")
TARGET_ENC    = _load("target_encoder.pkl")
TRAIN_COLS    = _load("train_columns.pkl")
TRAIN_MEDIANS = _load("train_medians.pkl")
THRESHOLD     = float(_load("best_threshold.pkl"))
SCALER        = _load("scaler.pkl") if (MODEL_NAME == "LR" and os.path.exists("scaler.pkl")) else None

EDUCATION_MAP = {"No_Formal":0,"Diploma":1,"Undergraduate":2,"Graduate":3,"Post_Graduate":4,"Unknown":-1}
ONE_HOT_COLS  = ["gender","urban_rural","employment_type","loan_type","loan_purpose"]

def engineer_features(df):
    d = df.copy()
    d["loan_to_income_ratio"]     = d["loan_amount_inr"] / d["annual_income_inr"]
    d["dti_credit_risk"]          = d["dti_ratio"] / (d["credit_score"] / 700)
    d["income_per_year_employed"] = d["annual_income_inr"] / (d["employment_years"] + 1)
    d["emi_estimate"]             = d["loan_amount_inr"] / d["loan_tenure_months"]
    d["loan_per_credit_score"]    = d["loan_amount_inr"] / d["credit_score"]
    d["high_dti"]                 = (d["dti_ratio"] > 0.4).astype(int)
    d["is_new_employee"]          = (d["employment_years"] < 2).astype(int)
    d["savings_to_loan_ratio"]    = d["savings_account_balance_inr"] / (d["loan_amount_inr"] + 1)
    d["credit_enquiry_risk"]      = d["bureau_enquiries_6m"] * d["dti_ratio"]
    d["repayment_risk_score"]     = (d["missed_payments_2y"] * (1 / (d["credit_score"] / 700))).fillna(0)
    d["loan_burden"]              = d["num_existing_loans"] * d["dti_ratio"]
    d["interest_x_tenure"]        = d["interest_rate_pct"] * d["loan_tenure_months"]
    return d

def preprocess(df):
    d = df.copy()
    for c in ["loan_id","default_flag"]:
        if c in d.columns: d.drop(columns=[c], inplace=True)
    d["application_date"]  = pd.to_datetime(d["application_date"])
    d["application_month"] = d["application_date"].dt.month
    d["application_year"]  = d["application_date"].dt.year
    d.drop(columns=["application_date"], inplace=True)
    d["ltv_missing_flag"]  = d["ltv_ratio"].isnull().astype(int)
    d["ltv_ratio"]         = d["ltv_ratio"].fillna(-1)
    for c in d.select_dtypes(include=np.number).columns:
        if c in TRAIN_MEDIANS.index: d[c] = d[c].fillna(TRAIN_MEDIANS[c])
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].fillna("Unknown")
    d = engineer_features(d)
    d["education"] = d["education"].map(EDUCATION_MAP).fillna(-1).astype(int)
    d = pd.get_dummies(d, columns=ONE_HOT_COLS, drop_first=True)
    d["state"] = TARGET_ENC.transform(d[["state"]])
    for c in TRAIN_COLS:
        if c not in d.columns: d[c] = 0
    d = d[TRAIN_COLS]
    if SCALER is not None:
        nc = d.select_dtypes(include=np.number).columns
        d[nc] = SCALER.transform(d[nc])
    return d

def score_batch(df_raw):
    """Returns np.ndarray of default probabilities for a DataFrame."""
    return MODEL.predict_proba(preprocess(df_raw))[:, 1]

def score_single(applicant_dict):
    """Returns (prob, pred, X_processed) for one applicant."""
    df   = pd.DataFrame([applicant_dict])
    X    = preprocess(df)
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    return prob, int(prob >= THRESHOLD), X

def classify(prob, low_thr, high_thr):
    if prob < low_thr:   return "Low"
    elif prob < high_thr: return "Medium"
    else:                 return "High"
