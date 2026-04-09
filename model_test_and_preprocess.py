# ============================================================
# model_test_and_preprocess.py
# Loan Default Risk Assessment — IndusCredit Finance
# Applies identical preprocessing pipeline to test set,
# scores with trained model, evaluates against test_labels.csv
# ============================================================

import pandas as pd
import numpy as np
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

os.makedirs("test_results", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ============================================================
# 1. LOAD ARTIFACTS
# ============================================================
print("=" * 60)
print("LOADING ARTIFACTS")
print("=" * 60)

with open("best_model.pkl",      "rb") as f: model         = pickle.load(f)
with open("best_model_name.pkl", "rb") as f: model_name    = pickle.load(f)
with open("target_encoder.pkl",  "rb") as f: target_enc    = pickle.load(f)
with open("train_columns.pkl",   "rb") as f: train_columns = pickle.load(f)
with open("train_medians.pkl",   "rb") as f: train_medians = pickle.load(f)
with open("best_threshold.pkl",  "rb") as f: threshold     = pickle.load(f)

scaler = None
if model_name == "LR" and os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

print(f"  Model loaded     : {model_name}")
print(f"  Threshold        : {threshold:.4f}")
print(f"  Feature columns  : {len(train_columns)}")

# ============================================================
# 2. LOAD TEST DATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

df_test = pd.read_csv("loan_test.csv")
print(f"  Test set shape   : {df_test.shape}")

# Load ground truth if available
has_labels = os.path.exists("test_labels.csv")
if has_labels:
    df_labels = pd.read_csv("test_labels.csv")
    # Align on loan_id
    df_test = df_test.merge(df_labels[["loan_id", "default_flag"]], on="loan_id", how="left")
    print(f"  Labels loaded    : {df_labels.shape[0]} rows")
    print(f"  Label distribution:\n{df_test['default_flag'].value_counts()}")
else:
    print("  test_labels.csv not found — running inference only (no evaluation metrics)")

loan_ids = df_test["loan_id"].copy()

# ============================================================
# 3. PREPROCESSING — IDENTICAL TO TRAINING PIPELINE
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Drop loan_id
df_test.drop(columns=["loan_id"], inplace=True)

# Drop label column temporarily for feature engineering
if has_labels and "default_flag" in df_test.columns:
    y_test = df_test.pop("default_flag")
else:
    y_test = None

# Date features
df_test["application_date"]  = pd.to_datetime(df_test["application_date"])
df_test["application_month"] = df_test["application_date"].dt.month
df_test["application_year"]  = df_test["application_date"].dt.year
df_test.drop(columns=["application_date"], inplace=True)

# LTV null handling — same sentinel + flag as training
df_test["ltv_missing_flag"] = df_test["ltv_ratio"].isnull().astype(int)
df_test["ltv_ratio"]        = df_test["ltv_ratio"].fillna(-1)

# Fill numeric nulls with TRAINING medians (no leakage)
num_cols = df_test.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    if col in train_medians.index:
        df_test[col] = df_test[col].fillna(train_medians[col])

# Fill categorical nulls
for col in df_test.select_dtypes(include="object").columns:
    df_test[col] = df_test[col].fillna("Unknown")

# ── Feature Engineering (exact mirror of training) ──────────
df_test["loan_to_income_ratio"]     = df_test["loan_amount_inr"] / df_test["annual_income_inr"]
df_test["dti_credit_risk"]          = df_test["dti_ratio"] / (df_test["credit_score"] / 700)
df_test["income_per_year_employed"] = df_test["annual_income_inr"] / (df_test["employment_years"] + 1)
df_test["emi_estimate"]             = df_test["loan_amount_inr"] / df_test["loan_tenure_months"]
df_test["loan_per_credit_score"]    = df_test["loan_amount_inr"] / df_test["credit_score"]
df_test["high_dti"]                 = (df_test["dti_ratio"] > 0.4).astype(int)
df_test["is_new_employee"]          = (df_test["employment_years"] < 2).astype(int)
df_test["savings_to_loan_ratio"]    = df_test["savings_account_balance_inr"] / (df_test["loan_amount_inr"] + 1)
df_test["credit_enquiry_risk"]      = df_test["bureau_enquiries_6m"] * df_test["dti_ratio"]
df_test["repayment_risk_score"]     = df_test["missed_payments_2y"] * (1 / (df_test["credit_score"] / 700))
raw_val = df_test["missed_payments_2y"] * (700 / (df_test["credit_score"] + 1))
log_val = np.log1p(raw_val)
df_test["loan_burden"]              = df_test["num_existing_loans"] * df_test["dti_ratio"]
df_test["interest_x_tenure"]        = df_test["interest_rate_pct"] * df_test["loan_tenure_months"]

# ── Encoding ─────────────────────────────────────────────────
education_map = {
    "No_Formal": 0, "Diploma": 1, "Undergraduate": 2,
    "Graduate": 3, "Post_Graduate": 4, "Unknown": -1
}
df_test["education"] = df_test["education"].map(education_map).fillna(-1).astype(int)

one_hot_cols = ["gender", "urban_rural", "employment_type", "loan_type", "loan_purpose"]
df_test = pd.get_dummies(df_test, columns=one_hot_cols, drop_first=True)

# Target encoding — transform only (fitted on training data)
df_test["state"] = target_enc.transform(df_test["state"])

# ── Align columns to training feature set ────────────────────
# Add any missing columns as 0 (unseen categories from OHE)
for col in train_columns:
    if col not in df_test.columns:
        df_test[col] = 0

# Keep only training columns in training order
X_test = df_test[train_columns]

print(f"  Test feature matrix shape: {X_test.shape}")

# Scale if LR
if scaler is not None:
    num_cols_scale = X_test.select_dtypes(include=np.number).columns
    X_test = X_test.copy()
    X_test[num_cols_scale] = scaler.transform(X_test[num_cols_scale])
    print("  StandardScaler applied (LR model)")

# ============================================================
# 4. INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE")
print("=" * 60)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

print(f"  Threshold used           : {threshold:.4f}")
print(f"  Predicted defaults       : {y_pred.sum()} / {len(y_pred)} ({y_pred.mean()*100:.1f}%)")

# ============================================================
# 5. EVALUATION (only if labels available)
# ============================================================
if has_labels and y_test is not None:
    print("\n" + "=" * 60)
    print("EVALUATION AGAINST TEST LABELS")
    print("=" * 60)

    auc_roc  = roc_auc_score(y_test, y_prob)
    auc_pr   = average_precision_score(y_test, y_prob)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n  AUC-ROC          : {auc_roc:.4f}")
    print(f"  AUC-PR           : {auc_pr:.4f}")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                     Pred 0    Pred 1")
    print(f"    Actual 0        {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"    Actual 1        {cm[1,0]:>6}    {cm[1,1]:>6}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))

    # ── ROC Curve ────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes   = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color="#E91E63", linewidth=2,
                 label=f"AUC-ROC = {auc_roc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {model_name} (Test Set)", fontweight="bold")
    axes[0].legend()

    # ── PR Curve ─────────────────────────────────────────────
    precision_arr, recall_arr, thr_arr = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall_arr, precision_arr, color="#009688", linewidth=2,
                 label=f"AUC-PR = {auc_pr:.4f}")
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--",
                    linewidth=1, label=f"Baseline = {y_test.mean():.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name} (Test Set)", fontweight="bold")
    axes[1].legend()

    plt.suptitle("Test Set Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("test_results/roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: test_results/roc_pr_curves.png")

    # ── Score distribution plot ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6,
            color="#2196F3", label="Non-Default", density=True)
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6,
            color="#F44336", label="Default", density=True)
    ax.axvline(threshold, color="navy", linestyle="--",
               linewidth=2, label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {model_name} (Test Set)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("test_results/score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: test_results/score_distribution.png")

# ============================================================
# 6. SAVE PREDICTIONS CSV
# ============================================================
output = pd.DataFrame({
    "loan_id":              loan_ids.values,
    "default_probability":  np.round(y_prob, 4),
    "predicted_default":    y_pred,
    "risk_band":            pd.cut(
        y_prob,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["Low", "Medium", "High", "Very High"]
    ),
})

if has_labels and y_test is not None:
    output["actual_default"] = y_test.values

output.to_csv("test_results/predictions.csv", index=False)
print(f"\n  Saved: test_results/predictions.csv  ({len(output)} rows)")
print(f"\n  Risk band distribution:")
print(output["risk_band"].value_counts().sort_index())

print("\n" + "=" * 60)
print("TEST SCORING COMPLETE")
print("  Results → test_results/")
print("=" * 60)
