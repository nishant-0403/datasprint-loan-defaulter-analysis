# ============================================================
# dashboard.py
# Streamlit Dashboard — IndusCredit Finance Credit Risk System
# Pages: Applicant Risk Lookup | Batch Scoring | Segment Analysis | Model Insights
# Run: streamlit run dashboard.py
# ============================================================
'''
import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IndusCredit — Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏦 IndusCredit Finance")
st.sidebar.caption("Credit Risk Assessment System")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Applicant Risk Lookup", "📂 Batch Scoring", "📊 Segment Analysis", "🧠 Model Insights"],
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API Online — Model: **{health['model']}**")
    st.sidebar.caption(f"Threshold: {health['threshold']}")
except Exception:
    st.sidebar.error("⚠️ API Offline. Start with: `uvicorn api:app --reload`")


# ============================================================
# PAGE 1 — APPLICANT RISK LOOKUP
# ============================================================
if page == "🔍 Applicant Risk Lookup":
    st.title("🔍 Applicant Risk Lookup")
    st.caption("Fill in applicant details to get an instant default risk score.")

    with st.form("applicant_form"):
        st.subheader("Personal Details")
        c1, c2, c3 = st.columns(3)
        loan_id    = c1.text_input("Loan ID", value="TEST_001")
        app_date   = c2.date_input("Application Date")
        age        = c3.number_input("Age", min_value=21, max_value=65, value=35)

        c4, c5, c6 = st.columns(3)
        gender     = c4.selectbox("Gender", ["Male", "Female"])
        education  = c5.selectbox("Education", ["Graduate", "Post_Graduate", "Undergraduate", "Diploma", "No_Formal"])
        state      = c6.text_input("State Code", value="MH")

        c7, c8 = st.columns(2)
        urban_rural     = c7.selectbox("Area Type", ["Urban", "Semi_Urban", "Rural"])
        employment_type = c8.selectbox("Employment Type", ["Salaried", "Self_Employed", "Business_Owner", "Government", "Retired"])

        st.subheader("Employment & Income")
        c9, c10, c11 = st.columns(3)
        employment_years       = c9.number_input("Years Employed", min_value=0, max_value=40, value=5)
        annual_income_inr      = c10.number_input("Annual Income (₹)", min_value=100000, max_value=5000000, value=800000, step=50000)
        savings_balance        = c11.number_input("Savings Balance (₹)", min_value=0, max_value=2000000, value=100000, step=10000)

        st.subheader("Loan Details")
        c12, c13 = st.columns(2)
        loan_type    = c12.selectbox("Loan Type", ["Home_Loan", "Personal_Loan", "Auto_Loan", "Education_Loan", "MSME_Loan", "Gold_Loan"])
        loan_purpose = c13.text_input("Loan Purpose", value="Purchase")

        c14, c15, c16 = st.columns(3)
        loan_amount_inr    = c14.number_input("Loan Amount (₹)", min_value=50000, max_value=5000000, value=1500000, step=50000)
        loan_tenure_months = c15.number_input("Tenure (months)", min_value=6, max_value=360, value=120)
        interest_rate_pct  = c16.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=9.5, step=0.1)

        ltv_ratio = None
        if loan_type == "Home_Loan":
            ltv_ratio = st.number_input("LTV Ratio (Home Loan only)", min_value=0.1, max_value=1.0, value=0.75, step=0.01)

        st.subheader("Credit Profile")
        c17, c18, c19 = st.columns(3)
        credit_score          = c17.number_input("Credit Score", min_value=550, max_value=900, value=720)
        num_existing_loans    = c18.number_input("Existing Loans", min_value=0, max_value=10, value=1)
        dti_ratio             = c19.number_input("DTI Ratio", min_value=0.0, max_value=0.65, value=0.35, step=0.01)

        c20, c21, c22 = st.columns(3)
        has_collateral        = c20.selectbox("Has Collateral", [0, 1], format_func=lambda x: "Yes" if x else "No")
        bureau_enquiries_6m   = c21.number_input("Bureau Enquiries (6m)", min_value=0, max_value=10, value=1)
        missed_payments_2y    = c22.number_input("Missed Payments (2yr)", min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("🚀 Assess Risk", use_container_width=True)

    if submitted:
        payload = {
            "loan_id":                     loan_id,
            "application_date":            str(app_date),
            "age":                         int(age),
            "gender":                      gender,
            "education":                   education,
            "state":                       state,
            "urban_rural":                 urban_rural,
            "employment_type":             employment_type,
            "employment_years":            int(employment_years),
            "annual_income_inr":           float(annual_income_inr),
            "loan_type":                   loan_type,
            "loan_purpose":                loan_purpose,
            "loan_amount_inr":             float(loan_amount_inr),
            "loan_tenure_months":          int(loan_tenure_months),
            "interest_rate_pct":           float(interest_rate_pct),
            "credit_score":                int(credit_score),
            "num_existing_loans":          int(num_existing_loans),
            "dti_ratio":                   float(dti_ratio),
            "ltv_ratio":                   ltv_ratio,
            "has_collateral":              int(has_collateral),
            "bureau_enquiries_6m":         int(bureau_enquiries_6m),
            "missed_payments_2y":          int(missed_payments_2y),
            "savings_account_balance_inr": float(savings_balance),
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            result = resp.json()

            prob    = result["default_probability"]
            band    = result["risk_band"]
            decision = result["decision"]

            color_map = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Default Probability", f"{prob*100:.1f}%")
            col_b.metric("Risk Band", band)
            col_c.metric("Decision", decision)

            # Probability gauge bar
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(["Risk"], [prob], color=color_map.get(band, "gray"), height=0.5)
            ax.barh(["Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.5)
            ax.axvline(0.3,  color="orange", linestyle="--", linewidth=1)
            ax.axvline(0.5,  color="red",    linestyle="--", linewidth=1)
            ax.axvline(0.7,  color="darkred",linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Default Probability")
            ax.set_title(f"Risk Score: {prob*100:.1f}%  |  Decision: {decision}", fontweight="bold")
            ax.set_yticks([])
            st.pyplot(fig)
            plt.close()

            if decision == "REJECT":
                st.error(f"❌ **REJECT** — High default risk detected (Probability: {prob*100:.1f}%)")
            else:
                st.success(f"✅ **APPROVE** — Low default risk (Probability: {prob*100:.1f}%)")

        except Exception as e:
            st.error(f"API Error: {e}")


# ============================================================
# PAGE 2 — BATCH SCORING
# ============================================================
elif page == "📂 Batch Scoring":
    st.title("📂 Batch Scoring")
    st.caption("Upload a CSV file (same format as loan_test.csv) to score multiple applicants at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.write(f"**Preview** — {len(df_preview)} rows, {len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(5))

        if st.button("🚀 Score All Applicants", use_container_width=True):
            with st.spinner("Scoring..."):
                uploaded.seek(0)
                resp = requests.post(
                    f"{API_URL}/batch_predict",
                    files={"file": ("batch.csv", uploaded.getvalue(), "text/csv")},
                    timeout=60,
                )
                data = resp.json()

            st.subheader("Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",         data["total"])
            c2.metric("Approved",      data["approved"])
            c3.metric("Rejected",      data["rejected"])
            c4.metric("Approval Rate", f"{data['approval_rate']}%")

            df_results = pd.DataFrame(data["predictions"])
            st.subheader("Predictions")
            st.dataframe(df_results, use_container_width=True)

            # Risk band distribution chart
            band_counts = df_results["risk_band"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336", "Very High": "#880E4F"}
            bars = ax.bar(band_counts.index, band_counts.values,
                          color=[colors.get(b, "gray") for b in band_counts.index])
            for bar, val in zip(bars, band_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
                        ha="center", fontweight="bold")
            ax.set_title("Risk Band Distribution", fontweight="bold")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

            # Download button
            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# PAGE 3 — SEGMENT ANALYSIS
# ============================================================
elif page == "📊 Segment Analysis":
    st.title("📊 Segment Analysis")
    st.caption("Default rates by borrower and loan segments — from training data.")

    try:
        seg_data = requests.get(f"{API_URL}/segment_analysis", timeout=10).json()

        segment_labels = {
            "loan_type":       "Loan Type",
            "employment_type": "Employment Type",
            "urban_rural":     "Urban / Rural",
            "gender":          "Gender",
            "education":       "Education Level",
        }

        cols = st.columns(2)
        plot_idx = 0
        for key, label in segment_labels.items():
            if key not in seg_data:
                continue
            rates = pd.Series(seg_data[key]).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, max(3, len(rates) * 0.6)))
            palette = sns.color_palette("RdYlGn_r", len(rates))
            bars = ax.barh(rates.index, rates.values * 100, color=palette)
            for bar, val in zip(bars, rates.values):
                ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val*100:.1f}%", va="center", fontsize=9)
            ax.axvline(27.85, color="navy", linestyle="--", linewidth=1.2, label="Overall avg")
            ax.set_xlabel("Default Rate (%)")
            ax.set_title(f"Default Rate by {label}", fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            cols[plot_idx % 2].pyplot(fig)
            plt.close()
            plot_idx += 1

    except Exception as e:
        st.error(f"Could not load segment data: {e}")


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")

    try:
        info = requests.get(f"{API_URL}/model_info", timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",      info["model_name"])
        c2.metric("AUC-ROC",   "0.9011")
        c3.metric("Threshold", info["threshold"])

        st.subheader("Top 10 Features by SHAP Importance")
        shap_vals = {
            "repayment_risk_score":    0.9248,
            "missed_payments_2y":      0.3166,
            "credit_score":            0.1027,
            "dti_ratio":               0.0924,
            "bureau_enquiries_6m":     0.0602,
            "dti_credit_risk":         0.0556,
            "num_existing_loans":      0.0505,
            "has_collateral":          0.0469,
            "loan_burden":             0.0363,
            "credit_enquiry_risk":     0.0222,
        }
        shap_df = pd.Series(shap_vals).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ["#F44336" if "risk" in f or "missed" in f or "dti" in f
                   else "#2196F3" for f in shap_df.index]
        ax.barh(shap_df.index, shap_df.values, color=colors)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("SHAP Plots from Training")
        shap_dir = "shap_plots"
        if os.path.exists(shap_dir):
            plot_files = {
                "Global Summary (Beeswarm)":  "4_1_shap_summary_beeswarm.png",
                "Top-10 Features (Bar)":      "4_1_shap_bar_top10.png",
                "Waterfall — Defaulter #1":   "4_2_waterfall_defaulted_1.png",
                "Waterfall — Defaulter #2":   "4_2_waterfall_defaulted_2.png",
                "Waterfall — Non-Default #1": "4_2_waterfall_non_defaulted_1.png",
                "Waterfall — Non-Default #2": "4_2_waterfall_non_defaulted_2.png",
            }
            selected = st.selectbox("Select SHAP Plot", list(plot_files.keys()))
            img_path = os.path.join(shap_dir, plot_files[selected])
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.warning(f"Plot not found: {img_path}")
        else:
            st.info("Run model_train_and_preprocess.py first to generate SHAP plots.")

    except Exception as e:
        st.error(f"Could not load model info: {e}")

        

'''
# ============================================================
# dashboard.py
# Streamlit Dashboard — IndusCredit Finance Credit Risk System
# Pages: Applicant Risk Lookup | Batch Scoring | Segment Analysis
#        | Model Insights | Test Set Evaluation  ← NEW
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import io
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="IndusCredit — Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏦 IndusCredit Finance")
st.sidebar.caption("Credit Risk Assessment System")
page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 Applicant Risk Lookup",
        "📂 Batch Scoring",
        "📊 Segment Analysis",
        "🧠 Model Insights",
        "🧪 Test Set Evaluation",   # ← NEW
    ],
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API Online — Model: **{health['model']}**")
    st.sidebar.caption(f"Threshold: {health['threshold']}")
except Exception:
    st.sidebar.error("⚠️ API Offline. Start with: `uvicorn api:app --reload`")


# ============================================================
# PAGE 1 — APPLICANT RISK LOOKUP
# ============================================================
if page == "🔍 Applicant Risk Lookup":
    st.title("🔍 Applicant Risk Lookup")
    st.caption("Fill in applicant details to get an instant default risk score.")

    with st.form("applicant_form"):
        st.subheader("Personal Details")
        c1, c2, c3 = st.columns(3)
        loan_id    = c1.text_input("Loan ID", value="TEST_001")
        app_date   = c2.date_input("Application Date")
        age        = c3.number_input("Age", min_value=21, max_value=65, value=35)

        c4, c5, c6 = st.columns(3)
        gender     = c4.selectbox("Gender", ["Male", "Female"])
        education  = c5.selectbox("Education", ["Graduate", "Post_Graduate", "Undergraduate", "Diploma", "No_Formal"])
        state      = c6.text_input("State Code", value="MH")

        c7, c8 = st.columns(2)
        urban_rural     = c7.selectbox("Area Type", ["Urban", "Semi_Urban", "Rural"])
        employment_type = c8.selectbox("Employment Type", ["Salaried", "Self_Employed", "Business_Owner", "Government", "Retired"])

        st.subheader("Employment & Income")
        c9, c10, c11 = st.columns(3)
        employment_years       = c9.number_input("Years Employed", min_value=0, max_value=40, value=5)
        annual_income_inr      = c10.number_input("Annual Income (₹)", min_value=100000, max_value=5000000, value=800000, step=50000)
        savings_balance        = c11.number_input("Savings Balance (₹)", min_value=0, max_value=2000000, value=100000, step=10000)

        st.subheader("Loan Details")
        c12, c13 = st.columns(2)
        loan_type    = c12.selectbox("Loan Type", ["Home_Loan", "Personal_Loan", "Auto_Loan", "Education_Loan", "MSME_Loan", "Gold_Loan"])
        loan_purpose = c13.text_input("Loan Purpose", value="Purchase")

        c14, c15, c16 = st.columns(3)
        loan_amount_inr    = c14.number_input("Loan Amount (₹)", min_value=50000, max_value=5000000, value=1500000, step=50000)
        loan_tenure_months = c15.number_input("Tenure (months)", min_value=6, max_value=360, value=120)
        interest_rate_pct  = c16.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=9.5, step=0.1)

        ltv_ratio = None
        if loan_type == "Home_Loan":
            ltv_ratio = st.number_input("LTV Ratio (Home Loan only)", min_value=0.1, max_value=1.0, value=0.75, step=0.01)

        st.subheader("Credit Profile")
        c17, c18, c19 = st.columns(3)
        credit_score          = c17.number_input("Credit Score", min_value=550, max_value=900, value=720)
        num_existing_loans    = c18.number_input("Existing Loans", min_value=0, max_value=10, value=1)
        dti_ratio             = c19.number_input("DTI Ratio", min_value=0.0, max_value=0.65, value=0.35, step=0.01)

        c20, c21, c22 = st.columns(3)
        has_collateral        = c20.selectbox("Has Collateral", [0, 1], format_func=lambda x: "Yes" if x else "No")
        bureau_enquiries_6m   = c21.number_input("Bureau Enquiries (6m)", min_value=0, max_value=10, value=1)
        missed_payments_2y    = c22.number_input("Missed Payments (2yr)", min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("🚀 Assess Risk", use_container_width=True)

    if submitted:
        payload = {
            "loan_id":                     loan_id,
            "application_date":            str(app_date),
            "age":                         int(age),
            "gender":                      gender,
            "education":                   education,
            "state":                       state,
            "urban_rural":                 urban_rural,
            "employment_type":             employment_type,
            "employment_years":            int(employment_years),
            "annual_income_inr":           float(annual_income_inr),
            "loan_type":                   loan_type,
            "loan_purpose":                loan_purpose,
            "loan_amount_inr":             float(loan_amount_inr),
            "loan_tenure_months":          int(loan_tenure_months),
            "interest_rate_pct":           float(interest_rate_pct),
            "credit_score":                int(credit_score),
            "num_existing_loans":          int(num_existing_loans),
            "dti_ratio":                   float(dti_ratio),
            "ltv_ratio":                   ltv_ratio,
            "has_collateral":              int(has_collateral),
            "bureau_enquiries_6m":         int(bureau_enquiries_6m),
            "missed_payments_2y":          int(missed_payments_2y),
            "savings_account_balance_inr": float(savings_balance),
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            result = resp.json()

            prob     = result["default_probability"]
            band     = result["risk_band"]
            decision = result["decision"]

            color_map = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Default Probability", f"{prob*100:.1f}%")
            col_b.metric("Risk Band", band)
            col_c.metric("Decision", decision)

            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh(["Risk"], [prob], color=color_map.get(band, "gray"), height=0.5)
            ax.barh(["Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.5)
            ax.axvline(0.3,  color="orange",  linestyle="--", linewidth=1)
            ax.axvline(0.5,  color="red",     linestyle="--", linewidth=1)
            ax.axvline(0.7,  color="darkred", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Default Probability")
            ax.set_title(f"Risk Score: {prob*100:.1f}%  |  Decision: {decision}", fontweight="bold")
            ax.set_yticks([])
            st.pyplot(fig)
            plt.close()

            if decision == "REJECT":
                st.error(f"❌ **REJECT** — High default risk detected (Probability: {prob*100:.1f}%)")
            else:
                st.success(f"✅ **APPROVE** — Low default risk (Probability: {prob*100:.1f}%)")

        except Exception as e:
            st.error(f"API Error: {e}")


# ============================================================
# PAGE 2 — BATCH SCORING
# ============================================================
elif page == "📂 Batch Scoring":
    st.title("📂 Batch Scoring")
    st.caption("Upload a CSV file (same format as loan_test.csv) to score multiple applicants at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.write(f"**Preview** — {len(df_preview)} rows, {len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(5))

        if st.button("🚀 Score All Applicants", use_container_width=True):
            with st.spinner("Scoring..."):
                uploaded.seek(0)
                resp = requests.post(
                    f"{API_URL}/batch_predict",
                    files={"file": ("batch.csv", uploaded.getvalue(), "text/csv")},
                    timeout=60,
                )
                data = resp.json()

            st.subheader("Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",         data["total"])
            c2.metric("Approved",      data["approved"])
            c3.metric("Rejected",      data["rejected"])
            c4.metric("Approval Rate", f"{data['approval_rate']}%")

            df_results = pd.DataFrame(data["predictions"])
            st.subheader("Predictions")
            st.dataframe(df_results, use_container_width=True)

            band_counts = df_results["risk_band"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336", "Very High": "#880E4F"}
            bars = ax.bar(band_counts.index, band_counts.values,
                          color=[colors.get(b, "gray") for b in band_counts.index])
            for bar, val in zip(bars, band_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
                        ha="center", fontweight="bold")
            ax.set_title("Risk Band Distribution", fontweight="bold")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# PAGE 3 — SEGMENT ANALYSIS
# ============================================================
elif page == "📊 Segment Analysis":
    st.title("📊 Segment Analysis")
    st.caption("Default rates by borrower and loan segments — from training data.")

    try:
        seg_data = requests.get(f"{API_URL}/segment_analysis", timeout=10).json()

        segment_labels = {
            "loan_type":       "Loan Type",
            "employment_type": "Employment Type",
            "urban_rural":     "Urban / Rural",
            "gender":          "Gender",
            "education":       "Education Level",
        }

        cols = st.columns(2)
        plot_idx = 0
        for key, label in segment_labels.items():
            if key not in seg_data:
                continue
            rates = pd.Series(seg_data[key]).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, max(3, len(rates) * 0.6)))
            palette = sns.color_palette("RdYlGn_r", len(rates))
            bars = ax.barh(rates.index, rates.values * 100, color=palette)
            for bar, val in zip(bars, rates.values):
                ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val*100:.1f}%", va="center", fontsize=9)
            ax.axvline(27.85, color="navy", linestyle="--", linewidth=1.2, label="Overall avg")
            ax.set_xlabel("Default Rate (%)")
            ax.set_title(f"Default Rate by {label}", fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            cols[plot_idx % 2].pyplot(fig)
            plt.close()
            plot_idx += 1

    except Exception as e:
        st.error(f"Could not load segment data: {e}")


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")

    try:
        info = requests.get(f"{API_URL}/model_info", timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",     info["model_name"])
        c2.metric("AUC-ROC",  "0.9011")
        c3.metric("Threshold", info["threshold"])

        st.subheader("Top 10 Features by SHAP Importance")
        shap_vals = {
            "repayment_risk_score":  0.9248,
            "missed_payments_2y":    0.3166,
            "credit_score":          0.1027,
            "dti_ratio":             0.0924,
            "bureau_enquiries_6m":   0.0602,
            "dti_credit_risk":       0.0556,
            "num_existing_loans":    0.0505,
            "has_collateral":        0.0469,
            "loan_burden":           0.0363,
            "credit_enquiry_risk":   0.0222,
        }
        shap_df = pd.Series(shap_vals).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ["#F44336" if "risk" in f or "missed" in f or "dti" in f
                   else "#2196F3" for f in shap_df.index]
        ax.barh(shap_df.index, shap_df.values, color=colors)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (SHAP)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("SHAP Plots from Training")
        shap_dir = "shap_plots"
        if os.path.exists(shap_dir):
            plot_files = {
                "Global Summary (Beeswarm)":  "4_1_shap_summary_beeswarm.png",
                "Top-10 Features (Bar)":      "4_1_shap_bar_top10.png",
                "Waterfall — Defaulter #1":   "4_2_waterfall_defaulted_1.png",
                "Waterfall — Defaulter #2":   "4_2_waterfall_defaulted_2.png",
                "Waterfall — Non-Default #1": "4_2_waterfall_non_defaulted_1.png",
                "Waterfall — Non-Default #2": "4_2_waterfall_non_defaulted_2.png",
            }
            selected = st.selectbox("Select SHAP Plot", list(plot_files.keys()))
            img_path = os.path.join(shap_dir, plot_files[selected])
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.warning(f"Plot not found: {img_path}")
        else:
            st.info("Run model_train_and_preprocess.py first to generate SHAP plots.")

    except Exception as e:
        st.error(f"Could not load model info: {e}")


# ============================================================
# PAGE 5 — TEST SET EVALUATION  ← NEW PAGE
# ============================================================
elif page == "🧪 Test Set Evaluation":
    st.title("🧪 Test Set Evaluation")
    st.caption(
        "Upload your unlabelled test CSV and the ground-truth labels CSV to score the model "
        "and view full evaluation metrics and charts."
    )

    # ── Step 1: file uploads ─────────────────────────────────
    st.subheader("Step 1 — Upload Files")
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown("**Test CSV** (`loan_test.csv` — no labels)")
        test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test_csv")

    with col_u2:
        st.markdown("**Labels CSV** (`test_labels.csv` — `loan_id` + `default_flag`)")
        labels_file = st.file_uploader("Upload labels CSV", type=["csv"], key="labels_csv")

    # ── Step 2: run evaluation ───────────────────────────────
    if test_file and labels_file:
        st.subheader("Step 2 — Run Evaluation")

        if st.button("🚀 Score & Evaluate", use_container_width=True):

            # ── 2a. Score via API ────────────────────────────
            with st.spinner("Sending test set to API for scoring..."):
                test_file.seek(0)
                try:
                    resp = requests.post(
                        f"{API_URL}/batch_predict",
                        files={"file": ("loan_test.csv", test_file.getvalue(), "text/csv")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    api_data = resp.json()
                except Exception as e:
                    st.error(f"API error during scoring: {e}")
                    st.stop()

            df_preds = pd.DataFrame(api_data["predictions"])
            df_preds["loan_id"] = df_preds["loan_id"].astype(str)

            # ── 2b. Load labels ──────────────────────────────
            labels_file.seek(0)
            df_labels = pd.read_csv(labels_file)

            # Normalise column names (handle is_fraud / default_flag aliases)
            df_labels.columns = df_labels.columns.str.strip().str.lower()
            if "is_fraud" in df_labels.columns and "default_flag" not in df_labels.columns:
                df_labels.rename(columns={"is_fraud": "default_flag"}, inplace=True)

            label_id_col = [c for c in df_labels.columns if "id" in c]
            if not label_id_col:
                st.error("Labels CSV must contain a loan_id (or similar ID) column.")
                st.stop()
            label_id_col = label_id_col[0]
            df_labels[label_id_col] = df_labels[label_id_col].astype(str)
            df_labels.rename(columns={label_id_col: "loan_id"}, inplace=True)

            # ── 2c. Merge ────────────────────────────────────
            df_merged = df_preds.merge(df_labels[["loan_id", "default_flag"]], on="loan_id", how="inner")

            if df_merged.empty:
                st.error("No matching loan_ids between test CSV and labels CSV. Check your files.")
                st.stop()

            y_true = df_merged["default_flag"].astype(int).values
            y_prob = df_merged["default_probability"].values
            y_pred = df_merged["predicted_default"].values

            st.success(f"Matched **{len(df_merged):,}** records for evaluation.")

            # ── METRICS ─────────────────────────────────────
            st.subheader("📈 Performance Metrics")

            auc_roc = roc_auc_score(y_true, y_prob)
            auc_pr  = average_precision_score(y_true, y_prob)
            f1      = f1_score(y_true, y_pred)
            cm      = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp = cm.ravel()
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # KS statistic
            data_ks = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
            cum_good = (1 - data_ks["y"]).cumsum() / max((1 - data_ks["y"]).sum(), 1e-9)
            cum_bad  = data_ks["y"].cumsum() / max(data_ks["y"].sum(), 1e-9)
            ks_stat  = float(np.max(np.abs(cum_bad.values - cum_good.values)))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AUC-ROC",     f"{auc_roc:.4f}")
            m2.metric("AUC-PR",      f"{auc_pr:.4f}")
            m3.metric("F1-Score",    f"{f1:.4f}")
            m4.metric("KS Statistic", f"{ks_stat:.4f}")
            m5.metric("Recall",      f"{recall_val:.4f}")

            m6, m7, m8, m9 = st.columns(4)
            m6.metric("Precision",   f"{precision_val:.4f}")
            m7.metric("Specificity", f"{specificity:.4f}")
            m8.metric("TP",  int(tp))
            m9.metric("FP",  int(fp))

            m10, m11 = st.columns(2)
            m10.metric("FN", int(fn))
            m11.metric("TN", int(tn))

            # ── CLASSIFICATION REPORT ────────────────────────
            with st.expander("📋 Full Classification Report"):
                report_str = classification_report(
                    y_true, y_pred, target_names=["Non-Default", "Default"]
                )
                st.code(report_str)

            # ── CHARTS ──────────────────────────────────────
            st.subheader("📊 Evaluation Charts")

            band_colors = {
                "Low": "#4CAF50", "Medium": "#FF9800",
                "High": "#F44336", "Very High": "#880E4F"
            }

            # Row 1: ROC + PR curves
            col1, col2 = st.columns(2)

            with col1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color="#E91E63", linewidth=2,
                        label=f"AUC-ROC = {auc_roc:.4f}")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve", fontweight="bold")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
                baseline = y_true.mean()
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(rec_arr, prec_arr, color="#009688", linewidth=2,
                        label=f"AUC-PR = {auc_pr:.4f}")
                ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
                           label=f"Baseline = {baseline:.3f}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve", fontweight="bold")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Row 2: Confusion matrix + Score distribution
            col3, col4 = st.columns(2)

            with col3:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred: Non-Default", "Pred: Default"],
                    yticklabels=["Actual: Non-Default", "Actual: Default"],
                    cbar=False, linewidths=0.5,
                )
                ax.set_title("Confusion Matrix", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col4:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6,
                        color="#2196F3", label="Non-Default", density=True)
                ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6,
                        color="#F44336", label="Default", density=True)

                # fetch threshold from API for the vertical line
                try:
                    thr_line = requests.get(f"{API_URL}/health", timeout=3).json()["threshold"]
                except Exception:
                    thr_line = 0.5
                ax.axvline(thr_line, color="navy", linestyle="--", linewidth=2,
                           label=f"Threshold = {thr_line:.3f}")
                ax.set_xlabel("Predicted Default Probability")
                ax.set_ylabel("Density")
                ax.set_title("Score Distribution", fontweight="bold")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Row 3: KS curve
            st.markdown("**KS Curve**")
            fig, ax = plt.subplots(figsize=(10, 4))
            x_axis = np.linspace(0, 1, len(cum_good))
            ax.plot(x_axis, cum_good.values, color="#2196F3", linewidth=2, label="Cumulative Non-Default")
            ax.plot(x_axis, cum_bad.values,  color="#F44336", linewidth=2, label="Cumulative Default")
            ks_idx = np.argmax(np.abs(cum_bad.values - cum_good.values))
            ax.axvline(x_axis[ks_idx], color="navy", linestyle="--", linewidth=1.5,
                       label=f"KS = {ks_stat:.4f}")
            ax.set_xlabel("Fraction of Population (sorted by score)")
            ax.set_ylabel("Cumulative Rate")
            ax.set_title("KS Curve", fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Row 4: Risk band breakdown vs actuals
            st.markdown("**Risk Band — Actual Default Rate**")
            df_band = df_merged.copy()
            band_order = ["Low", "Medium", "High", "Very High"]
            band_stats = (
                df_band.groupby("risk_band")["default_flag"]
                .agg(count="count", defaults="sum")
                .reindex(band_order)
                .dropna()
            )
            band_stats["default_rate"] = band_stats["defaults"] / band_stats["count"] * 100

            col5, col6 = st.columns(2)
            with col5:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(band_stats.index, band_stats["count"],
                       color=[band_colors.get(b, "gray") for b in band_stats.index])
                ax.set_title("Count per Risk Band", fontweight="bold")
                ax.set_ylabel("Count")
                for i, (idx, row) in enumerate(band_stats.iterrows()):
                    ax.text(i, row["count"] + 5, str(int(row["count"])),
                            ha="center", fontweight="bold", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col6:
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(band_stats.index, band_stats["default_rate"],
                              color=[band_colors.get(b, "gray") for b in band_stats.index])
                ax.set_title("Actual Default Rate per Risk Band", fontweight="bold")
                ax.set_ylabel("Default Rate (%)")
                for bar, val in zip(bars, band_stats["default_rate"]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.5, f"{val:.1f}%", ha="center",
                            fontweight="bold", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── FULL RESULTS TABLE ───────────────────────────
            st.subheader("📄 Full Predictions with Labels")
            display_cols = ["loan_id", "default_probability", "predicted_default",
                            "risk_band", "decision", "default_flag"]
            df_display = df_merged[display_cols].copy()
            df_display["correct"] = (df_display["predicted_default"] == df_display["default_flag"])
            st.dataframe(df_display, use_container_width=True)

            # ── DOWNLOAD ─────────────────────────────────────
            csv_out = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Evaluated Predictions CSV",
                data=csv_out,
                file_name="test_evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif test_file and not labels_file:
        st.info("📎 Labels CSV not uploaded yet. Upload `test_labels.csv` to enable evaluation.")
    elif labels_file and not test_file:
        st.info("📎 Test CSV not uploaded yet. Upload `loan_test.csv` to enable evaluation.")
    else:
        st.info("Upload both files above to begin evaluation.")
