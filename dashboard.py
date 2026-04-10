# ============================================================
# dashboard.py
# Streamlit Dashboard — IndusCredit Finance Credit Risk System
# Pages: Applicant Risk Lookup | Batch Scoring | Segment Analysis | Model Insights
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

    # On submit: fire /explain and persist result in session_state
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
            with st.spinner("Assessing risk and generating AI explanation..."):
                resp = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
                resp.raise_for_status()
                st.session_state["_result"] = resp.json()
                st.session_state["_error"]  = None
        except Exception as e:
            st.session_state["_result"] = None
            st.session_state["_error"]  = str(e)

    # Render — outside if submitted so it persists across re-renders
    if st.session_state.get("_error"):
        st.error(f"API Error: {st.session_state['_error']}")

    result = st.session_state.get("_result")
    if result:
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
            st.error(f"**REJECT** — High default risk detected (Probability: {prob*100:.1f}%)")
        else:
            st.success(f"**APPROVE** — Low default risk (Probability: {prob*100:.1f}%)")

        # SHAP chart
        top_factors = result.get("top_factors", [])
        if top_factors:
            st.subheader("🔍 Top Risk Factors (SHAP)")
            labels     = [f["feature_label"] for f in top_factors]
            shap_vals  = [f["shap_value"]    for f in top_factors]
            directions = [f["direction"]      for f in top_factors]
            bar_colors = ["#F44336" if d == "increases_risk" else "#4CAF50" for d in directions]

            fig2, ax2 = plt.subplots(figsize=(8, max(3, len(labels) * 0.6)))
            bars = ax2.barh(labels[::-1], shap_vals[::-1], color=bar_colors[::-1])
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("SHAP Value (impact on default probability)")
            ax2.set_title("Feature Contributions to Risk Score", fontweight="bold")
            for bar, val in zip(bars, shap_vals[::-1]):
                ax2.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8,
                )
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            with st.expander("📋 Factor Details"):
                factor_df = pd.DataFrame([{
                    "Rank":   f["rank"],
                    "Factor": f["feature_label"],
                    "SHAP":   f"{f['shap_value']:+.4f}",
                    "Effect": "⬆ Increases Risk" if f["direction"] == "increases_risk" else "⬇ Reduces Risk",
                } for f in top_factors])
                st.dataframe(factor_df, use_container_width=True, hide_index=True)

        # Gemini explanation
        explanation = result.get("plain_explanation", "")
        gemini_used = result.get("gemini_used", False)
        if explanation:
            icon  = "🤖" if gemini_used else "📝"
            label = "AI Explanation (Gemini)" if gemini_used else "Explanation (Template)"
            st.info(f"**{icon} {label}**\n\n{explanation}")


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
