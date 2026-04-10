# ============================================================
# applicant_app.py
# Applicant-facing view — Features 1-5
# Run: streamlit run applicant_app.py
# Free APIs only: HuggingFace (optional), Google Translate
# (free tier), gTTS for audio, ReportLab for PDF
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime, io, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import hashlib
import reportlab.pdfbase.pdfdoc as pdfdoc
pdfdoc.md5 = lambda *args, **kwargs: hashlib.md5()


from pipeline  import THRESHOLD, score_single, preprocess
from explainer import (
    compute_shap, top_shap_factors, explain_rejection,
    find_loan_counterfactual, narrate_loan_cf,
    find_profile_counterfactual, narrate_profile_cf,
)

# ── Optional: free PDF + translation + audio ─────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

try:
    from googletrans import Translator
    _gtrans = Translator()
    GTRANS_OK = True
except ImportError:
    GTRANS_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="IndusCredit — Loan Application",
                   page_icon="🏦", layout="wide")

st.markdown("""
<style>
.header-box{background:linear-gradient(135deg,#1B3A6B,#2563EB);
  padding:1.8rem;border-radius:12px;margin-bottom:1.2rem;text-align:center}
.header-box h1{color:white;margin:0;font-size:1.9rem}
.header-box p{color:#BFDBFE;margin:.3rem 0 0}
.approved{background:#DCFCE7;border:2px solid #166534;border-radius:10px;
  padding:1.2rem;text-align:center}
.rejected{background:#FEE2E2;border:2px solid #991B1B;border-radius:10px;
  padding:1.2rem;text-align:center}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
  <h1>🏦 IndusCredit Finance — Loan Application Portal</h1>
  <p>Fill in your details and get an instant credit decision with full explanation</p>
</div>""", unsafe_allow_html=True)

LANGUAGES = {
    "English":"en","Hindi":"hi","Marathi":"mr",
    "Tamil":"ta","Telugu":"te","Kannada":"kn",
    "Gujarati":"gu","Bengali":"bn",
}

# ════════════════════════════════════════════════════════════
# APPLICATION FORM
# ════════════════════════════════════════════════════════════
st.markdown("## 📝 Loan Application Form")

with st.form("app_form"):
    st.subheader("👤 Personal Details")
    c1,c2,c3 = st.columns(3)
    app_date  = c1.date_input("Application Date", value=datetime.date.today())
    age       = c2.number_input("Age", 21, 65, 35)
    gender    = c3.selectbox("Gender", ["Male","Female"])

    c4,c5,c6 = st.columns(3)
    education = c4.selectbox("Education",
                ["Graduate","Post_Graduate","Undergraduate","Diploma","No_Formal"])
    state     = c5.text_input("State Code (e.g. MH)", "MH")
    urban_rural = c6.selectbox("Area Type", ["Urban","Semi_Urban","Rural"])

    st.subheader("💼 Employment & Income")
    c7,c8,c9 = st.columns(3)
    employment_type  = c7.selectbox("Employment Type",
                       ["Salaried","Self_Employed","Business_Owner","Government","Retired"])
    employment_years = c8.number_input("Years Employed", 0, 40, 5)
    annual_income    = c9.number_input("Annual Income (₹)", 50000, 10000000, 600000, 10000)

    c10,c11,c12 = st.columns(3)
    savings       = c10.number_input("Savings Balance (₹)", 0, 5000000, 100000, 5000)
    credit_score  = c11.number_input("Credit Score", 550, 900, 680)
    existing_loans= c12.number_input("Existing Loans", 0, 10, 1)

    c13,c14,c15 = st.columns(3)
    dti_ratio    = c13.number_input("DTI Ratio", 0.0, 0.65, 0.35, 0.01)
    missed_pay   = c14.number_input("Missed Payments (last 2 yrs)", 0, 20, 0)
    bureau_enq   = c15.number_input("Bureau Enquiries (6m)", 0, 15, 1)

    st.subheader("🏠 Loan Details")
    c16,c17 = st.columns(2)
    loan_type    = c16.selectbox("Loan Type",
                   ["Home_Loan","Personal_Loan","Auto_Loan",
                    "Education_Loan","MSME_Loan","Gold_Loan"])
    loan_purpose = c17.text_input("Loan Purpose", "Purchase")

    c18,c19,c20 = st.columns(3)
    loan_amount  = c18.number_input("Loan Amount (₹)", 50000, 10000000, 1500000, 50000)
    loan_tenure  = c19.number_input("Tenure (months)", 6, 360, 120)
    interest_rate= c20.number_input("Interest Rate (%)", 5.0, 25.0, 9.5, 0.1)

    c21,c22 = st.columns(2)
    has_collateral = c21.selectbox("Has Collateral", [0,1],
                                   format_func=lambda x:"Yes" if x else "No")
    ltv_ratio = None
    if loan_type == "Home_Loan":
        ltv_ratio = c22.number_input("LTV Ratio", 0.1, 1.0, 0.75, 0.01)

    st.subheader("🌐 Report Language")
    language = st.selectbox("Preferred language for your report",
                             list(LANGUAGES.keys()))

    submitted = st.form_submit_button("🚀 Submit Application", use_container_width=True)

# ════════════════════════════════════════════════════════════
# PROCESSING
# ════════════════════════════════════════════════════════════
if submitted:
    applicant = {
        "application_date":            str(app_date),
        "age":                         int(age),
        "gender":                      gender,
        "education":                   education,
        "state":                       state,
        "urban_rural":                 urban_rural,
        "employment_type":             employment_type,
        "employment_years":            int(employment_years),
        "annual_income_inr":           float(annual_income),
        "loan_type":                   loan_type,
        "loan_purpose":                loan_purpose,
        "loan_amount_inr":             float(loan_amount),
        "loan_tenure_months":          int(loan_tenure),
        "interest_rate_pct":           float(interest_rate),
        "credit_score":                int(credit_score),
        "num_existing_loans":          int(existing_loans),
        "dti_ratio":                   float(dti_ratio),
        "ltv_ratio":                   ltv_ratio,
        "has_collateral":              int(has_collateral),
        "bureau_enquiries_6m":         int(bureau_enq),
        "missed_payments_2y":          int(missed_pay),
        "savings_account_balance_inr": float(savings),
    }

    with st.spinner("Scoring application..."):
        prob, pred, X_proc = score_single(applicant)

    decision = "APPROVED" if pred == 0 else "REJECTED"
    st.divider()
    st.markdown("## 📊 Decision")

    m1,m2,m3 = st.columns(3)
    m1.metric("Default Probability", f"{prob*100:.1f}%")
    m2.metric("Risk Band",
              "Low" if prob<0.3 else "Medium" if prob<0.5 else "High" if prob<0.7 else "Very High")
    m3.metric("Decision", decision)

    # ── Progress bar gauge ────────────────────────────────────
    bar_color = "#166534" if prob < THRESHOLD else "#991B1B"
    st.markdown(f"""
    <div style="background:#e5e7eb;border-radius:8px;height:18px;margin:6px 0">
      <div style="width:{prob*100:.1f}%;background:{bar_color};height:18px;
           border-radius:8px;transition:width .4s"></div>
    </div>
    <p style="font-size:0.8rem;color:#6b7280">Default Probability: {prob*100:.1f}%
       &nbsp;|&nbsp; Threshold: {THRESHOLD*100:.1f}%</p>
    """, unsafe_allow_html=True)

    if decision == "APPROVED":
        rate_adj = 0 if prob < 0.3 else (0.5 if prob < 0.5 else 1.5)
        st.markdown(f"""
        <div class="approved">
          <h2 style="color:#166534">✅ Your loan is APPROVED</h2>
          <p>Suggested Interest Rate: <strong>{interest_rate + rate_adj:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

        rejection_text = loan_cf_text = profile_cf_text = ""
        loan_cf = {"found": True, "already_approved": True, "original_prob": prob}
        profile_scenarios = []

    else:
        st.markdown("""
        <div class="rejected">
          <h2 style="color:#991B1B">❌ Application Not Approved</h2>
          <p>See below — we explain why, and show you exactly what to change.</p>
        </div>""", unsafe_allow_html=True)

        # ── Feature 2: Why rejected (SHAP) ───────────────────
        st.markdown("### 🔍 Why Was Your Loan Rejected?")
        with st.spinner("Computing explanation..."):
            shap_vals, feat_names, base_val = compute_shap(X_proc)
            factors       = top_shap_factors(shap_vals, feat_names, X_proc.iloc[0], n=5)
            rejection_text = explain_rejection(prob, factors)

        st.info(rejection_text)

        with st.expander("📊 SHAP Waterfall Chart"):
            exp = shap.Explanation(
                values=shap_vals, base_values=base_val,
                data=X_proc.values[0], feature_names=feat_names,
            )
            fig, _ = plt.subplots(figsize=(10,5))
            shap.plots.waterfall(exp, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── Feature 3: Loan parameter counterfactual ─────────
        st.markdown("### 🔧 What Loan Terms Would Be Approved?")
        with st.spinner("Searching loan combinations..."):
            loan_cf      = find_loan_counterfactual(applicant)
            loan_cf_text = narrate_loan_cf(applicant, loan_cf)

        if loan_cf.get("found") and not loan_cf.get("already_approved"):
            st.success(loan_cf_text)
            st.markdown("**Required changes:**")
            for c in loan_cf.get("changes", []):
                st.markdown(f"  • {c}")
            col_a, col_b = st.columns(2)
            col_a.metric("Current probability", f"{loan_cf['original_prob']*100:.1f}%")
            col_b.metric("After changes", f"{loan_cf['approved_prob']*100:.1f}%",
                         delta=f"-{(loan_cf['original_prob']-loan_cf['approved_prob'])*100:.1f}%",
                         delta_color="inverse")
        else:
            st.warning(loan_cf.get("message", "No feasible loan combination found."))

        # ── Feature 4: Profile improvement ───────────────────
        st.markdown("### 📈 What Profile Changes Would Get This Exact Loan Approved?")
        with st.spinner("Finding improvement scenarios..."):
            profile_scenarios = find_profile_counterfactual(applicant)
            profile_cf_text   = narrate_profile_cf(profile_scenarios)

        st.info(profile_cf_text)
        if profile_scenarios:
            for s in profile_scenarios:
                ca,cb,cc,cd = st.columns([3,1.5,1.5,1])
                ca.write(s["description"])
                cb.metric("Current", str(s["from"]))
                cc.metric("Required", str(s["to"]))
                cd.markdown(f"**{s['difficulty']}**")

    # ════════════════════════════════════════════════════════
    # Feature 5 — Report (PDF + optional translation + audio)
    # ════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 📄 Download Your Report")

    # Build report text
    lines = [
        f"INDUSCREDIT FINANCE — LOAN ASSESSMENT REPORT",
        f"Date: {datetime.date.today().strftime('%d %b %Y')}",
        "",
        "APPLICATION SUMMARY",
        f"Loan Type: {loan_type}  |  Amount: ₹{loan_amount:,.0f}  |  Tenure: {loan_tenure} months",
        f"Interest Rate: {interest_rate}%  |  Collateral: {'Yes' if has_collateral else 'No'}",
        f"Credit Score: {credit_score}  |  DTI Ratio: {dti_ratio}",
        "",
        "CREDIT DECISION",
        f"Decision: {decision}",
        f"Default Probability: {prob*100:.1f}%",
        "",
    ]
    if decision == "REJECTED":
        lines += [
            "REASONS FOR REJECTION",
            rejection_text, "",
            "LOAN TERMS THAT WOULD BE APPROVED",
            loan_cf_text if loan_cf_text else "See dashboard for details.", "",
            "STEPS TO IMPROVE YOUR PROFILE",
            profile_cf_text, "",
        ]
    lines += ["NEXT STEPS",
              "Contact your nearest IndusCredit branch or call 1800-XXX-XXXX for assistance.",
              "",
              "This report is confidential and intended only for the applicant."]

    report_en = "\n".join(lines)

    # Translate if needed
    report_final = report_en
    if language != "English" and GTRANS_OK:
        with st.spinner(f"Translating to {language}..."):
            try:
                lang_code     = LANGUAGES[language]
                report_final  = _gtrans.translate(report_en, dest=lang_code).text
            except Exception as e:
                st.warning(f"Translation failed ({e}). Showing English report.")

    with st.expander(f"👁 Preview Report ({language})"):
        st.text(report_final)

    # PDF generation
    pdf_bytes = None
    if REPORTLAB_OK:
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        H1  = ParagraphStyle("h1", fontSize=14, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1B3A6B"), spaceAfter=6)
        BD  = ParagraphStyle("body", fontSize=10, fontName="Helvetica",
                              textColor=colors.HexColor("#374151"), leading=15,
                              spaceAfter=5, alignment=TA_JUSTIFY)
        story = []
        # Header
        hdr = Table([[Paragraph("IndusCredit Finance — Loan Assessment Report",
                                ParagraphStyle("t",fontSize=16,fontName="Helvetica-Bold",
                                               textColor=colors.white,alignment=TA_CENTER))]])
        hdr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#1B3A6B")),
                                  ("ROWPADDING",(0,0),(-1,-1),14)]))
        story.append(hdr); story.append(Spacer(1,.3*cm))

        # Decision badge
        dec_col = colors.HexColor("#DCFCE7") if decision=="APPROVED" else colors.HexColor("#FEE2E2")
        dec_txt_col = colors.HexColor("#166534") if decision=="APPROVED" else colors.HexColor("#991B1B")
        dec_tbl = Table([[Paragraph(f"{'✓ APPROVED' if decision=='APPROVED' else '✗ REJECTED'}  —  {prob*100:.1f}% default probability",
                                    ParagraphStyle("d",fontSize=14,fontName="Helvetica-Bold",
                                                   textColor=dec_txt_col,alignment=TA_CENTER))]])
        dec_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),dec_col),
                                      ("ROWPADDING",(0,0),(-1,-1),12),
                                      ("BOX",(0,0),(-1,-1),1,dec_txt_col)]))
        story.append(dec_tbl); story.append(Spacer(1,.4*cm))

        # Body paragraphs
        for ln in report_final.split("\n"):
            ln = ln.strip()
            if not ln:
                story.append(Spacer(1,.12*cm))
            elif ln.isupper():
                story.append(HRFlowable(width="100%",thickness=1,
                                        color=colors.HexColor("#2563EB")))
                story.append(Paragraph(ln, H1))
            else:
                story.append(Paragraph(ln, BD))

        # Loan comparison table if rejected
        if decision == "REJECTED" and loan_cf.get("found") and not loan_cf.get("already_approved"):
            story.append(Spacer(1,.2*cm))
            story.append(Paragraph("APPROVED LOAN TERMS COMPARISON", H1))
            ap = loan_cf["approved_params"]
            rows = [["Parameter","Requested","Would Be Approved"],
                    ["Amount", f"₹{loan_amount:,.0f}", f"₹{ap.get('loan_amount_inr',loan_amount):,.0f}"],
                    ["Tenure", f"{loan_tenure}m", f"{ap.get('loan_tenure_months',loan_tenure)}m"],
                    ["Collateral","Yes" if has_collateral else "No","Yes" if ap.get("has_collateral") else "No"]]
            t = Table(rows, colWidths=[5.5*cm,5.5*cm,6*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1B3A6B")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#D1D5DB")),
                ("ROWPADDING",(0,0),(-1,-1),6),
                ("BACKGROUND",(2,1),(2,-1),colors.HexColor("#DCFCE7")),
            ]))
            story.append(t)

        doc.build(story)
        pdf_bytes = buf.getvalue()

    # Audio
    audio_bytes = None
    if GTTS_OK:
        with st.spinner("Generating audio..."):
            try:
                lang_code_audio = LANGUAGES.get(language, "en")
                tts = gTTS(text=report_final[:4500], lang=lang_code_audio, slow=False)
                abuf = io.BytesIO()
                tts.write_to_fp(abuf)
                audio_bytes = abuf.getvalue()
            except Exception as e:
                st.warning(f"Audio generation failed: {e}")

    # Download buttons
    st.success(f"✅ Report ready in {language}!")
    dl1, dl2, dl3 = st.columns(3)

    dl1.download_button(
        "⬇️ Download as TXT",
        data=report_final.encode("utf-8"),
        file_name=f"loan_report_{language.lower()}.txt",
        mime="text/plain", use_container_width=True,
    )
    if pdf_bytes:
        dl2.download_button(
            "⬇️ Download as PDF",
            data=pdf_bytes,
            file_name=f"loan_report_{language.lower()}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    else:
        dl2.info("Install `reportlab` for PDF")

    if audio_bytes:
        dl3.download_button(
            "🔊 Download Audio (.mp3)",
            data=audio_bytes,
            file_name=f"loan_report_{language.lower()}.mp3",
            mime="audio/mpeg", use_container_width=True,
        )
    else:
        dl3.info("Install `gtts` for audio")
