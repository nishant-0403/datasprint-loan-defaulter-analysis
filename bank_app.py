# ============================================================
# bank_app.py
# Bank-facing dashboard — Policy Engine + Portfolio Analytics
#
# Features:
#   1. Upload scored applicant batch (CSV/Excel)
#   2. Adjustable threshold sliders → instant reclassification
#   3. Portfolio allocation constraints by loan type
#   4. Full visual analytics with expected revenue
#
# Run: streamlit run bank_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pipeline import score_batch, THRESHOLD

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="IndusCredit — Bank Dashboard",
                   page_icon="🏦", layout="wide")

st.markdown("""
<style>
.bank-header{background:linear-gradient(135deg,#0f172a,#1e3a5f);
  padding:1.6rem;border-radius:12px;margin-bottom:1rem;text-align:center}
.bank-header h1{color:white;margin:0;font-size:1.8rem}
.bank-header p{color:#93c5fd;margin:.3rem 0 0}
.metric-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:1rem;text-align:center}
.metric-card .val{font-size:2rem;font-weight:700;color:#1B3A6B}
.metric-card .lbl{font-size:.85rem;color:#64748b}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bank-header">
  <h1>🏦 IndusCredit Finance — Credit Officer Dashboard</h1>
  <p>Policy Engine · Portfolio Analytics · Risk Simulation</p>
</div>""", unsafe_allow_html=True)

LOAN_TYPES = ["Home_Loan","Personal_Loan","Auto_Loan",
              "Education_Loan","MSME_Loan","Gold_Loan"]

# ── Colour palette ────────────────────────────────────────────
RISK_COLORS = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}
LOAN_COLORS = {
    "Home_Loan":"#3b82f6","Personal_Loan":"#8b5cf6",
    "Auto_Loan":"#f59e0b","Education_Loan":"#10b981",
    "MSME_Loan":"#f97316","Gold_Loan":"#eab308",
}


# ════════════════════════════════════════════════════════════
# SIDEBAR — UPLOAD + POLICY CONTROLS
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📂 Data Upload")
    uploaded = st.file_uploader("Upload applicant CSV / Excel",
                                type=["csv","xlsx","xls"])
    st.markdown("---")
    st.markdown("## ⚙️ Policy Engine")
    st.caption("Drag sliders to adjust risk thresholds. Changes apply instantly.")

    low_thr  = st.slider("Low → Medium threshold (PD)", 0.10, 0.60,
                          0.30, 0.01, help="Below this = Low Risk")
    high_thr = st.slider("Medium → High threshold (PD)", 0.30, 0.90,
                          0.60, 0.01, help="Above this = High Risk")
    if low_thr >= high_thr:
        st.error("Low threshold must be below High threshold")
        st.stop()

    st.markdown("---")
    st.markdown("## 📊 Portfolio Constraints")
    st.caption("Set target allocation % per loan type. Leave 0 for no constraint.")

    constraints = {}
    total_alloc = 0
    for lt in LOAN_TYPES:
        pct = st.number_input(f"{lt} (%)", 0, 100, 0, 5, key=f"alloc_{lt}")
        constraints[lt] = pct
        total_alloc += pct

    if total_alloc > 100:
        st.error(f"Total allocation {total_alloc}% > 100%. Reduce values.")
    elif total_alloc > 0:
        st.success(f"Total constrained: {total_alloc}%")

    st.markdown("---")
    st.markdown("## 💰 Revenue Parameters")
    avg_loan_amt = st.number_input("Avg loan amount (₹)", 100000, 5000000,
                                   1000000, 100000)
    bank_cof     = st.number_input("Bank cost of funds (%)", 1.0, 12.0, 6.5, 0.1,
                                   help="Net Interest Margin = Rate - CoF")


# ════════════════════════════════════════════════════════════
# LOAD + SCORE DATA
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Scoring applicants...")
def load_and_score(file_bytes: bytes, fname: str) -> pd.DataFrame:
    if fname.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    probs      = score_batch(df)
    df["pd"]   = np.round(probs, 4)
    return df


if uploaded is None:
    st.info("👈 Upload a CSV or Excel file with applicant data to begin. "
            "Use the same format as `loan_test.csv`.")
    st.markdown("""
    **Required columns:** `application_date`, `age`, `gender`, `education`, `state`,
    `urban_rural`, `employment_type`, `employment_years`, `annual_income_inr`,
    `loan_type`, `loan_purpose`, `loan_amount_inr`, `loan_tenure_months`,
    `interest_rate_pct`, `credit_score`, `num_existing_loans`, `dti_ratio`,
    `ltv_ratio`, `has_collateral`, `bureau_enquiries_6m`, `missed_payments_2y`,
    `savings_account_balance_inr`
    """)
    st.stop()

raw_df = load_and_score(uploaded.read(), uploaded.name)

# Apply policy: classify each applicant under current thresholds
df = raw_df.copy()
df["risk_band"] = df["pd"].apply(
    lambda p: "Low" if p < low_thr else ("Medium" if p < high_thr else "High")
)
df["decision"] = df["risk_band"].apply(
    lambda r: "Approve" if r == "Low" else ("Review" if r == "Medium" else "Reject")
)

# Expected default per applicant (= PD itself for approved)
df["expected_default_contribution"] = np.where(
    df["decision"] == "Approve", df["pd"], 0
)

# Estimate interest income
loan_amt_col = "loan_amount_inr" if "loan_amount_inr" in df.columns else None
df["loan_amt"] = df[loan_amt_col] if loan_amt_col else avg_loan_amt

df["nim"]              = np.where(df["decision"]=="Approve",
                                  df["interest_rate_pct"] - bank_cof
                                  if "interest_rate_pct" in df.columns
                                  else (9.0 - bank_cof), 0)
df["expected_revenue"] = df["loan_amt"] * (df["nim"] / 100) * np.where(
    df["decision"]=="Approve", 1, 0)

N           = len(df)
n_approve   = (df["decision"]=="Approve").sum()
n_review    = (df["decision"]=="Review").sum()
n_reject    = (df["decision"]=="Reject").sum()
exp_defaults= df["expected_default_contribution"].sum()
total_rev   = df["expected_revenue"].sum()


# ════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Portfolio Overview",
    "🎚️ Policy Simulator",
    "📐 Portfolio Constraints",
    "🔍 Applicant Explorer",
])


# ─────────────────────────────────────────────
# TAB 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Portfolio Overview")

    # KPI row
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(f"""<div class="metric-card">
      <div class="val">{N:,}</div><div class="lbl">Total Applicants</div></div>""",
      unsafe_allow_html=True)
    k2.markdown(f"""<div class="metric-card">
      <div class="val" style="color:#22c55e">{n_approve:,}</div>
      <div class="lbl">Approved ({n_approve/N*100:.0f}%)</div></div>""",
      unsafe_allow_html=True)
    k3.markdown(f"""<div class="metric-card">
      <div class="val" style="color:#f59e0b">{n_review:,}</div>
      <div class="lbl">Manual Review ({n_review/N*100:.0f}%)</div></div>""",
      unsafe_allow_html=True)
    k4.markdown(f"""<div class="metric-card">
      <div class="val" style="color:#ef4444">{n_reject:,}</div>
      <div class="lbl">Rejected ({n_reject/N*100:.0f}%)</div></div>""",
      unsafe_allow_html=True)
    k5.markdown(f"""<div class="metric-card">
      <div class="val">₹{total_rev/1e7:.1f}Cr</div>
      <div class="lbl">Expected NIM Revenue</div></div>""",
      unsafe_allow_html=True)

    st.markdown("---")

    row1c1, row1c2 = st.columns(2)

    # Risk distribution donut
    with row1c1:
        st.markdown("**Risk Band Distribution**")
        counts = df["risk_band"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        wedge_colors = [RISK_COLORS.get(r,"gray") for r in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            colors=wedge_colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"linewidth":2,"edgecolor":"white"},
        )
        for at in autotexts: at.set_fontsize(9); at.set_color("white")
        ax.set_title("Risk Band Distribution", fontweight="bold", pad=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # PD histogram
    with row1c2:
        st.markdown("**Default Probability Distribution**")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(df["pd"][df["risk_band"]=="Low"],    bins=30, color="#22c55e",
                alpha=0.7, label="Low")
        ax.hist(df["pd"][df["risk_band"]=="Medium"], bins=30, color="#f59e0b",
                alpha=0.7, label="Medium")
        ax.hist(df["pd"][df["risk_band"]=="High"],   bins=30, color="#ef4444",
                alpha=0.7, label="High")
        ax.axvline(low_thr,  color="#1B3A6B", linestyle="--", lw=1.5,
                   label=f"Low thr={low_thr:.2f}")
        ax.axvline(high_thr, color="#991B1B", linestyle="--", lw=1.5,
                   label=f"High thr={high_thr:.2f}")
        ax.set_xlabel("Default Probability"); ax.set_ylabel("Count")
        ax.set_title("PD Distribution by Risk Band", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    row2c1, row2c2 = st.columns(2)

    # Decision by loan type
    with row2c1:
        st.markdown("**Decision by Loan Type**")
        if "loan_type" in df.columns:
            grp = df.groupby(["loan_type","decision"]).size().unstack(fill_value=0)
            for c in ["Approve","Review","Reject"]:
                if c not in grp.columns: grp[c] = 0
            fig, ax = plt.subplots(figsize=(6,4))
            x = np.arange(len(grp))
            w = 0.25
            bars_a = ax.bar(x-w, grp["Approve"], w, color="#22c55e", label="Approve")
            bars_r = ax.bar(x,   grp["Review"],  w, color="#f59e0b", label="Review")
            bars_j = ax.bar(x+w, grp["Reject"],  w, color="#ef4444", label="Reject")
            ax.set_xticks(x); ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Count"); ax.legend(fontsize=8)
            ax.set_title("Decisions by Loan Type", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Expected revenue by loan type
    with row2c2:
        st.markdown("**Expected Revenue by Loan Type**")
        if "loan_type" in df.columns:
            rev_grp = df[df["decision"]=="Approve"].groupby("loan_type")["expected_revenue"].sum() / 1e7
            fig, ax = plt.subplots(figsize=(6,4))
            colors_lt = [LOAN_COLORS.get(lt,"#6b7280") for lt in rev_grp.index]
            bars = ax.barh(rev_grp.index, rev_grp.values, color=colors_lt)
            for bar, val in zip(bars, rev_grp.values):
                ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                        f"₹{val:.2f}Cr", va="center", fontsize=8)
            ax.set_xlabel("Expected NIM Revenue (₹ Crore)")
            ax.set_title("Revenue by Loan Type", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # Expected defaults
    st.markdown("---")
    st.markdown("**Expected Default Exposure**")
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    exp_c1.metric("Total Expected Defaults", f"{exp_defaults:.1f}")
    exp_c2.metric("Expected Default Rate",
                  f"{exp_defaults/max(n_approve,1)*100:.1f}%")
    exp_c3.metric("Expected Default ₹ Exposure",
                  f"₹{exp_defaults*avg_loan_amt/1e7:.2f} Cr")


# ─────────────────────────────────────────────
# TAB 2 — POLICY SIMULATOR
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Policy Simulator")
    st.caption(
        "Adjust thresholds in the sidebar. This tab shows exactly how approval/rejection/review "
        "rates and expected defaults respond — before you commit to a policy change."
    )

    # Build threshold sweep for current data
    thresh_range = np.arange(0.10, 0.95, 0.05)
    approval_rates, rejection_rates, exp_def_rates, rev_rates = [], [], [], []

    for thr in thresh_range:
        app_ = (df["pd"] < thr).sum() / N
        rej_ = (df["pd"] >= high_thr).sum() / N
        exp_ = df.loc[df["pd"] < thr, "pd"].sum()
        rev_ = df.loc[df["pd"] < thr, "expected_revenue"].sum() / 1e7
        approval_rates.append(app_)
        rejection_rates.append(rej_)
        exp_def_rates.append(exp_)
        rev_rates.append(rev_)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Approval rate curve
    axes[0,0].plot(thresh_range, [r*100 for r in approval_rates],
                   color="#22c55e", lw=2)
    axes[0,0].axvline(high_thr, color="#1B3A6B", linestyle="--", lw=1.5,
                      label=f"Current={high_thr:.2f}")
    axes[0,0].set_xlabel("High-Risk Threshold"); axes[0,0].set_ylabel("Approval Rate (%)")
    axes[0,0].set_title("Approval Rate vs Threshold", fontweight="bold")
    axes[0,0].legend(fontsize=8)

    # Expected defaults curve
    axes[0,1].plot(thresh_range, exp_def_rates, color="#ef4444", lw=2)
    axes[0,1].axvline(high_thr, color="#1B3A6B", linestyle="--", lw=1.5,
                      label=f"Current={high_thr:.2f}")
    axes[0,1].set_xlabel("High-Risk Threshold"); axes[0,1].set_ylabel("Expected Defaults (count)")
    axes[0,1].set_title("Expected Defaults vs Threshold", fontweight="bold")
    axes[0,1].legend(fontsize=8)

    # Revenue curve
    axes[1,0].plot(thresh_range, rev_rates, color="#3b82f6", lw=2)
    axes[1,0].axvline(high_thr, color="#1B3A6B", linestyle="--", lw=1.5,
                      label=f"Current={high_thr:.2f}")
    axes[1,0].set_xlabel("High-Risk Threshold"); axes[1,0].set_ylabel("Expected Revenue (₹ Cr)")
    axes[1,0].set_title("Expected Revenue vs Threshold", fontweight="bold")
    axes[1,0].legend(fontsize=8)

    # Risk vs Reward scatter
    axes[1,1].scatter(exp_def_rates, rev_rates, c=thresh_range,
                      cmap="RdYlGn_r", s=60, zorder=3)
    # mark current point
    cur_exp  = df.loc[df["pd"] < high_thr, "pd"].sum()
    cur_rev  = df.loc[df["pd"] < high_thr, "expected_revenue"].sum() / 1e7
    axes[1,1].scatter([cur_exp], [cur_rev], color="#1B3A6B", s=120, zorder=4,
                      label=f"Current thr={high_thr:.2f}")
    axes[1,1].set_xlabel("Expected Defaults"); axes[1,1].set_ylabel("Expected Revenue (₹ Cr)")
    axes[1,1].set_title("Risk vs Reward Frontier", fontweight="bold")
    axes[1,1].legend(fontsize=8)

    plt.suptitle("Policy Simulation — Impact of Threshold Changes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Current Policy Metrics at Selected Thresholds**")
    sm1,sm2,sm3,sm4,sm5 = st.columns(5)
    sm1.metric("Low threshold",  f"{low_thr:.2f}")
    sm2.metric("High threshold", f"{high_thr:.2f}")
    sm3.metric("Approve",  f"{n_approve/N*100:.1f}%")
    sm4.metric("Review",   f"{n_review/N*100:.1f}%")
    sm5.metric("Reject",   f"{n_reject/N*100:.1f}%")

    st.markdown("**NPA Risk vs Growth Trade-off**")
    risk_info = pd.DataFrame({
        "Threshold":  thresh_range,
        "Approval %": [r*100 for r in approval_rates],
        "Exp Defaults": exp_def_rates,
        "Revenue (₹Cr)": rev_rates,
    })
    st.dataframe(risk_info.style.format({
        "Threshold":".2f","Approval %":".1f",
        "Exp Defaults":".1f","Revenue (₹Cr)":".2f"
    }), use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3 — PORTFOLIO CONSTRAINTS
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Portfolio Allocation Constraints")
    st.caption(
        "Set target % allocation per loan type in the sidebar. "
        "The system shows how many applicants to approve per segment, "
        "and the risk/revenue profile of the constrained portfolio."
    )

    active_constraints = {k:v for k,v in constraints.items() if v > 0}

    if not active_constraints:
        st.info("No constraints set. Use the sidebar sliders to define allocation targets.")
    else:
        st.markdown("**Constrained Portfolio — Approved Counts & Risk Profile**")

        if "loan_type" not in df.columns:
            st.error("loan_type column not found in uploaded data.")
        else:
            total_budget = n_approve  # total approvals budget
            alloc_rows   = []

            for lt, pct in active_constraints.items():
                target_n  = int(total_budget * pct / 100)
                subset    = df[df["loan_type"]==lt].sort_values("pd")
                approved  = subset.head(target_n)
                n_lt      = len(approved)
                avg_pd    = approved["pd"].mean() if n_lt > 0 else 0
                tot_rev   = approved["expected_revenue"].sum() / 1e7
                exp_def   = approved["pd"].sum()
                risk_dist = approved["risk_band"].value_counts().to_dict()

                alloc_rows.append({
                    "Loan Type":      lt,
                    "Target %":       f"{pct}%",
                    "Target N":       target_n,
                    "Actual Approved":n_lt,
                    "Avg PD":         f"{avg_pd:.3f}",
                    "Low Risk":       risk_dist.get("Low",0),
                    "Medium Risk":    risk_dist.get("Medium",0),
                    "High Risk":      risk_dist.get("High",0),
                    "Exp Defaults":   f"{exp_def:.1f}",
                    "Revenue (₹Cr)":  f"{tot_rev:.2f}",
                })

            alloc_df = pd.DataFrame(alloc_rows)
            st.dataframe(alloc_df, use_container_width=True)

            # Stacked bar: risk composition per loan type
            st.markdown("**Risk Composition in Constrained Portfolio**")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            lts   = [r["Loan Type"] for r in alloc_rows]
            lows  = [r["Low Risk"]    for r in alloc_rows]
            meds  = [r["Medium Risk"] for r in alloc_rows]
            highs = [r["High Risk"]   for r in alloc_rows]

            x = np.arange(len(lts))
            axes[0].bar(x, lows,  color="#22c55e", label="Low")
            axes[0].bar(x, meds,  bottom=lows, color="#f59e0b", label="Medium")
            axes[0].bar(x, highs,
                        bottom=[l+m for l,m in zip(lows,meds)],
                        color="#ef4444", label="High")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(lts, rotation=30, ha="right", fontsize=8)
            axes[0].set_ylabel("Approved Count")
            axes[0].set_title("Risk Composition by Loan Type", fontweight="bold")
            axes[0].legend(fontsize=8)

            # Revenue vs default exposure
            revs  = [float(r["Revenue (₹Cr)"]) for r in alloc_rows]
            exps  = [float(r["Exp Defaults"])   for r in alloc_rows]
            axes[1].scatter(exps, revs,
                            c=[LOAN_COLORS.get(lt,"gray") for lt in lts],
                            s=150, zorder=3)
            for i, lt in enumerate(lts):
                axes[1].annotate(lt.replace("_Loan",""), (exps[i], revs[i]),
                                 textcoords="offset points", xytext=(6,4), fontsize=8)
            axes[1].set_xlabel("Expected Defaults")
            axes[1].set_ylabel("Expected Revenue (₹ Cr)")
            axes[1].set_title("Risk vs Revenue per Loan Type", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Unconstrained vs constrained comparison
            st.markdown("---")
            st.markdown("**Unconstrained vs Constrained Portfolio Comparison**")
            comp_c1, comp_c2 = st.columns(2)

            total_constrained_approvals = sum(r["Actual Approved"] for r in alloc_rows)
            comp_c1.metric("Unconstrained approvals", f"{n_approve:,}")
            comp_c2.metric("Constrained approvals",  f"{total_constrained_approvals:,}")

            # Loan type money allocation pie
            if total_constrained_approvals > 0:
                constrained_rev = [float(r["Revenue (₹Cr)"]) for r in alloc_rows]
                fig, ax = plt.subplots(figsize=(5,4))
                ax.pie(constrained_rev, labels=lts, autopct="%1.1f%%",
                       colors=[LOAN_COLORS.get(lt,"gray") for lt in lts],
                       startangle=90,
                       wedgeprops={"linewidth":2,"edgecolor":"white"})
                ax.set_title("Revenue Share — Constrained Portfolio", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            # Download constrained portfolio
            st.download_button(
                "⬇️ Download Constrained Portfolio CSV",
                data=alloc_df.to_csv(index=False).encode(),
                file_name="constrained_portfolio.csv",
                mime="text/csv", use_container_width=True,
            )


# ─────────────────────────────────────────────
# TAB 4 — APPLICANT EXPLORER
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Applicant-Level Explorer")
    st.caption("Filter and explore individual applicant decisions.")

    col_f1, col_f2, col_f3 = st.columns(3)
    filt_decision = col_f1.multiselect("Decision", ["Approve","Review","Reject"],
                                       default=["Approve","Review","Reject"])
    filt_risk     = col_f2.multiselect("Risk Band", ["Low","Medium","High"],
                                       default=["Low","Medium","High"])
    if "loan_type" in df.columns:
        filt_lt = col_f3.multiselect("Loan Type", LOAN_TYPES, default=LOAN_TYPES)
    else:
        filt_lt = LOAN_TYPES

    mask = (
        df["decision"].isin(filt_decision) &
        df["risk_band"].isin(filt_risk)
    )
    if "loan_type" in df.columns:
        mask &= df["loan_type"].isin(filt_lt)

    filtered = df[mask]
    st.write(f"**{len(filtered):,} applicants** match current filters")

    # Display columns
    disp_cols = ["pd","risk_band","decision"]
    for c in ["loan_id","loan_type","loan_amount_inr","credit_score",
              "annual_income_inr","dti_ratio","missed_payments_2y","expected_revenue"]:
        if c in filtered.columns: disp_cols.append(c)

    st.dataframe(
        filtered[disp_cols].sort_values("pd", ascending=False).head(500),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("**Risk Factor Analysis (Approved Applicants)**")
    approved_df = filtered[filtered["decision"]=="Approve"]
    if len(approved_df) > 5:
        num_cols = ["pd","credit_score","dti_ratio","annual_income_inr",
                    "loan_amount_inr","missed_payments_2y"]
        num_cols = [c for c in num_cols if c in approved_df.columns]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, col in enumerate(num_cols[:6]):
            axes[i].hist(approved_df[col].dropna(), bins=30,
                         color="#3b82f6", alpha=0.8, edgecolor="white")
            axes[i].set_title(col.replace("_"," ").title(), fontweight="bold")
            axes[i].set_ylabel("Count")
        plt.suptitle("Distribution of Key Features — Approved Applicants",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Download filtered results
    st.download_button(
        "⬇️ Download Filtered Results CSV",
        data=filtered.to_csv(index=False).encode(),
        file_name="filtered_applicants.csv",
        mime="text/csv", use_container_width=True,
    )

