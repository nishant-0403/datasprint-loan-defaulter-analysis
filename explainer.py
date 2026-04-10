# ============================================================
# explainer.py
# SHAP explainability + counterfactual search (Features 2,3,4)
# 100% free — uses only SHAP + your trained XGBoost model.
# LLM explanation uses HuggingFace Inference API (free tier)
# with a lightweight summarization approach as fallback.
# ============================================================
import copy, os, re
import numpy as np
import pandas as pd
import shap
import requests

from pipeline import MODEL, THRESHOLD, preprocess, score_single

# ── Free LLM via HuggingFace Inference API ───────────────────
# Uses google/flan-t5-large (free, no signup required for basic use)
# Falls back to rule-based text if API is unavailable.

#HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
#HF_HEADERS = {}
#_hf_token  = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}
#if _hf_token:
#   HF_HEADERS = {"Authorization": f"Bearer {_hf_token}"}


def _hf_generate(prompt: str, max_new_tokens: int = 300) -> str:
    """Call HuggingFace Inference API. Falls back to None on failure."""
    try:
        resp = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
    except Exception:
        pass
    return None


# ── Feature 2: SHAP computation ──────────────────────────────

def compute_shap(X_processed: pd.DataFrame):
    """Returns (shap_values_1d, feature_names, base_value)."""
    explainer = shap.TreeExplainer(MODEL)
    raw       = explainer.shap_values(X_processed)
    if isinstance(raw, list):
        vals = np.array(raw[1], dtype=np.float64).flatten()
        ev   = float(explainer.expected_value[1])
    else:
        vals = np.array(raw, dtype=np.float64).flatten()
        ev   = float(explainer.expected_value)
    return vals, X_processed.columns.tolist(), ev


def top_shap_factors(shap_vals, feat_names, applicant_vals, n=5):
    idx = np.argsort(np.abs(shap_vals))[::-1][:n]
    return [
        {
            "feature":    feat_names[i],
            "shap":       round(float(shap_vals[i]), 4),
            "value":      round(float(applicant_vals.iloc[i]), 4),
            "direction":  "risk_increase" if shap_vals[i] > 0 else "risk_decrease",
        }
        for i in idx
    ]


def _rule_based_explanation(prob: float, factors: list) -> str:
    """Fallback explanation when LLM is unavailable."""
    risk_word = "high" if prob > 0.6 else ("moderate" if prob > 0.4 else "low")
    increase  = [f["feature"].replace("_", " ") for f in factors if f["direction"] == "risk_increase"]
    decrease  = [f["feature"].replace("_", " ") for f in factors if f["direction"] == "risk_decrease"]

    lines = [
        f"Your application was assessed with a default probability of {prob:.1%}, indicating {risk_word} risk.",
    ]
    if increase:
        lines.append(f"The main factors increasing risk were: {', '.join(increase[:3])}.")
    if decrease:
        lines.append(f"Factors working in your favour included: {', '.join(decrease[:2])}.")
    lines.append(
        "To improve your eligibility, focus on reducing existing loan obligations, "
        "maintaining timely EMI payments, and building a higher savings balance."
    )
    return " ".join(lines)


def explain_rejection(prob: float, factors: list) -> str:
    """Plain-language rejection explanation. Uses HF LLM or rule-based fallback."""
    increase = [f for f in factors if f["direction"] == "risk_increase"]
    decrease = [f for f in factors if f["direction"] == "risk_decrease"]

    factor_lines = "\n".join(
        f"- {f['feature'].replace('_',' ')}: value={f['value']}, impact={'increases' if f['direction']=='risk_increase' else 'reduces'} risk"
        for f in factors
    )

    prompt = (
        f"A loan application was rejected with a default probability of {prob:.0%}. "
        f"Explain in 3 simple sentences why, based on these risk factors:\n{factor_lines}\n"
        f"Write for a non-technical applicant. Be empathetic and constructive."
    )

    result = _hf_generate(prompt, max_new_tokens=200)
    if result and len(result) > 40:
        return result

    return _rule_based_explanation(prob, factors)


# ── Feature 3: Loan parameter counterfactual ─────────────────

def _rescore(applicant: dict) -> float:
    try:
        prob, _, _ = score_single(applicant)
        return prob
    except Exception:
        return 1.0


TENURE_OPTIONS = [12, 24, 36, 60, 84, 120, 180, 240, 300, 360]
AMT_STEP       = 50_000


def find_loan_counterfactual(applicant: dict) -> dict:
    """
    Keeps applicant profile fixed. Tweaks loan params.
    Returns nearest combination that crosses approval threshold.
    """
    orig_prob = _rescore(applicant)
    if orig_prob < THRESHOLD:
        return {"found": True, "already_approved": True, "original_prob": orig_prob}

    best = None

    # 1 — Add collateral
    c = copy.deepcopy(applicant)
    if not c.get("has_collateral"):
        c["has_collateral"] = 1
        p = _rescore(c)
        if p < THRESHOLD:
            best = {"params": c, "prob": p, "changes": ["Add collateral / security"]}

    # 2 — Extend tenure
    if not best:
        orig_t = applicant.get("loan_tenure_months", 120)
        for t in sorted(x for x in TENURE_OPTIONS if x > orig_t):
            c = copy.deepcopy(applicant)
            c["loan_tenure_months"] = t
            p = _rescore(c)
            if p < THRESHOLD:
                best = {"params": c, "prob": p,
                        "changes": [f"Extend tenure to {t} months ({t//12} yrs)"]}
                break

    # 3 — Reduce loan amount
    if not best:
        orig_a = applicant.get("loan_amount_inr", 1_000_000)
        amt = orig_a - AMT_STEP
        while amt >= 100_000:
            c = copy.deepcopy(applicant)
            c["loan_amount_inr"] = amt
            p = _rescore(c)
            if p < THRESHOLD:
                best = {"params": c, "prob": p,
                        "changes": [f"Reduce loan to ₹{amt:,.0f} (cut of ₹{orig_a-amt:,.0f})"]}
                break
            amt -= AMT_STEP

    # 4 — Amount + tenure combo
    if not best:
        orig_a = applicant.get("loan_amount_inr", 1_000_000)
        orig_t = applicant.get("loan_tenure_months", 120)
        for t in sorted(x for x in TENURE_OPTIONS if x > orig_t):
            amt = orig_a - AMT_STEP
            while amt >= 100_000:
                c = copy.deepcopy(applicant)
                c["loan_amount_inr"] = amt
                c["loan_tenure_months"] = t
                p = _rescore(c)
                if p < THRESHOLD:
                    best = {
                        "params": c, "prob": p,
                        "changes": [
                            f"Reduce loan to ₹{amt:,.0f}",
                            f"Extend tenure to {t} months",
                        ],
                    }
                    break
                amt -= AMT_STEP
            if best:
                break

    # 5 — Collateral + reduced amount
    if not best:
        orig_a = applicant.get("loan_amount_inr", 1_000_000)
        amt = orig_a - AMT_STEP
        while amt >= 100_000:
            c = copy.deepcopy(applicant)
            c["has_collateral"] = 1
            c["loan_amount_inr"] = amt
            p = _rescore(c)
            if p < THRESHOLD:
                best = {
                    "params": c, "prob": p,
                    "changes": [
                        "Add collateral",
                        f"Reduce loan to ₹{amt:,.0f}",
                    ],
                }
                break
            amt -= AMT_STEP

    if not best:
        return {"found": False, "original_prob": orig_prob,
                "message": "No feasible loan parameter combination found."}

    return {
        "found": True, "already_approved": False,
        "original_prob": orig_prob,
        "approved_prob": best["prob"],
        "approved_params": best["params"],
        "changes": best["changes"],
    }


def narrate_loan_cf(applicant: dict, cf: dict) -> str:
    if not cf.get("found") or cf.get("already_approved"):
        return ""
    changes_str = "; ".join(cf["changes"])
    prompt = (
        f"A loan was rejected. These changes would get it approved: {changes_str}. "
        f"Default probability would drop from {cf['original_prob']:.0%} to {cf['approved_prob']:.0%}. "
        f"Explain in 2 sentences why these changes reduce risk, for a non-technical applicant."
    )
    result = _hf_generate(prompt, max_new_tokens=150)
    if result and len(result) > 30:
        return result
    return (
        f"If you {changes_str.lower()}, the bank would be more confident in your ability to repay. "
        f"Your default probability would drop from {cf['original_prob']:.0%} to {cf['approved_prob']:.0%}, "
        f"bringing it below the approval threshold."
    )


# ── Feature 4: Profile improvement counterfactual ────────────

def find_profile_counterfactual(applicant: dict) -> list:
    """
    Keeps loan params fixed. Finds which profile changes lead to approval.
    Returns list of scenarios sorted by achievability.
    """
    orig_prob = _rescore(applicant)
    if orig_prob < THRESHOLD:
        return []

    scenarios = []

    # 1 — Clear missed payments
    if applicant.get("missed_payments_2y", 0) > 0:
        c = copy.deepcopy(applicant)
        c["missed_payments_2y"] = 0
        p = _rescore(c)
        if p < THRESHOLD:
            scenarios.append({
                "field": "missed_payments_2y",
                "from": applicant["missed_payments_2y"], "to": 0,
                "prob": round(p, 4),
                "description": "Clear all missed EMI payments and maintain clean repayment record",
                "difficulty": "Moderate",
            })

    # 2 — Close existing loans
    orig_loans = int(applicant.get("num_existing_loans", 0))
    for n in range(1, min(orig_loans + 1, 4)):
        c = copy.deepcopy(applicant)
        c["num_existing_loans"] = max(0, orig_loans - n)
        c["dti_ratio"] = max(0.05, float(applicant.get("dti_ratio", 0.3)) * (1 - 0.15 * n))
        p = _rescore(c)
        if p < THRESHOLD:
            scenarios.append({
                "field": "num_existing_loans",
                "from": orig_loans, "to": c["num_existing_loans"],
                "prob": round(p, 4),
                "description": f"Close {n} existing loan(s) before applying",
                "difficulty": "Moderate",
            })
            break

    # 3 — Improve credit score
    orig_cs = int(applicant.get("credit_score", 650))
    cs = orig_cs + 10
    while cs <= 900:
        c = copy.deepcopy(applicant)
        c["credit_score"] = cs
        p = _rescore(c)
        if p < THRESHOLD:
            diff = cs - orig_cs
            scenarios.append({
                "field": "credit_score",
                "from": orig_cs, "to": cs,
                "prob": round(p, 4),
                "description": f"Improve credit score by {diff} points (pay bills on time, reduce utilisation)",
                "difficulty": "Hard" if diff > 50 else "Moderate",
            })
            break
        cs += 10

    # 4 — Increase income
    orig_inc = float(applicant.get("annual_income_inr", 500_000))
    inc = orig_inc + 25_000
    while inc <= orig_inc * 2.5:
        c = copy.deepcopy(applicant)
        c["annual_income_inr"] = inc
        p = _rescore(c)
        if p < THRESHOLD:
            scenarios.append({
                "field": "annual_income_inr",
                "from": orig_inc, "to": inc,
                "prob": round(p, 4),
                "description": f"Increase annual income to ₹{inc:,.0f} (add co-applicant or income source)",
                "difficulty": "Hard",
            })
            break
        inc += 25_000

    # 5 — Increase savings
    orig_sav = float(applicant.get("savings_account_balance_inr", 50_000))
    sav = orig_sav + 10_000
    while sav <= orig_sav + 500_000:
        c = copy.deepcopy(applicant)
        c["savings_account_balance_inr"] = sav
        p = _rescore(c)
        if p < THRESHOLD:
            scenarios.append({
                "field": "savings_account_balance_inr",
                "from": orig_sav, "to": sav,
                "prob": round(p, 4),
                "description": f"Build savings to ₹{sav:,.0f} (save ₹{sav-orig_sav:,.0f} more)",
                "difficulty": "Easy",
            })
            break
        sav += 10_000

    order = {"Easy": 0, "Moderate": 1, "Hard": 2}
    scenarios.sort(key=lambda x: order.get(x["difficulty"], 3))
    return scenarios[:4]


def narrate_profile_cf(scenarios: list) -> str:
    if not scenarios:
        return "No single profile change found within feasible bounds. Consider applying for a smaller loan or adding a co-applicant."

    top = scenarios[0]
    prompt = (
        f"A loan applicant was rejected. The easiest improvement is: {top['description']} "
        f"(difficulty: {top['difficulty']}). This would drop the default probability to {top['prob']:.0%}. "
        f"Write 2 encouraging sentences advising the applicant what to do first."
    )
    result = _hf_generate(prompt, max_new_tokens=150)
    if result and len(result) > 30:
        return result

    lines = ["Here are the changes that would get your exact loan approved:"]
    for s in scenarios:
        lines.append(f"• {s['description']} [{s['difficulty']}]")
    lines.append(f"We recommend starting with the easiest option: {scenarios[0]['description']}.")
    return "\n".join(lines)
