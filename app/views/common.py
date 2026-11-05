"""Shared helpers: build data if missing, load data, sidebar filters, formatting."""
import json
import os
import re
import pandas as pd
import streamlit as st

import analytics

DATA = analytics.DATA
SEG_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825",
              "High": "#ef6c00", "Very High": "#c62828"}
FILES = ["scored_accounts.csv", "anomalies.csv", "model_metrics.json",
         "decile_table.csv", "feature_importance.csv"]


def ensure_data():
    """If processed files are missing (e.g. first run on the cloud), build them."""
    if all((DATA / f).exists() for f in FILES):
        return
    with st.spinner("First run: cleaning data and training the model (about 1 minute)..."):
        import ingest
        import model
        import anomaly
        ingest.build()
        model.main()
        anomaly.main()
        analytics.main()


@st.cache_data
def load_data():
    scored = pd.read_csv(DATA / "scored_accounts.csv")
    for col, order in analytics.ORDER.items():
        if col in scored.columns:
            scored[col] = pd.Categorical(scored[col], order, ordered=True)
    with open(DATA / "model_metrics.json") as f:
        metrics = json.load(f)
    return {
        "scored": scored,
        "anomalies": pd.read_csv(DATA / "anomalies.csv"),
        "metrics": metrics,
        "deciles": pd.read_csv(DATA / "decile_table.csv"),
        "importance": pd.read_csv(DATA / "feature_importance.csv"),
    }


def sidebar_filters(df):
    st.sidebar.header("Filters")
    st.sidebar.caption("Empty = all loans. Applies to every tab except Model & Monitoring, "
                       "which always uses all test loans.")
    out = df
    for col, label in [("loan_type", "Loan type"), ("region", "Region"),
                       ("employment_type", "Employment type"),
                       ("income_band", "Income band"), ("risk_segment", "Risk segment")]:
        if hasattr(df[col], "cat"):
            opts = list(df[col].cat.categories)
        else:
            opts = sorted(df[col].dropna().unique())
        sel = st.sidebar.multiselect(label, opts)
        if sel:
            out = out[out[col].isin(sel)]
    st.sidebar.divider()
    key_on = bool(os.getenv("GEMINI_API_KEY"))
    st.sidebar.caption(f"AI assistant: {'🟢 key set' if key_on else '⚪ no key (quick questions only)'}")
    return out


def money(v):
    return f"{v / 1e6:,.1f}M" if abs(v) >= 1e6 else f"{v / 1e3:,.1f}K"


def pct(v):
    return f"{v:.1%}"


def show_table(t):
    """Format a profile table for display."""
    d = t.copy()
    for c in ["default_rate", "exposure_share", "avg_pd"]:
        if c in d.columns:
            d[c] = d[c].map(pct)
    if "exposure" in d.columns:
        d["exposure"] = d["exposure"].map(money)
    st.dataframe(d, hide_index=True, use_container_width=True)


# ---------- Plain-English risk reasons ----------


def _yrs(v):
    if v == 0:
        return "0 years"
    return f"{v * 12:.0f} months" if v < 1 else f"{v:.1f} years"


def _days(v):
    return f"{v:.0f} days"


# feature: (which side is risky, short phrase with {v} = this customer's value, formatter)
REASON_TEXT = {
    "annual_income": ("low", "Low income ({v})", lambda v: money(v)),
    "years_employed": ("low", "New in job ({v})", _yrs),
    "age": ("low", "Young ({v})", lambda v: f"{v:.0f}"),
    "years_at_residence": ("low", "New at address ({v})", _yrs),
    "rep_mean_dpd": ("high", "Pays us late (avg {v})", _days),
    "rep_max_dpd": ("high", "Was very late with us (up to {v})", _days),
    "rep_late": ("high", "Late with us in {v} months", lambda v: f"{v:.0f}"),
    "rep_last_balance": ("high", "Still owes a lot ({v})", lambda v: money(v)),
    "bur_mean_dpd": ("high", "Pays other banks late (avg {v})", _days),
    "bur_max_dpd": ("high", "Was very late at other banks (up to {v})", _days),
    "bur_active": ("high", "Many loans elsewhere ({v})", lambda v: f"{v:.0f}"),
    "bur_writeoffs": ("high", "Unpaid loans at other banks ({v})", lambda v: f"{v:.0f}"),
    "bur_utilization": ("high", "Uses most of credit elsewhere ({v})", lambda v: f"{v:.0%}"),
    "loan_to_income": ("high", "Big loan for income ({v})", lambda v: f"{v:.1f}x"),
    "loan_amount": ("high", "Large loan ({v})", lambda v: money(v)),
    "interest_rate_pct": ("high", "High interest rate ({v})", lambda v: f"{v:.1f}%"),
}


@st.cache_data
def typical_values(_scored):
    cols = [c for c in REASON_TEXT if c in _scored.columns]
    return _scored[cols].median(numeric_only=True).to_dict()


def typical_text(typical):
    """One line describing a typical customer, shown once above the table."""
    parts = []
    if "annual_income" in typical:
        parts.append(f"income {money(typical['annual_income'])}")
    if "years_employed" in typical:
        parts.append(f"{_yrs(typical['years_employed'])} in job")
    if "rep_mean_dpd" in typical:
        d = typical["rep_mean_dpd"]
        parts.append("pays us on time" if d < 1 else f"pays us {d:.0f} days late on average")
    if "loan_to_income" in typical:
        parts.append(f"loan {typical['loan_to_income']:.1f}x income")
    return ", ".join(parts)


def risk_reasons(row, typical, n=3):
    """Turn 'annual_income (+0.35); rep_mean_dpd (+0.33)' into short plain phrases,
    biggest reason first. Only shows a reason when the value is on the risky side."""
    found = []
    for name, val in re.findall(r"([A-Za-z_]+)\s*\(([+-]?[\d.]+)\)", str(row.get("top_drivers", ""))):
        if float(val) > 0:
            found.append((float(val), name))
    out = []
    for _, name in sorted(found, reverse=True):
        if name not in REASON_TEXT:
            continue
        risky_side, phrase, fmt = REASON_TEXT[name]
        v, t = row.get(name), typical.get(name)
        if v is None or t is None or pd.isna(v):
            continue
        side = "high" if v > t else ("low" if v < t else None)
        if side != risky_side:
            continue  # value is on the safe side: not a clear reason
        out.append(phrase.format(v=fmt(v)))
        if len(out) == n:
            break
    return " · ".join(out) if out else "No single strong reason"