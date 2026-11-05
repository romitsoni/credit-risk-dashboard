"""Shared helpers: build data if missing, load data, sidebar filters, formatting."""
import json
import os
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
    st.sidebar.caption("Empty = all. Applies to Portfolio, Account and Anomaly views.")
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