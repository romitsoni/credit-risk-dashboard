"""
Credit Risk Monitor - dashboard entry point.
Run from the project root:  streamlit run app/app.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                # so data paths work locally and on the cloud
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st

st.set_page_config(page_title="Credit Risk Monitor", page_icon="📊", layout="wide")

# On Streamlit Cloud the key lives in Secrets; copy it to the environment.
try:
    for k in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
except Exception:
    pass  # no secrets file locally: .env is used instead

from views import common, portfolio, account, ask, anomalies, collections, model_monitor

st.title("📊 Credit Risk Monitor")
st.caption("Synthetic portfolio · 10,000 customers · 14,685 loans")

common.ensure_data()
data = common.load_data()
fdf = common.sidebar_filters(data["scored"])

tabs = st.tabs(["📊 Portfolio", "🔎 Account", "💬 Ask the Portfolio",
                "🚩 Anomaly Review", "💰 Collections", "📈 Model & Monitoring"])
with tabs[0]:
    portfolio.render(data, fdf)
with tabs[1]:
    account.render(data, fdf)
with tabs[2]:
    ask.render(data, fdf)
with tabs[3]:
    anomalies.render(data, fdf)
with tabs[4]:
    collections.render(data, fdf)
with tabs[5]:
    model_monitor.render(data, fdf)