"""Model & Monitoring: performance, lift, calibration, drivers, drift (PSI), fairness."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import SEG_COLORS, money, pct

LABELS = {"annual_income": "Annual income", "years_employed": "Years employed",
          "rep_mean_dpd": "Avg DPD (our loan)", "age": "Age", "rep_late": "Late months",
          "bur_mean_dpd": "Avg DPD (other lenders)", "loan_to_income": "Loan-to-income",
          "rep_max_dpd": "Max DPD (our loan)", "interest_rate_pct": "Interest rate",
          "bur_active": "Active bureau lines", "bur_writeoffs": "Bureau write-offs",
          "years_at_residence": "Years at residence", "bur_max_dpd": "Max DPD (other lenders)",
          "loan_amount": "Loan amount", "rep_last_balance": "Current balance"}


def psi(base, curr, bins=10):
    """Population Stability Index: how much a distribution has shifted."""
    base, curr = pd.Series(base).dropna(), pd.Series(curr).dropna()
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base, edges)[0] / len(base)
    c = np.histogram(curr, edges)[0] / len(curr)
    b, c = np.clip(b, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - b) * np.log(c / b)))


def psi_status(v):
    return "🟢 Stable" if v < 0.10 else ("🟡 Watch" if v < 0.25 else "🔴 Retrain")


def render(data, df):
    st.subheader("📈 Model & Monitoring")
    s = data["scored"]
    test = s[s["in_test_set"] == 1]
    m = data["metrics"]

    # ---------- Performance ----------
    st.markdown("#### 1. Performance on the held-out test set")
    st.caption(f"Test set = {m['test_rows']:,} loans from customers the model never saw "
               f"({m['split']}).")
    perf = pd.DataFrame({
        "Model": ["Logistic regression (baseline)", "XGBoost (used in dashboard)"],
        "AUC": [m["logistic_regression"]["AUC"], m["xgboost"]["AUC"]],
        "KS": [m["logistic_regression"]["KS"], m["xgboost"]["KS"]],
        "Gini": [m["logistic_regression"]["Gini"], m["xgboost"]["Gini"]]})
    st.dataframe(perf, hide_index=True, use_container_width=True)
    st.caption("AUC: how well the model ranks bad loans above good ones (0.5 = random). "
               "KS: largest gap between the bad and good score curves. Gini = 2 x AUC - 1. "
               "Both models perform about the same: the signal is mostly simple, so the "
               "extra complexity of XGBoost adds little.")

    # ---------- Lift + segments ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### 2. Lift: are defaults in the riskiest scores?")
        d = test.assign(decile=pd.qcut(test["pd_score"].rank(method="first"), 10,
                                       labels=list(range(10, 0, -1))))
        t = d.groupby("decile", observed=True).agg(
            loans=("account_id", "count"), defaults=("target_default", "sum"),
            default_rate=("target_default", "mean")).sort_index(ascending=False).reset_index()
        t["decile"] = t["decile"].astype(int)
        t = t.sort_values("decile")
        t["cum_capture"] = t["defaults"].cumsum() / t["defaults"].sum()
        top2 = t.loc[t["decile"] <= 2, "cum_capture"].max()
        fig = go.Figure()
        fig.add_bar(x=t["decile"], y=t["default_rate"], name="Default rate",
                    text=t["default_rate"].map(pct), marker_color="#1565c0")
        fig.add_scatter(x=t["decile"], y=t["cum_capture"], name="Cumulative % of defaults",
                        mode="lines+markers", line_color="#c62828")
        fig.update_layout(height=340, yaxis_tickformat=".0%",
                          xaxis_title="Score decile (1 = riskiest 10%)",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"The riskiest 20% of loans hold **{pct(top2)}** of all defaults "
                   f"(random would be 20%).")
    with right:
        st.markdown("#### 3. Risk segments (test set)")
        seg = test.groupby("risk_segment", observed=True).agg(
            loans=("account_id", "count"), default_rate=("target_default", "mean"),
            avg_pd=("pd_score", "mean"), exposure=("loan_amount", "sum")).reset_index()
        fig = px.bar(seg, x="risk_segment", y="default_rate", color="risk_segment",
                     color_discrete_map=SEG_COLORS, text=seg["default_rate"].map(pct),
                     labels={"risk_segment": "Segment", "default_rate": "Actual default rate"})
        fig.update_layout(height=340, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Honest view: test set only. Default rate rises with every segment, "
                   "so the ranking works.")

    # ---------- Calibration + drivers ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### 4. Calibration: does 20% PD mean 20% default?")
        c = test.assign(bin=pd.qcut(test["pd_score"], 10, duplicates="drop"))
        c = c.groupby("bin", observed=True).agg(predicted=("pd_score", "mean"),
                                                actual=("target_default", "mean")).reset_index()
        fig = go.Figure()
        mx = max(c["predicted"].max(), c["actual"].max()) * 1.05
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect",
                        line=dict(dash="dash", color="gray"))
        fig.add_scatter(x=c["predicted"], y=c["actual"], mode="lines+markers",
                        name="Model", line_color="#1565c0")
        fig.update_layout(height=340, xaxis_title="Predicted PD", yaxis_title="Actual default rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Average predicted {pct(test['pd_score'].mean())} vs actual "
                   f"{pct(test['target_default'].mean())}. Points near the dashed line = well calibrated.")
    with right:
        st.markdown("#### 5. Top drivers (mean |SHAP|)")
        imp = data["importance"].head(12).copy()
        imp["feature"] = imp["feature"].map(lambda f: LABELS.get(f, f))
        fig = px.bar(imp.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                     labels={"mean_abs_shap": "Average impact on risk", "feature": ""})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Collections data, IDs, dates, gender and marital status are NOT model inputs.")

    # ---------- Drift monitoring (PSI) ----------
    st.markdown("#### 6. Drift monitoring demo (PSI)")
    st.caption("Simulation: baseline = loans made 2018-2022, 'current' = loans made in 2023. "
               "In production, baseline = the training data and current = each new month of loans.")
    yr = pd.to_datetime(s["origination_date"]).dt.year
    base, curr = s[yr <= 2022], s[yr == 2023]
    rows = [("Model score (PD)", psi(base["pd_score"], curr["pd_score"]))]
    for f in ["annual_income", "years_employed", "age", "loan_to_income",
              "bur_mean_dpd", "rep_mean_dpd", "interest_rate_pct", "loan_amount"]:
        rows.append((LABELS.get(f, f), psi(base[f], curr[f])))
    p = pd.DataFrame(rows, columns=["Variable", "PSI"])
    p["Status"] = p["PSI"].map(psi_status)
    p["PSI"] = p["PSI"].round(4)
    left, right = st.columns([2, 3])
    left.dataframe(p, hide_index=True, use_container_width=True)
    right.markdown("""
**Monitoring plan after deployment**

| What | How often | Alert | Action |
|---|---|---|---|
| **PSI** of score and key inputs | Monthly | > 0.10 watch, > 0.25 alert | Investigate; retrain if > 0.25 |
| **AUC / KS** on loans whose outcome is known | Quarterly | AUC drops > 0.03 | Retrain / re-tune |
| **Calibration**: avg PD vs actual default rate | Monthly | Gap > 2 pts | Recalibrate |
| **Default rate by segment** | Monthly | Segments stop ranking in order | Review cut-offs |
| **Data quality**: missing %, new categories | Every load | Change > 5 pts | Fix pipeline first |
| **Fairness**: PD vs actual by gender / age | Quarterly | Large gap | Review features |

Every alert goes to the model owner; a human approves any retrain.
""")

    # ---------- Fairness ----------
    st.markdown("#### 7. Fairness check (gender is not a model input)")
    f = test.groupby("gender").agg(loans=("account_id", "count"),
                                   avg_predicted_pd=("pd_score", "mean"),
                                   actual_default_rate=("target_default", "mean")).reset_index()
    f["gap"] = f["avg_predicted_pd"] - f["actual_default_rate"]
    for col in ["avg_predicted_pd", "actual_default_rate", "gap"]:
        f[col] = f[col].map(lambda v: f"{v:+.1%}" if col == "gap" else pct(v))
    st.dataframe(f, hide_index=True, use_container_width=True)
    st.caption("If predicted and actual rates are close for each group, the model is not "
               "over- or under-scoring one gender.")