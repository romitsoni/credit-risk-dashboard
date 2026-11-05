"""Model check: how good the model is, what drives it, drift checks and fairness."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import SEG_COLORS, pct

LABELS = {"annual_income": "Annual income", "years_employed": "Years in job",
          "rep_mean_dpd": "Average days late on this loan", "age": "Age",
          "rep_late": "Months paid late on this loan",
          "bur_mean_dpd": "Average days late at other banks",
          "loan_to_income": "Loan size compared to income",
          "rep_max_dpd": "Longest delay on this loan", "interest_rate_pct": "Interest rate",
          "bur_active": "Open loans at other banks", "bur_writeoffs": "Loans written off at other banks",
          "years_at_residence": "Years at current address",
          "bur_max_dpd": "Longest delay at other banks",
          "loan_amount": "Loan amount", "rep_last_balance": "Amount still owed"}
GENDER = {"M": "Male", "F": "Female"}


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
    if v < 0.10:
        return "🟢 No real change"
    return "🟡 Some change: watch" if v < 0.25 else "🔴 Big change: retrain"


def render(data, df):
    st.subheader("📈 Model check")
    st.caption("How well the risk model works, what it looks at, and how we keep checking it.")
    s = data["scored"]
    test = s[s["in_test_set"] == 1]
    m = data["metrics"]
    xgb, lr = m["xgboost"], m["logistic_regression"]

    # Lift numbers, used in the headline and in section 2
    d = test.assign(group=pd.qcut(test["pd_score"].rank(method="first"), 10,
                                  labels=list(range(10, 0, -1))))
    t = d.groupby("group", observed=True).agg(
        loans=("account_id", "count"), bad=("target_default", "sum"),
        bad_rate=("target_default", "mean")).reset_index()
    t["group"] = t["group"].astype(int)
    t = t.sort_values("group")
    t["caught"] = t["bad"].cumsum() / t["bad"].sum()
    top2 = t.loc[t["group"] <= 2, "caught"].max()

    st.info(f"The model puts **{pct(top2)} of all bad loans** into the riskiest 20% of "
            f"customers. Guessing would catch only 20%. It was tested on "
            f"{m['test_rows']:,} loans from customers it had never seen.")

    with st.expander("ℹ️ How to read this page"):
        st.markdown("""
- **The model** gives each loan a *predicted risk*: the chance it will go bad.
- **The test** was done on customers the model never saw while it was learning. This is a fair exam, not a memory test.
- **No model is perfect.** The aim is to sort loans so the risky ones come first and the team looks at them early.
- **Monitoring** means we keep checking the model every month. If customers or data change, we fix or retrain it.
""")

    # ---------- 1. How good ----------
    st.markdown("#### 1. How good is the model?")
    pairs = round(xgb["AUC"] * 100)
    perf = pd.DataFrame({
        "Model": ["Simple model (logistic regression, our baseline)",
                  "Main model (XGBoost, used in this dashboard)"],
        "AUC": [lr["AUC"], xgb["AUC"]],
        "KS": [lr["KS"], xgb["KS"]],
        "Gini": [lr["Gini"], xgb["Gini"]]})
    st.dataframe(perf, hide_index=True, use_container_width=True)
    st.markdown(f"""
- **AUC ({xgb['AUC']:.2f}):** take one bad loan and one good loan. In **{pairs} out of 100**
  pairs the model gives the bad loan the higher risk. 50 would be pure guessing.
- **KS ({xgb['KS']:.2f}):** how clearly the model separates good loans from bad ones.
  0 means no separation and 1 means perfect.
- **Gini ({xgb['Gini']:.2f}):** the same idea as AUC on a 0 to 1 scale (0 = guessing, 1 = perfect).
  Banks often use this number.
""")
    st.caption("Both models score about the same, so the patterns are mostly simple. "
               "The simple model is easier to explain, and the main model is kept for the "
               "per-loan reasons it gives.")

    # ---------- 2 + 3. Lift + risk groups ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### 2. Does the model find the bad loans?")
        fig = go.Figure()
        fig.add_bar(x=t["group"], y=t["bad_rate"], name="How often loans go bad",
                    text=t["bad_rate"].map(pct), marker_color="#1565c0")
        fig.add_scatter(x=t["group"], y=t["caught"], name="Share of all bad loans caught so far",
                        mode="lines+markers", line_color="#c62828")
        fig.update_layout(height=340, yaxis_tickformat=".0%",
                          xaxis_title="Group of 10% (1 = riskiest, 10 = safest)",
                          xaxis=dict(dtick=1), legend=dict(orientation="h", y=1.18))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Loans are sorted by predicted risk and split into 10 equal groups. "
                   f"The riskiest 2 groups (20% of loans) hold **{pct(top2)}** of all bad loans.")
    with right:
        st.markdown("#### 3. Do the risk groups work?")
        seg = test.groupby("risk_segment", observed=True).agg(
            loans=("account_id", "count"), bad_rate=("target_default", "mean")).reset_index()
        fig = px.bar(seg, x="risk_segment", y="bad_rate", color="risk_segment",
                     color_discrete_map=SEG_COLORS, text=seg["bad_rate"].map(pct),
                     labels={"risk_segment": "Risk group", "bad_rate": "How often loans went bad"})
        fig.update_layout(height=340, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Test customers only. Each higher group goes bad more often, so the "
                   "groups are in the right order.")

    # ---------- 4 + 5. Accuracy + drivers ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### 4. Is the predicted risk accurate?")
        c = test.assign(bin=pd.qcut(test["pd_score"], 10, duplicates="drop"))
        c = c.groupby("bin", observed=True).agg(predicted=("pd_score", "mean"),
                                                actual=("target_default", "mean")).reset_index()
        mx = max(c["predicted"].max(), c["actual"].max()) * 1.05
        fig = go.Figure()
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect match",
                        line=dict(dash="dash", color="gray"))
        fig.add_scatter(x=c["predicted"], y=c["actual"], mode="lines+markers",
                        name="Our model", line_color="#1565c0")
        fig.update_layout(height=340, xaxis_title="Predicted risk",
                          yaxis_title="How often loans really went bad",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"When the model says 20%, about 20% should go bad. On average it predicted "
                   f"**{pct(test['pd_score'].mean())}** and **{pct(test['target_default'].mean())}** "
                   f"really went bad. Points close to the dashed line mean the risk numbers can "
                   f"be trusted (this is called calibration).")
    with right:
        st.markdown("#### 5. What drives the score")
        imp = data["importance"].head(12).copy()
        imp["feature"] = imp["feature"].map(lambda f: LABELS.get(f, f))
        fig = px.bar(imp.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                     labels={"mean_abs_shap": "How much it moves the risk score (average)",
                             "feature": ""})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Longer bars matter more (measured with SHAP). The model does **not** use "
                   "gender, marital status, IDs, dates or collection actions.")

    # ---------- 6. Drift ----------
    st.markdown("#### 6. Has anything changed? (drift check)")
    st.caption("We compare older loans (2018 to 2022) with newer loans (2023). If customers "
               "start to look very different, the model may need retraining. In live use, "
               "this check runs every month on new loans.")
    yr = pd.to_datetime(s["origination_date"]).dt.year
    base, curr = s[yr <= 2022], s[yr == 2023]
    rows = [("Predicted risk score", psi(base["pd_score"], curr["pd_score"]))]
    for f in ["annual_income", "years_employed", "age", "loan_to_income",
              "bur_mean_dpd", "rep_mean_dpd", "interest_rate_pct", "loan_amount"]:
        rows.append((LABELS.get(f, f), psi(base[f], curr[f])))
    p = pd.DataFrame(rows, columns=["What we checked", "Change score (PSI)"])
    p["Result"] = p["Change score (PSI)"].map(psi_status)
    p["Change score (PSI)"] = p["Change score (PSI)"].round(3)
    changed = (p["Result"] != "🟢 No real change").sum()
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(p, hide_index=True, use_container_width=True)
        st.caption("Below 0.10 = no real change. 0.10 to 0.25 = some change, watch it. "
                   "Above 0.25 = big change, retrain.")
        if changed == 0:
            st.success("Nothing has changed much. The model is still safe to use.")
        else:
            st.warning(f"{changed} item(s) have changed. Look into these first.")
    with right:
        st.markdown("""
**Our checking plan after go-live**

| What we check | How often | Alarm when | What we do |
|---|---|---|---|
| Have customers changed? (PSI on the score and key inputs) | Monthly | Above 0.10 = watch, above 0.25 = alarm | Find the cause; retrain if above 0.25 |
| Does it still sort loans well? (AUC, KS) | Every 3 months | AUC drops by more than 0.03 | Retrain the model |
| Is predicted risk still accurate? | Monthly | Predicted and real differ by more than 2 points | Re-adjust the scores |
| Are risk groups still in order? | Monthly | A lower group goes bad more than a higher one | Review the group limits |
| Is the data clean? | Every data load | Missing values or new categories move by more than 5 points | Fix the data first |
| Is it fair? | Every 3 months | Big gap for one group (gender, age) | Review the inputs |

Every alarm goes to the model owner. **A person approves every retrain**, so nothing changes automatically.
""")

    # ---------- 7. Fairness ----------
    st.markdown("#### 7. Is the model fair?")
    st.caption("Gender is not given to the model. We still check that it does not score one "
               "group too high or too low.")
    f = test.groupby("gender").agg(loans=("account_id", "count"),
                                   predicted=("pd_score", "mean"),
                                   actual=("target_default", "mean")).reset_index()
    f["gap"] = f["predicted"] - f["actual"]
    show = pd.DataFrame({
        "Group": f["gender"].map(lambda g: GENDER.get(g, g)),
        "Loans": f["loans"].map("{:,}".format),
        "Predicted risk (average)": f["predicted"].map(pct),
        "How often loans really went bad": f["actual"].map(pct),
        "Difference": f["gap"].map(lambda v: f"{v * 100:+.1f} points"),
    })
    st.dataframe(show, hide_index=True, use_container_width=True)
    worst = f["gap"].abs().max()
    if worst <= 0.02:
        st.success(f"The difference is small for every group (at most {worst * 100:.1f} points), "
                   f"so the model treats men and women the same way.")
    else:
        st.warning(f"One group has a difference of {worst * 100:.1f} points. "
                   f"The model owner should review this.")