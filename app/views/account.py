"""Account view: one loan's risk score, key drivers (SHAP), history and flags."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import DATA, SEG_COLORS, money, pct

RAW = "data/raw/"
CATS = ["loan_type", "employment_type", "region", "city_tier", "housing_type",
        "education_level", "disbursal_channel", "loan_purpose", "occupation_sector"]
LABELS = {
    "annual_income": "Annual income", "years_employed": "Years employed",
    "age": "Age", "rep_mean_dpd": "Avg days past due (our loan)",
    "rep_max_dpd": "Max days past due (our loan)", "rep_late": "Late months (our loan)",
    "rep_missed": "Missed months (our loan)", "rep_pay_ratio": "Share of EMI paid",
    "rep_dpd_trend": "DPD trend (last 3 vs first 3 months)",
    "rep_missed_last3": "Missed in last 3 months", "rep_last_balance": "Current balance",
    "bur_mean_dpd": "Avg days past due (other lenders)",
    "bur_max_dpd": "Max days past due (other lenders)",
    "bur_writeoffs": "Written-off lines (other lenders)",
    "bur_active": "Active lines (other lenders)", "bur_lines": "Bureau lines",
    "bur_utilization": "Credit utilization (other lenders)",
    "bur_sanctioned": "Credit limit (other lenders)",
    "bur_outstanding": "Balance owed (other lenders)",
    "bur_no_history": "No bureau history", "loan_to_income": "Loan-to-income",
    "interest_rate_pct": "Interest rate %", "loan_amount": "Loan amount",
    "tenure_months": "Tenure (months)", "years_at_residence": "Years at residence",
    "num_dependents": "Dependents", "customer_loans": "Loans with us",
}


@st.cache_data
def load_shap():
    return pd.read_csv(DATA / "shap_values.csv").set_index("account_id")


@st.cache_data
def load_raw():
    return (pd.read_csv(RAW + "repayment_history.csv"),
            pd.read_csv(RAW + "bureau_credit.csv"),
            pd.read_csv(RAW + "collections_events.csv"))


def feature_label(f, row):
    """Plain-English name and this account's value for a model feature."""
    for c in CATS:
        if f.startswith(c + "_"):
            return f"{c.replace('_', ' ').capitalize()} = {f[len(c) + 1:]}", str(row[c])
    v = row.get(f, "")
    if isinstance(v, float):
        if f in ("bur_utilization", "rep_pay_ratio"):
            v = pct(v)
        elif f in ("annual_income", "loan_amount", "rep_last_balance",
                   "bur_sanctioned", "bur_outstanding"):
            v = money(v)
        else:
            v = f"{v:,.2f}".rstrip("0").rstrip(".")
    return LABELS.get(f, f), str(v)


def render(data, df):
    st.subheader("Account view")
    if df.empty:
        st.warning("No loans match the filters.")
        return

    ids = df.sort_values("pd_score", ascending=False)["account_id"].tolist()
    aid = st.selectbox("Account (type to search, sorted highest risk first)", ids,
                       key="acc_id")
    row = data["scored"].set_index("account_id").loc[aid]
    rep, bur, coll = load_raw()

    # ---------- Header ----------
    seg = str(row["risk_segment"])
    st.markdown(f"### {aid} · {row['loan_type']} · "
                f"<span style='color:{SEG_COLORS[seg]}'>{seg} risk</span>",
                unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Model PD", pct(row["pd_score"]),
                help="XGBoost probability of default")
    c[1].metric("Logistic PD", pct(row["pd_score_lr"]), help="Baseline model, for comparison")
    c[2].metric("Loan amount", money(row["loan_amount"]))
    c[3].metric("Interest rate", f"{row['interest_rate_pct']:.2f}%")
    c[4].metric("Observed outcome", "Defaulted" if row["target_default"] == 1 else "Performing")

    # ---------- Key drivers ----------
    st.markdown("#### Key drivers of this score")
    s = load_shap().loc[aid]
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(8)
    names, vals = zip(*[feature_label(f, row) for f in top.index])
    d = pd.DataFrame({"driver": names, "value": vals, "impact": top.values})
    d["direction"] = d["impact"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    d["label"] = d["driver"] + " (" + d["value"] + ")"
    fig = px.bar(d.iloc[::-1], x="impact", y="label", orientation="h", color="direction",
                 color_discrete_map={"Raises risk": "#c62828", "Lowers risk": "#2e7d32"},
                 labels={"impact": "Impact on risk (SHAP, log-odds)", "label": ""})
    fig.update_layout(height=360, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars push this account's risk up; green bars pull it down. "
               "Values in brackets are this customer's actual figures.")

    # ---------- Anomaly flags ----------
    an = data["anomalies"]
    hit = an[an["account_id"] == aid]
    if len(hit):
        st.error("🚩 **Anomaly flags:** " + hit["reasons"].iloc[0].replace(" | ", " · "))
    else:
        st.success("No anomaly flags on this account.")

    # ---------- Repayment history ----------
    st.markdown("#### Repayment history (12 months)")
    r = rep[rep["account_id"] == aid].sort_values("payment_month")
    fig = go.Figure()
    fig.add_bar(x=r["payment_month"], y=r["emi_due"], name="EMI due", marker_color="#b0bec5")
    fig.add_bar(x=r["payment_month"], y=r["emi_paid"], name="EMI paid", marker_color="#1565c0")
    fig.add_scatter(x=r["payment_month"], y=r["days_past_due"], name="Days past due",
                    yaxis="y2", mode="lines+markers", line_color="#c62828")
    fig.update_layout(barmode="group", height=340, xaxis_title="Month since origination",
                      yaxis_title="Amount", yaxis2=dict(title="Days past due",
                                                        overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Customer + bureau ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### Customer profile")
        prof = {"Customer ID": row["customer_id"], "Age": int(row["age"]),
                "Employment": row["employment_type"],
                "Years employed": f"{row['years_employed']:.1f}",
                "Annual income": money(row["annual_income"]),
                "Income band": str(row["income_band"]), "Region": row["region"],
                "City tier": row["city_tier"], "Housing": row["housing_type"],
                "Loans with us": int(row["customer_loans"])}
        st.table(pd.DataFrame(prof.items(), columns=["Field", "Value"]).set_index("Field"))
    with right:
        st.markdown("#### Credit bureau (other lenders)")
        b = bur[bur["customer_id"] == row["customer_id"]]
        if b.empty:
            st.info("No bureau history for this customer.")
        else:
            st.dataframe(b[["credit_type", "lender_category", "sanctioned_amount",
                            "outstanding_amount", "account_status", "days_past_due"]],
                         hide_index=True, use_container_width=True)

    # ---------- Other loans + collections ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### Other loans of this customer")
        o = data["scored"][(data["scored"]["customer_id"] == row["customer_id"])
                           & (data["scored"]["account_id"] != aid)]
        if o.empty:
            st.caption("No other loans.")
        else:
            v = o[["account_id", "loan_type", "loan_amount", "pd_score", "risk_segment"]].copy()
            v["loan_amount"] = v["loan_amount"].map(money)
            v["pd_score"] = v["pd_score"].map(pct)
            st.dataframe(v, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Collections actions")
        e = coll[coll["account_id"] == aid].sort_values("event_date")
        if e.empty:
            st.caption("No collections actions.")
        else:
            st.dataframe(e[["event_date", "action_type", "outcome", "amount_recovered"]],
                         hide_index=True, use_container_width=True)
        st.caption("Shown for context only. Collections data is NOT used in the model (leakage).")