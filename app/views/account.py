"""Account page: everything an analyst needs to judge one loan, in plain language."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import DATA, SEG_COLORS, money, pct

RAW = "data/raw/"
CATS = ["loan_type", "employment_type", "region", "city_tier", "housing_type",
        "education_level", "disbursal_channel", "loan_purpose", "occupation_sector"]
LABELS = {
    "annual_income": "Yearly income", "years_employed": "Years in current job",
    "age": "Age", "rep_mean_dpd": "Average days late on this loan",
    "rep_max_dpd": "Worst delay on this loan (days)", "rep_late": "Months paid late",
    "rep_missed": "Months not paid", "rep_pay_ratio": "Share of payments made",
    "rep_dpd_trend": "Lateness getting worse (last 3 vs first 3 months)",
    "rep_dpd_first3": "Average days late, first 3 months",
    "rep_dpd_last3": "Average days late, last 3 months",
    "rep_missed_last3": "Months not paid, last 3 months",
    "rep_ever_30dpd": "Ever 30+ days late", "rep_ever_60dpd": "Ever 60+ days late",
    "rep_last_balance": "Amount still owed",
    "bur_mean_dpd": "Average days late at other banks",
    "bur_max_dpd": "Worst delay at other banks (days)",
    "bur_writeoffs": "Loans written off at other banks",
    "bur_active": "Open loans at other banks", "bur_lines": "Loans at other banks",
    "bur_utilization": "Share of credit limits used at other banks",
    "bur_sanctioned": "Credit limits at other banks",
    "bur_outstanding": "Amount owed to other banks",
    "bur_no_history": "No history at other banks", "loan_to_income": "Loan size vs income",
    "interest_rate_pct": "Interest rate", "loan_amount": "Loan amount",
    "tenure_months": "Loan length (months)", "years_at_residence": "Years at current address",
    "num_dependents": "Dependents", "customer_loans": "Loans with us",
    "bur_nbfc_fintech_share": "Share of loans from non-bank lenders",
}
CAT_LABELS = {"loan_type": "Loan type", "employment_type": "Job type", "region": "Region",
              "city_tier": "City tier", "housing_type": "Housing", "education_level": "Education",
              "disbursal_channel": "Sales channel", "loan_purpose": "Loan purpose",
              "occupation_sector": "Work sector"}
GROUP_TEXT = {"Low": "under 10% chance of going bad", "Medium": "10% to 20% chance",
              "High": "20% to 35% chance", "Very High": "over 35% chance"}


@st.cache_data
def load_shap():
    return pd.read_csv(DATA / "shap_values.csv").set_index("account_id")


@st.cache_data
def load_raw():
    return (pd.read_csv(RAW + "repayment_history.csv"),
            pd.read_csv(RAW + "bureau_credit.csv"),
            pd.read_csv(RAW + "collections_events.csv"))


def feature_label(f, row):
    """Plain-English name and this customer's value for a model input."""
    for c in CATS:
        if f.startswith(c + "_"):
            return f"{CAT_LABELS[c]}: {f[len(c) + 1:]}", str(row[c])
    v = row.get(f, "")
    if isinstance(v, float):
        if f in ("bur_utilization", "rep_pay_ratio", "bur_nbfc_fintech_share"):
            v = pct(v)
        elif f in ("annual_income", "loan_amount", "rep_last_balance",
                   "bur_sanctioned", "bur_outstanding"):
            v = money(v)
        elif f == "interest_rate_pct":
            v = f"{v:.2f}%"
        elif f == "loan_to_income":
            v = f"{v:.1f}x income"
        elif f in ("rep_ever_30dpd", "rep_ever_60dpd", "bur_no_history"):
            v = "yes" if v >= 1 else "no"
        else:
            v = f"{v:,.1f}".rstrip("0").rstrip(".")
    return LABELS.get(f, f.replace("_", " ")), str(v)


def next_step(row, flagged):
    seg = str(row["risk_segment"])
    if row["target_default"] == 1:
        return "This loan has already gone bad. Follow the collections process."
    if seg == "Very High" or (flagged and seg == "High"):
        return "Contact the customer soon and review the loan."
    if seg == "High" or flagged:
        return "Keep a close watch and review it at the next check."
    return "No action needed. Normal monitoring."


def render(data, df):
    st.subheader("Account details")
    if df.empty:
        st.warning("No loans match these filters.")
        return

    ids = df.sort_values("pd_score", ascending=False)["account_id"].tolist()
    aid = st.selectbox("Choose an account (type to search; riskiest are listed first)",
                       ids, key="acc_id")
    row = data["scored"].set_index("account_id").loc[aid]
    rep, bur, coll = load_raw()
    seg = str(row["risk_segment"])
    an = data["anomalies"]
    hit = an[an["account_id"] == aid]

    # ---------- Why this loan is risky (computed first, used in the summary) ----------
    s = load_shap().loc[aid]
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(8)
    names, vals = zip(*[feature_label(f, row) for f in top.index])
    d = pd.DataFrame({"reason": names, "value": vals, "impact": top.values})
    ups = d[d["impact"] > 0].head(3)
    reasons = "; ".join(f"{r.reason.lower()} ({r.value})" for r in ups.itertuples()) \
        or "nothing stands out"

    # ---------- Header + summary ----------
    st.markdown(f"### {aid} · {row['loan_type']} · "
                f"<span style='color:{SEG_COLORS[seg]}'>{seg} risk</span>",
                unsafe_allow_html=True)
    box = st.error if seg in ("High", "Very High") else (st.warning if seg == "Medium" else st.success)
    box(f"**{seg} risk** ({GROUP_TEXT[seg]}). Main reasons: {reasons}.  \n"
        f"**Suggested next step:** {next_step(row, len(hit) > 0)}")

    c = st.columns(5)
    c[0].metric("Predicted risk", pct(row["pd_score"]),
                help="The model's estimate of the chance this loan goes bad")
    c[1].metric("Second opinion", pct(row["pd_score_lr"]),
                help="Estimate from a simpler model. If the two are far apart, look closer")
    c[2].metric("Loan amount", money(row["loan_amount"]))
    c[3].metric("Interest rate", f"{row['interest_rate_pct']:.2f}%")
    c[4].metric("Status", "Went bad" if row["target_default"] == 1 else "Paying")

    # ---------- Reasons chart ----------
    st.markdown("#### Why the model gives this score")
    d["direction"] = d["impact"].map(lambda v: "Pushes risk up" if v > 0 else "Pushes risk down")
    d["label"] = d["reason"] + " (" + d["value"] + ")"
    fig = px.bar(d.iloc[::-1], x="impact", y="label", orientation="h", color="direction",
                 color_discrete_map={"Pushes risk up": "#c62828", "Pushes risk down": "#2e7d32"},
                 labels={"impact": "How much it moves the risk score", "label": ""})
    fig.update_layout(height=360, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars make this loan riskier, green bars make it safer. Longer bars matter "
               "more. The value in brackets is this customer's own figure.")

    # ---------- Warning signs ----------
    if len(hit):
        st.markdown("#### ⚠️ Warning signs")
        from views.anomalies import plain_reason
        for reason in plain_reason(hit["reasons"].iloc[0]).split(" | "):
            st.markdown(f"- {reason}")
        st.caption("Review and record a decision in the Anomaly Review tab.")
    else:
        st.success("No unusual behaviour spotted on this account.")

    # ---------- Payment history ----------
    st.markdown("#### Payment history (last 12 months)")
    r = rep[rep["account_id"] == aid].sort_values("payment_month")
    p = st.columns(4)
    p[0].metric("Months paid on time", int((r["payment_status"] == "On-Time").sum()))
    p[1].metric("Months paid late", int((r["payment_status"] == "Late").sum()))
    p[2].metric("Months not paid", int((r["payment_status"] == "Missed").sum()))
    p[3].metric("Worst delay", f"{int(r['days_past_due'].max())} days")
    fig = go.Figure()
    fig.add_bar(x=r["payment_month"], y=r["emi_due"], name="Payment due", marker_color="#b0bec5")
    fig.add_bar(x=r["payment_month"], y=r["emi_paid"], name="Payment made", marker_color="#1565c0")
    fig.add_scatter(x=r["payment_month"], y=r["days_past_due"], name="Days late",
                    yaxis="y2", mode="lines+markers", line_color="#c62828")
    fig.update_layout(barmode="group", height=340, xaxis_title="Month of the loan",
                      yaxis_title="Amount", xaxis=dict(dtick=1),
                      yaxis2=dict(title="Days late", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey = what was due, blue = what was paid. The red line shows how many days "
               "late each payment was.")

    # ---------- Customer + other banks ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### About the customer")
        prof = {"Customer ID": row["customer_id"], "Age": int(row["age"]),
                "Job type": row["employment_type"],
                "Years in current job": f"{row['years_employed']:.1f}",
                "Yearly income": money(row["annual_income"]),
                "Income group": str(row["income_band"]),
                "Loan size vs income": f"{row['loan_to_income']:.1f}x",
                "Region": row["region"], "City tier": row["city_tier"],
                "Housing": row["housing_type"], "Loans with us": int(row["customer_loans"])}
        st.table(pd.DataFrame(prof.items(), columns=["", "Value"]).set_index(""))
    with right:
        st.markdown("#### Loans at other banks")
        b = bur[bur["customer_id"] == row["customer_id"]]
        if b.empty:
            st.info("This customer has no loans at other banks on record.")
        else:
            bt = pd.DataFrame({
                "Type": b["credit_type"], "Lender": b["lender_category"],
                "Limit": b["sanctioned_amount"].map(money),
                "Still owed": b["outstanding_amount"].map(money),
                "Status": b["account_status"],
                "Days late": b["days_past_due"].map(lambda v: "-" if pd.isna(v) else int(v)),
            })
            st.dataframe(bt, hide_index=True, use_container_width=True)
            wo = int((b["account_status"] == "Written-Off").sum())
            st.caption(f"{len(b)} loans elsewhere, {wo} written off. "
                       "Written off = the other bank gave up trying to collect.")

    # ---------- Other loans + collections ----------
    left, right = st.columns(2)
    with left:
        st.markdown("#### This customer's other loans with us")
        o = data["scored"][(data["scored"]["customer_id"] == row["customer_id"])
                           & (data["scored"]["account_id"] != aid)]
        if o.empty:
            st.caption("No other loans.")
        else:
            v = pd.DataFrame({"Account": o["account_id"], "Loan type": o["loan_type"],
                              "Amount": o["loan_amount"].map(money),
                              "Predicted risk": o["pd_score"].map(pct),
                              "Risk group": o["risk_segment"].astype(str)})
            st.dataframe(v, hide_index=True, use_container_width=True)
            st.caption("If one of a customer's loans goes bad, their other loans are riskier too.")
    with right:
        st.markdown("#### Collection actions on this loan")
        e = coll[coll["account_id"] == aid].sort_values("event_date")
        if e.empty:
            st.caption("No collection actions.")
        else:
            et = pd.DataFrame({"Date": e["event_date"], "Action": e["action_type"],
                               "Result": e["outcome"],
                               "Money recovered": e["amount_recovered"].map(money)})
            st.dataframe(et, hide_index=True, use_container_width=True)
        st.caption("Shown for context only. Collection actions are not used to work out the "
                   "risk score, because they only start after a customer stops paying.")