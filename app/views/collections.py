"""Collections strategy (C-score view): which actions and customers recover money.
Separate from the default model: collections data is an INPUT here, not a leak."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score

from views.common import money, pct

LEVEL = {"Reminder Call": 1, "SMS/Email Notice": 1, "Field Visit": 2,
         "Settlement Offer": 3, "Legal Notice": 4, "Written-Off Referral": 5}
# Illustrative cost units ONLY (not from the data). The analyst edits them.
DEFAULT_COST = {"SMS/Email Notice": 1, "Reminder Call": 5, "Settlement Offer": 20,
                "Written-Off Referral": 30, "Field Visit": 50, "Legal Notice": 100}
SEG = {"employment_type": "Employment type", "income_band": "Income band",
       "age_band": "Age band", "loan_type": "Loan type",
       "risk_segment": "Risk segment", "region": "Region"}


@st.cache_data
def load_events():
    e = pd.read_csv("data/raw/collections_events.csv")
    e["event_date"] = pd.to_datetime(e["event_date"])
    e["paid"] = (e["amount_recovered"] > 0).astype(int)
    e["level"] = e["action_type"].map(LEVEL)
    return e


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def render(data, df):
    st.subheader("💰 Collections Strategy")
    st.caption("Which actions recover money, and from which customers. "
               "Uses the current sidebar filters.")

    with st.expander("ℹ️ Why this is separate from the default model", expanded=False):
        st.markdown("""
Banks use three different scores:
- **A-score (application):** should we give this loan?
- **B-score (behaviour):** how risky is this existing loan? → *our default model*
- **C-score (collections):** this customer is late; will they pay, and what action works?

Collections data **cannot** be an input to the default model: it is the bank's
*reaction* to non-payment (leakage). But it is exactly the right data for **this**
question: how to recover money once a customer is delinquent.

**"Paid"** = `amount_recovered > 0`. Money is a fact; the outcome label is typed by staff
and, as shown below, often disagrees with the money.
""")

    e = load_events()
    e = e[e["account_id"].isin(df["account_id"])]
    if e.empty:
        st.info("No collections actions for these filters.")
        return

    # ---------- 1. Data quality ----------
    st.markdown("#### 1. Can we trust the collections data?")
    pr = e[e["outcome"] == "Payment Received"]
    nr = e[e["outcome"] == "No Response"]
    multi = e.sort_values("event_date").groupby("account_id").filter(lambda g: len(g) > 1)
    logical = multi.groupby("account_id")["level"].apply(lambda s: s.is_monotonic_increasing)
    c = st.columns(4)
    c[0].metric("Collection actions", f"{len(e):,}", help=f"{e['account_id'].nunique():,} accounts")
    c[1].metric("'Payment Received' with 0 money", pct((pr["paid"] == 0).mean()) if len(pr) else "-")
    c[2].metric("'No Response' with money", pct(nr["paid"].mean()) if len(nr) else "-")
    c[3].metric("Accounts escalating logically", pct(logical.mean()) if len(logical) else "-",
                help="Reminder → visit → settlement → legal → write-off, in date order")
    st.warning("Outcome labels do not match the money recovered, and actions do not escalate "
               "in a logical order. Treat every result below as a first look, and fix the "
               "data capture before using it for decisions.")

    # ---------- 2. Action effectiveness ----------
    st.markdown("#### 2. Which action recovers money?")
    act = e.groupby("action_type").agg(
        events=("collection_event_id", "count"), pay_rate=("paid", "mean"),
        avg_when_paid=("amount_recovered", lambda x: x[x > 0].mean()),
        total_recovered=("amount_recovered", "sum")).reset_index()
    act["ci"] = ci95(act["pay_rate"], act["events"])
    act["share_of_money"] = act["total_recovered"] / act["total_recovered"].sum()
    act = act.sort_values("pay_rate", ascending=False)
    fig = px.bar(act, x="action_type", y="pay_rate", error_y="ci",
                 text=act["pay_rate"].map(pct),
                 labels={"action_type": "", "pay_rate": "Share of actions that recovered money"},
                 title="Pay rate per action (bars = 95% confidence range)")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    best, worst = act.iloc[0], act.iloc[-1]
    se = np.sqrt(best["pay_rate"] * (1 - best["pay_rate"]) / best["events"]
                 + worst["pay_rate"] * (1 - worst["pay_rate"]) / worst["events"])
    z = (best["pay_rate"] - worst["pay_rate"]) / se if se > 0 else 0
    verdict = "a real difference" if abs(z) > 1.96 else "within noise"
    st.caption(f"Best ({best['action_type']}, {pct(best['pay_rate'])}) vs worst "
               f"({worst['action_type']}, {pct(worst['pay_rate'])}): gap is **{verdict}** "
               f"(z = {z:.2f}; above 1.96 means real). Overlapping bars mean the actions "
               f"perform about the same.")
    show = act.copy()
    show["pay_rate"] = show["pay_rate"].map(pct)
    show["ci"] = show["ci"].map(lambda v: f"±{v * 100:.1f} pts")
    show["avg_when_paid"] = show["avg_when_paid"].map(money)
    show["total_recovered"] = show["total_recovered"].map(money)
    show["share_of_money"] = show["share_of_money"].map(pct)
    st.dataframe(show, hide_index=True, use_container_width=True)

    # ---------- 3. Cost calculator ----------
    st.markdown("#### 3. Money recovered per unit of cost")
    st.caption("⚠️ The costs below are **illustrative assumptions, not data**. "
               "Type in the bank's real cost per action to rank actions by return.")
    costs = pd.DataFrame({"action_type": list(DEFAULT_COST),
                          "cost_per_action": list(DEFAULT_COST.values())})
    costs = st.data_editor(costs, hide_index=True, use_container_width=True,
                           disabled=["action_type"], key="coll_costs")
    roi = act.merge(costs, on="action_type")
    roi["total_cost"] = roi["events"] * roi["cost_per_action"]
    roi["recovered_per_cost"] = np.where(roi["total_cost"] > 0,
                                         roi["total_recovered"] / roi["total_cost"], np.nan)
    roi = roi.sort_values("recovered_per_cost", ascending=False)
    fig = px.bar(roi, x="action_type", y="recovered_per_cost",
                 text=roi["recovered_per_cost"].round(1),
                 labels={"action_type": "", "recovered_per_cost": "Money recovered per cost unit"},
                 title="Return per action (with the costs above)")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- 4. Which customers pay ----------
    st.markdown("#### 4. Which customers pay when in collections?")
    a = e.groupby("account_id").agg(
        events=("collection_event_id", "count"), any_paid=("paid", "max"),
        total_recovered=("amount_recovered", "sum")).reset_index()
    a = a.merge(data["scored"], on="account_id", how="left")
    dim = st.selectbox("Group by", list(SEG), format_func=SEG.get, key="coll_dim")
    g = a.groupby(dim, observed=True).agg(accounts=("account_id", "count"),
                                          pay_rate=("any_paid", "mean"),
                                          avg_recovered=("total_recovered", "mean")).reset_index()
    g["ci"] = ci95(g["pay_rate"], g["accounts"])
    fig = px.bar(g, x=dim, y="pay_rate", error_y="ci", text=g["pay_rate"].map(pct),
                 labels={dim: SEG[dim], "pay_rate": "Accounts that paid something"})
    fig.update_layout(height=320, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(a):,} accounts in collections; {pct(a['any_paid'].mean())} paid something; "
               f"total recovered {money(a['total_recovered'].sum())}.")

    # ---------- 5. Does the default score help? ----------
    st.markdown("#### 5. Does our default risk score predict who pays?")
    left, right = st.columns(2)
    if a["any_paid"].nunique() == 2:
        auc = roc_auc_score(a["any_paid"], -a["pd_score"])
        left.metric("AUC (low PD → pays)", f"{auc:.3f}", help="0.5 = no link at all")
    cure = a.groupby("any_paid")["target_default"].mean()
    if len(cure) == 2:
        right.metric("Default rate: paid in collections vs not",
                     f"{pct(cure.get(1, np.nan))} vs {pct(cure.get(0, np.nan))}")
    st.caption("If the AUC is near 0.5, the default model cannot rank who will pay. "
               "A separate collections score (C-score) is needed.")

    # ---------- 6. Recommendation ----------
    st.markdown("#### 6. Recommended strategy")
    st.markdown(f"""
1. **Least-cost escalation.** Actions recover money at similar rates
   ({pct(act['pay_rate'].min())}-{pct(act['pay_rate'].max())}), so start with the cheapest
   actions and escalate only when they fail.
2. **Prefer settlement offers over SMS-only for larger balances.** It is the clearest
   difference in the data, and settlement offers bring the largest share of money
   ({pct(act.set_index('action_type').loc['Settlement Offer', 'share_of_money'])}).
3. **Build a separate collections score (C-score).** The default score does not
   predict who pays.
4. **Fix data capture first.** Record outcomes from the payment system, not typed by staff,
   and log the real order of actions.
5. **Run champion/challenger tests.** Randomly assign similar customers to different
   actions, then compare. This is the only reliable way to learn what works.

**Human-in-the-loop:** these are suggestions for the collections team; staff choose the
final action for each customer.
""")