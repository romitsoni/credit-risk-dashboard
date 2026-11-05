"""Collections: which actions bring money back from late customers, and from whom.
Separate from the risk model: collection actions happen AFTER a customer is late."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score

from views.common import money, pct

LEVEL = {"Reminder Call": 1, "SMS/Email Notice": 1, "Field Visit": 2,
         "Settlement Offer": 3, "Legal Notice": 4, "Written-Off Referral": 5}
# Example costs ONLY (not from the data). The analyst can type the real ones.
DEFAULT_COST = {"SMS/Email Notice": 1, "Reminder Call": 5, "Settlement Offer": 20,
                "Written-Off Referral": 30, "Field Visit": 50, "Legal Notice": 100}
SEG = {"employment_type": "Type of job", "income_band": "Income group",
       "age_band": "Age group", "loan_type": "Loan type",
       "risk_segment": "Risk group", "region": "Region"}


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
    st.subheader("💰 Getting money back from late customers")
    st.caption("Which collection actions bring money back, and which customers pay. "
               "Uses the filters on the left.")

    with st.expander("ℹ️ Why this page is separate from the risk model"):
        st.markdown("""
- The **risk model** answers: *will this loan go bad?* It must not use collection actions,
  because those only happen **after** a customer is late. Using them would be cheating.
- **This page** answers a different question: *the customer is already late, so what action
  gets the money back?* For this question, collection data is exactly the right data.
- **"Paid"** here means money actually came in (amount recovered above 0). We trust the
  money, not the outcome typed in by staff, because the two often disagree (see below).
""")

    e = load_events()
    e = e[e["account_id"].isin(df["account_id"])]
    if e.empty:
        st.info("No collection actions for these filters.")
        return

    # ---------- 1. Data quality ----------
    st.markdown("#### 1. Can we trust this data?")
    pr = e[e["outcome"] == "Payment Received"]
    nr = e[e["outcome"] == "No Response"]
    multi = e.sort_values("event_date").groupby("account_id").filter(lambda g: len(g) > 1)
    logical = multi.groupby("account_id")["level"].apply(lambda s: s.is_monotonic_increasing)
    c = st.columns(4)
    c[0].metric("Collection actions", f"{len(e):,}",
                help=f"On {e['account_id'].nunique():,} accounts")
    c[1].metric('Marked "paid" but no money came in',
                pct((pr["paid"] == 0).mean()) if len(pr) else "-")
    c[2].metric('Marked "no response" but money came in',
                pct(nr["paid"].mean()) if len(nr) else "-")
    c[3].metric("Actions in a sensible order", pct(logical.mean()) if len(logical) else "-",
                help="Reminder, then visit, then settlement, then legal, then write-off, by date")
    st.warning("The outcomes typed by staff often do not match the money, and actions are "
               "often not in a sensible order. Treat the results below as a first look. "
               "Fixing how this data is recorded comes first.")

    # ---------- 2. Which action works ----------
    st.markdown("#### 2. Which action brings in money?")
    act = e.groupby("action_type").agg(
        events=("collection_event_id", "count"), pay_rate=("paid", "mean"),
        avg_when_paid=("amount_recovered", lambda x: x[x > 0].mean()),
        total_recovered=("amount_recovered", "sum")).reset_index()
    act["ci"] = ci95(act["pay_rate"], act["events"])
    act["share_of_money"] = act["total_recovered"] / act["total_recovered"].sum()
    act = act.sort_values("pay_rate", ascending=False)
    fig = px.bar(act, x="action_type", y="pay_rate", error_y="ci",
                 text=act["pay_rate"].map(pct),
                 labels={"action_type": "", "pay_rate": "Actions that brought in money"},
                 title="How often each action brings in money")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    best, worst = act.iloc[0], act.iloc[-1]
    se = np.sqrt(best["pay_rate"] * (1 - best["pay_rate"]) / best["events"]
                 + worst["pay_rate"] * (1 - worst["pay_rate"]) / worst["events"])
    z = (best["pay_rate"] - worst["pay_rate"]) / se if se > 0 else 0
    verdict = ("**The gap is real**, not luck" if abs(z) > 1.96
               else "**The gap could be chance**, so the actions work about the same")
    st.caption(f"Best: {best['action_type']} ({pct(best['pay_rate'])}). Worst: "
               f"{worst['action_type']} ({pct(worst['pay_rate'])}). {verdict}. "
               f"The thin line on each bar shows the range the true value is likely in; "
               f"if the lines overlap, the actions perform about the same. "
               f"(Test score z = {z:.2f}; above 1.96 means a real gap.)")
    show = pd.DataFrame({
        "Action": act["action_type"],
        "Times used": act["events"].map("{:,}".format),
        "Brought in money": act["pay_rate"].map(pct),
        "Possible range": act["ci"].map(lambda v: f"± {v * 100:.1f} points"),
        "Average amount when paid": act["avg_when_paid"].map(money),
        "Total money back": act["total_recovered"].map(money),
        "Share of all money back": act["share_of_money"].map(pct),
    })
    st.dataframe(show, hide_index=True, use_container_width=True)

    # ---------- 3. Cost ----------
    st.markdown("#### 3. Is each action worth its cost?")
    st.caption("⚠️ These costs are **examples, not real data**. Type in the bank's real cost "
               "of each action and the chart updates.")
    costs = pd.DataFrame({"action_type": list(DEFAULT_COST),
                          "cost_per_action": list(DEFAULT_COST.values())})
    costs = st.data_editor(costs, hide_index=True, use_container_width=True,
                           disabled=["action_type"], key="coll_costs",
                           column_config={"action_type": "Action",
                                          "cost_per_action": "Cost of one action"})
    roi = act.merge(costs, on="action_type")
    roi["total_cost"] = roi["events"] * roi["cost_per_action"]
    roi["recovered_per_cost"] = np.where(roi["total_cost"] > 0,
                                         roi["total_recovered"] / roi["total_cost"], np.nan)
    roi = roi.sort_values("recovered_per_cost", ascending=False)
    fig = px.bar(roi, x="action_type", y="recovered_per_cost",
                 text=roi["recovered_per_cost"].round(1),
                 labels={"action_type": "", "recovered_per_cost": "Money back for every 1 spent"},
                 title="Money back for every 1 spent (using the costs above)")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    if len(roi):
        st.caption(f"With these costs, **{roi.iloc[0]['action_type']}** gives the most money "
                   f"back for what it costs.")

    # ---------- 4. Which customers pay ----------
    st.markdown("#### 4. Which late customers pay?")
    a = e.groupby("account_id").agg(
        events=("collection_event_id", "count"), any_paid=("paid", "max"),
        total_recovered=("amount_recovered", "sum")).reset_index()
    a = a.merge(data["scored"], on="account_id", how="left")
    dim = st.selectbox("Compare by", list(SEG), format_func=SEG.get, key="coll_dim")
    g = a.groupby(dim, observed=True).agg(accounts=("account_id", "count"),
                                          pay_rate=("any_paid", "mean"),
                                          avg_recovered=("total_recovered", "mean")).reset_index()
    g["ci"] = ci95(g["pay_rate"], g["accounts"])
    fig = px.bar(g, x=dim, y="pay_rate", error_y="ci", text=g["pay_rate"].map(pct),
                 labels={dim: SEG[dim], "pay_rate": "Customers who paid something"})
    fig.update_layout(height=320, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(a):,} accounts had collection actions. {pct(a['any_paid'].mean())} "
               f"paid something back, {money(a['total_recovered'].sum())} in total. "
               f"If the bars are about the same height, this group does not change who pays.")

    # ---------- 5. Does the risk score help? ----------
    st.markdown("#### 5. Does our risk score tell us who will pay?")
    left, right = st.columns(2)
    auc = None
    if a["any_paid"].nunique() == 2:
        auc = roc_auc_score(a["any_paid"], -a["pd_score"])
        left.metric("How well the risk score picks who pays (AUC)", f"{auc:.3f}",
                    help="0.5 = no better than guessing, 1.0 = perfect")
    cure = a.groupby("any_paid")["target_default"].mean()
    if len(cure) == 2:
        right.metric("Went bad: customers who paid vs did not pay",
                     f"{pct(cure.get(1, np.nan))} vs {pct(cure.get(0, np.nan))}")
    if auc is not None and abs(auc - 0.5) < 0.05:
        st.info("**No.** The score is close to 0.5, which is the same as guessing. The risk "
                "model is good at spotting who will fall behind, but not at who will pay back "
                "once they are late. That needs its own model, built on collections data.")
    else:
        st.caption("0.5 means no better than guessing. The further from 0.5, the more the "
                   "risk score helps.")

    # ---------- 6. Recommendation ----------
    st.markdown("#### 6. What we suggest")
    s_share = act.set_index("action_type")["share_of_money"].get("Settlement Offer")
    s_text = (f" Settlement offers bring in the biggest share of the money ({pct(s_share)})."
              if s_share is not None and not pd.isna(s_share) else "")
    st.markdown(f"""
1. **Start cheap, step up only if needed.** All actions bring in money at similar rates
   ({pct(act['pay_rate'].min())} to {pct(act['pay_rate'].max())}), so begin with the cheapest
   (SMS, call) and move to costly ones only when those fail.
2. **Use settlement offers for larger amounts owed.**{s_text}
3. **Build a separate "who will pay back" model.** Our risk score does not predict who pays
   once they are late.
4. **Fix the data first.** Take the outcome from the payment system, not typed by staff,
   and record the real order of actions.
5. **Test before you switch.** Give similar customers different actions at random and
   compare the results. This is the only reliable way to know what really works.

**People stay in control:** these are suggestions for the collections team. Staff choose
the final action for each customer.
""")