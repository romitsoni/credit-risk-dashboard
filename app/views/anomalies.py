"""Warning signs: accounts behaving unusually, why, and the analyst's decision."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from views.account import load_raw
from views.common import money, pct

DEC_DIR = Path("data/decisions")
DEC_FILE = DEC_DIR / "decisions.csv"
DEC_COLS = ["account_id", "decision", "reviewer", "note", "timestamp"]
RULES = {
    "sudden_miss": "Stopped paying suddenly",
    "dpd_jump": "Paying much later than before",
    "peer_bur_utilization": "Using far more credit at other banks than similar customers",
    "peer_bur_mean_dpd": "Much later at other banks than similar customers",
    "isolation_forest": "Unusual mix of behaviour",
}
STATUS = {"Pending": "Waiting for review", "Confirmed": "Confirmed", "Dismissed": "Dismissed"}
SHORT = {"sudden_miss": "Stopped paying", "dpd_jump": "Paying later",
         "peer_bur_utilization": "High credit use elsewhere",
         "peer_bur_mean_dpd": "Late at other banks", "isolation_forest": "Unusual pattern"}
PLAIN = [
    ("No late payment in months 1-9, then missed", "Paid on time for 9 months, then missed"),
    ("Average DPD rose from", "Average days late rose from"),
    ("(months 1-9 vs 10-12)", "(first 9 months vs last 3)"),
    ("Average bureau DPD", "Average days late at other banks:"),
    ("Bureau utilization", "Share of credit limits used at other banks:"),
    ("is above the top 1% of peers", "which is in the top 1% for similar customers"),
    ("Rare combination of repayment and bureau behaviour (Isolation Forest, top 2%)",
     "Unusual mix of payment and other-bank behaviour (among the rarest 2% of accounts)"),
]


def plain_reason(text):
    for old, new in PLAIN:
        text = text.replace(old, new)
    return text


def load_decisions():
    if DEC_FILE.exists():
        return pd.read_csv(DEC_FILE)
    return pd.DataFrame(columns=DEC_COLS)


def save_decision(aid, decision, reviewer, note):
    DEC_DIR.mkdir(parents=True, exist_ok=True)
    d = load_decisions()
    row = pd.DataFrame([{"account_id": aid, "decision": decision,
                         "reviewer": reviewer or "analyst", "note": note,
                         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    pd.concat([d, row], ignore_index=True).to_csv(DEC_FILE, index=False)


def render(data, df):
    st.subheader("🚩 Accounts that need a look")
    st.caption("These accounts are behaving differently from their own past, or from "
               "similar customers. Each one comes with the reason. You decide what happens next.")

    with st.expander("ℹ️ How accounts get flagged, and the limits"):
        st.markdown("""
**An account is flagged when** (it can match more than one):
- **Stopped paying suddenly:** it paid on time for the first 9 months, then missed a payment
  in the last 3 months.
- **Paying much later than before:** its average delay in the last 3 months is at least
  30 days more than in the first 9 months.
- **Using far more credit at other banks:** it uses more of its credit limits at other banks
  than 99% of similar customers (same loan type and income group).
- **Much later at other banks:** it is later on payments at other banks than 99% of
  similar customers.
- **Unusual mix of behaviour:** a statistical check (Isolation Forest) finds it among the
  rarest 2% of accounts when payment and other-bank behaviour are looked at together.

| | |
|---|---|
| **When it runs** (trigger) | Every time the data is refreshed, on every account |
| **False alarm** (false positive) | A flagged account that is fine, for example one late month because a salary came late, or a customer who uses a lot of credit but always pays |
| **Missed case** (false negative) | A risky account that is not flagged, for example one that gets slowly worse without crossing any line, or a problem our data cannot see |
| **Your decision** (human review) | Read the reason, check the account, then choose Confirm or Dismiss and add a note. Every decision is saved with your name and the time. The system never acts on its own |
""")

    an = data["anomalies"]
    an = an[an["account_id"].isin(df["account_id"])].copy()
    if an.empty:
        st.info("No accounts are flagged for these filters.")
        return

    dec = load_decisions()
    latest = dec.drop_duplicates("account_id", keep="last").set_index("account_id")
    an["status"] = an["account_id"].map(latest["decision"]).fillna("Pending")
    an["plain_reasons"] = an["reasons"].map(plain_reason)

    # ---------- Summary ----------
    rest = df[~df["account_id"].isin(an["account_id"])]
    f_rate, r_rate = an["target_default"].mean(), rest["target_default"].mean()
    ratio = f_rate / r_rate if r_rate > 0 else float("nan")
    st.info(f"**{len(an):,} accounts** are flagged. They go bad **{ratio:.1f} times as often** "
            f"as the rest ({pct(f_rate)} vs {pct(r_rate)}), so the flags point to real risk.")
    c = st.columns(5)
    c[0].metric("Accounts flagged", f"{len(an):,}", help=f"Out of {len(df):,} loans shown")
    c[1].metric("How often flagged accounts go bad", pct(f_rate))
    c[2].metric("How often other accounts go bad", pct(r_rate))
    c[3].metric("Money in flagged loans", money(an["loan_amount"].sum()))
    c[4].metric("Waiting for review", f"{(an['status'] == 'Pending').sum():,}")

    counts = (an["flags"].str.split(", ").explode().map(RULES)
              .value_counts().rename_axis("reason").reset_index(name="accounts"))
    fig = px.bar(counts, x="accounts", y="reason", orientation="h",
                 title="Why accounts were flagged",
                 labels={"reason": "", "accounts": "Number of accounts"})
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Review list ----------
    st.markdown("#### Review list")
    st.caption("Accounts with the most warning signs come first, then the highest predicted risk.")
    a, b, c2 = st.columns(3)
    rules = a.multiselect("Show only this reason", list(RULES), format_func=RULES.get)
    status = b.multiselect("Status", ["Pending", "Confirmed", "Dismissed"],
                           default=["Pending"], format_func=STATUS.get)
    min_flags = c2.slider("At least this many warning signs", 1, int(an["flag_count"].max()), 1)
    v = an[an["flag_count"] >= min_flags]
    if rules:
        v = v[v["flags"].apply(lambda s: any(r in s.split(", ") for r in rules))]
    if status:
        v = v[v["status"].isin(status)]

    
    show = pd.DataFrame({
        "Account": v["account_id"],
        "Status": v["status"].map(STATUS),
        "Warning signs": v["flag_count"],
        "Why (short)": v["flags"].map(lambda s: " + ".join(SHORT.get(f, f) for f in s.split(", "))),
        "Loan type": v["loan_type"],
        "Amount": v["loan_amount"].map(money),
        "Predicted risk": v["pd_score"].map(pct),
        "Risk group": v["risk_segment"].astype(str),
    })
    st.caption(f"{len(v):,} accounts match. Pick one in **Review one account** below "
               f"to read the full reasons.")
    st.dataframe(show, hide_index=True, use_container_width=True, height=300,
                 column_config={"Why (short)": st.column_config.TextColumn(width="large")})

    if v.empty:
        return

    # ---------- Review one account ----------
    st.markdown("#### Review one account")
    aid = st.selectbox("Choose an account", v["account_id"].tolist(), key="an_pick")
    r = v[v["account_id"] == aid].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{aid}** · {r['loan_type']} · {money(r['loan_amount'])} · "
                    f"predicted risk {pct(r['pd_score'])} ({r['risk_segment']}) · "
                    f"status: **{STATUS[r['status']]}**")
        st.markdown("**Why it was flagged:**")
        for reason in r["plain_reasons"].split(" | "):
            st.markdown(f"- {reason}")
        rep = load_raw()[0]
        h = rep[rep["account_id"] == aid].sort_values("payment_month").copy()
        h["Payment"] = h["payment_status"].replace(
            {"On-Time": "On time", "Late": "Late", "Missed": "Not paid"})
        fig = px.bar(h, x="payment_month", y="days_past_due", color="Payment",
                     color_discrete_map={"On time": "#2e7d32", "Late": "#f9a825",
                                         "Not paid": "#c62828"},
                     labels={"payment_month": "Month of the loan", "days_past_due": "Days late"},
                     title="Payment history (on-time months show no bar)")
        fig.update_layout(height=260, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        hist = dec[dec["account_id"] == aid]
        if len(hist):
            st.caption("Earlier decisions on this account:")
            st.dataframe(hist.rename(columns={"account_id": "Account", "decision": "Decision",
                                              "reviewer": "Reviewer", "note": "Note",
                                              "timestamp": "When"}),
                         hide_index=True, use_container_width=True)

    with right:
        st.markdown("**Before you decide:**")
        st.markdown("- Look at the payment history on the left.\n"
                    "- Open the account in the **Account** tab for the full picture.\n"
                    "- If needed, contact the customer.")
        with st.form(f"decide_{aid}", clear_on_submit=True):
            st.markdown("**Your decision**")
            choice = st.radio("Decision", ["Confirmed", "Dismissed"], horizontal=True,
                              format_func=lambda x: "Confirm" if x == "Confirmed" else "Dismiss",
                              captions=["Real concern: act early",
                                        "False alarm: no action"])
            reviewer = st.text_input("Your name", placeholder="e.g. Priya")
            note = st.text_area("Note", placeholder="e.g. Called the customer: salary came late, "
                                                    "will pay this week")
            if st.form_submit_button("Save decision", type="primary"):
                save_decision(aid, choice, reviewer, note)
                st.success(f"Saved: {aid} marked as {choice.lower()}.")
                st.rerun()

    # ---------- Decision log ----------
    st.markdown("#### Decisions made so far")
    if dec.empty:
        st.caption("No decisions yet.")
    else:
        last = dec.drop_duplicates("account_id", keep="last")
        c = st.columns(3)
        c[0].metric("Accounts reviewed", f"{len(last):,}")
        c[2].metric("Dismissed", f"{(last['decision'] == 'Dismissed').sum():,}")
        log = dec.sort_values("timestamp", ascending=False).rename(columns={
            "account_id": "Account", "decision": "Decision", "reviewer": "Reviewer",
            "note": "Note", "timestamp": "When"})
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.download_button("Download the decision log (CSV)",
                           dec.to_csv(index=False).encode(), "decisions.csv", "text/csv")
    st.caption("In this demo, decisions are kept until the app restarts. In the bank's live "
               "system they would be stored permanently, with your login and a full history.")