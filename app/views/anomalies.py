"""Anomaly review: flagged accounts, why they were flagged, analyst decisions."""
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
    "sudden_miss": "Own history: clean 9 months, then missed",
    "dpd_jump": "Own history: DPD jumped 30+ days",
    "peer_bur_utilization": "Peer: bureau utilization in top 1%",
    "peer_bur_mean_dpd": "Peer: bureau DPD in top 1%",
    "isolation_forest": "Rare combination (Isolation Forest)",
}


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
    st.subheader("🚩 Anomaly Review")

    with st.expander("ℹ️ How this works: trigger, false positive / negative, human review",
                     expanded=False):
        st.markdown("""
**What triggers a flag** (an account can break several rules):
- **Own history:** no late payment in months 1-9, then a missed payment in months 10-12;
  or average DPD in months 10-12 is 30+ days higher than in months 1-9.
- **Peer segment:** bureau utilization or bureau DPD is above the top 1% of similar
  borrowers (same loan type x income band).
- **Rare combination:** Isolation Forest marks the most unusual 2% of accounts across
  repayment + bureau features.

**False positive:** a flagged account that is fine, e.g. one late month due to a bank
holiday or a salary delay, or a customer with high utilization who always pays.

**False negative:** a risky account that is *not* flagged, e.g. slow, steady decline that
never crosses a threshold, or problems not visible in our data.

**Human-in-the-loop:** the analyst reads the reason, checks the account, then records
**Confirm** (send to early collections / review) or **Dismiss** (no action), with a note.
Decisions are logged with reviewer and time. The system never acts on its own.
""")

    an = data["anomalies"]
    an = an[an["account_id"].isin(df["account_id"])].copy()
    if an.empty:
        st.info("No flagged accounts for these filters.")
        return

    dec = load_decisions()
    latest = dec.drop_duplicates("account_id", keep="last").set_index("account_id")
    an["status"] = an["account_id"].map(latest["decision"]).fillna("Pending")

    # ---------- Summary ----------
    rest = df[~df["account_id"].isin(an["account_id"])]
    c = st.columns(5)
    c[0].metric("Flagged accounts", f"{len(an):,}", help=f"of {len(df):,} loans in view")
    c[1].metric("Flagged default rate", pct(an["target_default"].mean()))
    c[2].metric("Not-flagged default rate", pct(rest["target_default"].mean()))
    c[3].metric("Flagged exposure", money(an["loan_amount"].sum()))
    c[4].metric("Pending review", f"{(an['status'] == 'Pending').sum():,}")

    rule_counts = (an["flags"].str.split(", ").explode().map(RULES)
                   .value_counts().rename_axis("rule").reset_index(name="accounts"))
    fig = px.bar(rule_counts, x="accounts", y="rule", orientation="h",
                 title="Flags by rule", labels={"rule": ""})
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Review list ----------
    st.markdown("#### Review list (most flags first, then highest model PD)")
    a, b, c2 = st.columns(3)
    rules = a.multiselect("Rule", list(RULES), format_func=RULES.get)
    status = b.multiselect("Status", ["Pending", "Confirmed", "Dismissed"],
                           default=["Pending"])
    min_flags = c2.slider("Minimum flags", 1, int(an["flag_count"].max()), 1)
    v = an[an["flag_count"] >= min_flags]
    if rules:
        v = v[v["flags"].apply(lambda s: any(r in s.split(", ") for r in rules))]
    if status:
        v = v[v["status"].isin(status)]

    show = v[["account_id", "status", "flag_count", "loan_type", "loan_amount",
              "pd_score", "risk_segment", "reasons"]].copy()
    show["loan_amount"] = show["loan_amount"].map(money)
    show["pd_score"] = show["pd_score"].map(pct)
    st.caption(f"{len(v):,} accounts match.")
    st.dataframe(show, hide_index=True, use_container_width=True, height=300)

    if v.empty:
        return

    # ---------- Review one account ----------
    st.markdown("#### Review an account")
    aid = st.selectbox("Account to review", v["account_id"].tolist(), key="an_pick")
    r = v[v["account_id"] == aid].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{aid}** · {r['loan_type']} · {money(r['loan_amount'])} · "
                    f"PD {pct(r['pd_score'])} ({r['risk_segment']}) · status **{r['status']}**")
        st.markdown("**Why it was flagged:**")
        for reason in r["reasons"].split(" | "):
            st.markdown(f"- {reason}")
        rep = load_raw()[0]
        h = rep[rep["account_id"] == aid].sort_values("payment_month")
        fig = px.bar(h, x="payment_month", y="days_past_due", color="payment_status",
                     color_discrete_map={"On-Time": "#2e7d32", "Late": "#f9a825",
                                         "Missed": "#c62828"},
                     labels={"payment_month": "Month", "days_past_due": "Days past due"},
                     title="Own repayment history")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)
        hist = dec[dec["account_id"] == aid]
        if len(hist):
            st.caption("Decision history:")
            st.dataframe(hist, hide_index=True, use_container_width=True)

    with right:
        with st.form(f"decide_{aid}", clear_on_submit=True):
            st.markdown("**Record decision**")
            choice = st.radio("Decision", ["Confirmed", "Dismissed"], horizontal=True,
                              captions=["Real concern: send for early action",
                                        "False alarm: no action"])
            reviewer = st.text_input("Reviewer", placeholder="Your name")
            note = st.text_area("Note", placeholder="e.g. Called customer, salary delayed")
            if st.form_submit_button("Save decision", type="primary"):
                save_decision(aid, choice, reviewer, note)
                st.success(f"Saved: {aid} → {choice}")
                st.rerun()

    # ---------- Decision log ----------
    st.markdown("#### Decision log")
    if dec.empty:
        st.caption("No decisions yet.")
    else:
        d = dec.merge(data["anomalies"][["account_id", "target_default"]],
                      on="account_id", how="left")
        conf = d.drop_duplicates("account_id", keep="last")
        c = st.columns(3)
        c[0].metric("Decisions recorded", f"{len(conf):,}")
        c[1].metric("Confirmed", f"{(conf['decision'] == 'Confirmed').sum():,}")
        c[2].metric("Dismissed", f"{(conf['decision'] == 'Dismissed').sum():,}")
        st.dataframe(dec.sort_values("timestamp", ascending=False),
                     hide_index=True, use_container_width=True)
        st.download_button("Download decision log (CSV)",
                           dec.to_csv(index=False).encode(), "decisions.csv", "text/csv")
    st.caption("Demo storage: decisions are saved to a CSV file. On Streamlit Cloud this file "
               "resets when the app restarts; in production it would go to a database "
               "with user login and an audit trail.")