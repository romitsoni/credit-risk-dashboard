"""Ask the Portfolio: plain-English questions answered from the data, every number checked."""
import pandas as pd
import streamlit as st

import llm_assistant as llm   # importing this also loads .env

EXAMPLES = [
    "Which loan type goes bad most often?",
    "How often do unemployed people in the lowest income group go bad?",
    "How much money is in each risk group?",
    "Show the 5 riskiest mortgage accounts",
    "How many accounts show warning signs?",
    "What will the default rate be next year?",
]
QUICK = {
    "Which loan types go bad most often": ("profile", {"dim": "loan_type"}),
    "Which job types go bad most often": ("profile", {"dim": "employment_type"}),
    "Which regions go bad most often": ("profile", {"dim": "region"}),
    "Which age groups go bad most often": ("profile", {"dim": "age_band"}),
    "Which income groups go bad most often": ("profile", {"dim": "income_band"}),
    "Income group and job type together": ("cross", {"row": "income_band", "col": "employment_type"}),
    "Key numbers for the whole book": ("kpis", {}),
    "The 10 riskiest accounts": ("top_accounts", {"n": 10}),
    "Accounts with warning signs": ("anomalies", {}),
    "How good is the risk model": ("model_metrics", {}),
}
BADGE = {
    "verified": ("✅ Checked against our data", "Every number in this answer matches the data."),
    "fallback": ("⚠️ Rewritten for safety", "The AI's wording had a number we could not "
                                            "confirm, so this answer was written straight from the data."),
    "declined": ("🚫 Can't answer from our data", "This question is outside what our data covers."),
    "direct": ("📐 Straight from the data", "A ready-made report, no AI used."),
    "no_key": ("⚪ Assistant off", "The AI service is not set up."),
    "error": ("⚪ Assistant unavailable", "The AI service did not respond. Try again in a minute."),
}
TOOL_TEXT = {
    "kpis": "Key numbers for the loans shown",
    "profile": "Breakdown by one field",
    "cross": "Two fields together",
    "account": "Details of one account",
    "top_accounts": "Riskiest accounts",
    "anomalies": "Accounts with warning signs",
    "model_metrics": "Risk model quality (tested on loans the model never saw)",
}
COLUMNS = {
    "accounts": "Loans", "defaults": "Went bad", "default_rate": "Share that went bad",
    "exposure": "Money lent", "exposure_share": "Share of money", "account_id": "Account",
    "loan_type": "Loan type", "loan_amount": "Amount", "pd_score": "Predicted risk",
    "risk_segment": "Risk group", "employment_type": "Job type", "income_band": "Income group",
    "age_band": "Age group", "region": "Region", "city_tier": "City tier",
    "housing_type": "Housing", "education_level": "Education", "disbursal_channel": "Sales channel",
    "loan_purpose": "Loan purpose",
}
PCT_COLS = {"Share that went bad", "Share of money", "Predicted risk"}


def friendly_table(t):
    t = t.rename(columns=COLUMNS).copy()
    for c in t.columns:
        if c in PCT_COLS:
            t[c] = t[c].map(lambda v: f"{v:.1%}")
        elif c in ("Money lent", "Amount"):
            t[c] = t[c].map(lambda v: f"{v / 1e6:,.1f}M" if v >= 1e6 else f"{v / 1e3:,.1f}K")
    return t


def show_result(r):
    label, why = BADGE.get(r["status"], ("", ""))
    st.markdown(f"**Q: {r['question']}**")
    st.markdown(f"{label} · <small>{why}</small>", unsafe_allow_html=True)
    st.info(r["answer"])
    if r.get("source"):
        what = TOOL_TEXT.get(r["tool"], r["tool"])
        detail = ", ".join(str(v).replace("_", " ") for v in (r.get("args") or {}).values())
        with st.expander(f"Where this answer comes from: {what}" + (f" ({detail})" if detail else "")):
            t = r["table"]
            if isinstance(t, pd.DataFrame):
                st.dataframe(friendly_table(t), hide_index=True, use_container_width=True)
            else:
                st.json(t)
            if r.get("unverified_numbers"):
                st.caption("Numbers in the AI's first draft that did not match the data: "
                           f"{', '.join(r['unverified_numbers'])}")


def render(data, df):
    st.subheader("💬 Ask the Portfolio")
    st.caption(f"Ask a question in plain English. Answers cover the loans currently shown "
               f"({len(df):,}), so the filters on the left apply here too.")

    with st.expander("ℹ️ How the assistant works, and its limits"):
        st.markdown("""
**How it works**
1. You ask a question.
2. The AI picks the right report from an approved list (for example *"breakdown by loan type"*).
   At this step it sees **no customer data**.
3. Our system works out the numbers from the data.
4. The AI writes a short answer using only those numbers.
5. Our system **checks every number** in the answer. If one does not match, the AI's wording
   is thrown away and a safe answer is written straight from the data.

**What it can answer:** key numbers, breakdowns by one or two fields, one account,
the riskiest accounts, accounts with warning signs, and how good the risk model is.
**What it will not answer:** forecasts, other banks, opinions, or anything not in our data.

**What is sent to the AI service:** your question, the list of reports, and then one small
summary table. Never individual customer records.

| | |
|---|---|
| **When it runs** (trigger) | When you ask a question |
| **Wrong answer risk** (false positive) | It answers, but picks the wrong report, for example city tier when you meant region. The "Where this answer comes from" box always shows which report was used |
| **Missed answer** (false negative) | It says it can't answer a question our data could answer. Try rewording it, or use a ready-made report |
| **Your check** (human review) | Check the badge and the source table before using a number in a report or a decision |
""")

    if df.empty:
        st.warning("No loans match these filters.")
        return
    an = data["anomalies"][data["anomalies"]["account_id"].isin(df["account_id"])]

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # ---------- Ask a question ----------
    if not llm.has_key():
        st.warning("The AI assistant is not set up right now. The ready-made reports below "
                   "still work, straight from the data.")
    st.markdown("**Try one of these:**")
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(EXAMPLES):
        if cols[i % 3].button(q, key=f"ex_{i}", use_container_width=True):
            clicked = q

    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("Or type your own question",
                          placeholder="e.g. Which region has the fewest loans going bad?")
        sent = st.form_submit_button("Ask", type="primary")
    question = clicked or (q.strip() if sent and q.strip() else None)

    if question:
        with st.spinner("Working it out..."):
            r = llm.ask(question, df, an, data["metrics"])
        st.session_state.qa_history.insert(0, r)

    # ---------- Ready-made reports ----------
    with st.expander("📐 Ready-made reports (work even without the AI)"):
        a, b = st.columns([3, 1])
        pick = a.selectbox("Choose a report", list(QUICK), label_visibility="collapsed")
        if b.button("Show", use_container_width=True):
            tool, args = QUICK[pick]
            r = llm.answer_without_llm(tool, args, df, an, data["metrics"])
            r["question"] = pick
            st.session_state.qa_history.insert(0, r)

    # ---------- Answers ----------
    if st.session_state.qa_history:
        st.markdown("---")
        top = st.columns([4, 1])
        top[0].markdown("#### Your answers (newest first)")
        if top[1].button("Clear", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()
        for r in st.session_state.qa_history[:10]:
            show_result(r)
            st.markdown("")