"""Ask the Portfolio: LLM Q&A grounded in the data, with number verification."""
import pandas as pd
import streamlit as st

import llm_assistant as llm   # importing this also loads .env

EXAMPLES = [
    "Which loan type has the highest default rate?",
    "What is the default rate for unemployed people in the lowest income band?",
    "How much exposure is in each risk segment?",
    "Show the 5 riskiest mortgage accounts",
    "How many accounts are flagged as anomalies?",
    "What will the default rate be next year?",
]
QUICK = {
    "Default rate by loan type": ("profile", {"dim": "loan_type"}),
    "Default rate by employment type": ("profile", {"dim": "employment_type"}),
    "Default rate by region": ("profile", {"dim": "region"}),
    "Default rate by age band": ("profile", {"dim": "age_band"}),
    "Default rate by income band": ("profile", {"dim": "income_band"}),
    "Income band x employment type": ("cross", {"row": "income_band", "col": "employment_type"}),
    "Portfolio KPIs": ("kpis", {}),
    "Top 10 riskiest accounts": ("top_accounts", {"n": 10}),
    "Anomaly summary": ("anomalies", {}),
    "Model performance": ("model_metrics", {}),
}
BADGE = {
    "verified": ("✅ Verified", "Every number in this answer was checked against the data."),
    "fallback": ("⚠️ Safe answer", "The AI's wording failed the number check, so this answer "
                                  "was written directly by code from the data."),
    "declined": ("🚫 Declined", "The data cannot answer this question."),
    "direct": ("📐 Direct from data", "Answered by code, without the AI."),
    "no_key": ("⚪ AI off", "No API key is set."),
    "error": ("⚪ AI unavailable", "The AI service did not respond."),
}


def show_result(r):
    label, why = BADGE.get(r["status"], ("", ""))
    st.markdown(f"**Q: {r['question']}**")
    st.markdown(f"{label} · <small>{why}</small>", unsafe_allow_html=True)
    st.info(r["answer"])
    if r.get("source"):
        with st.expander(f"Source: {r['source']} (tool `{r['tool']}`, args {r['args']})"):
            t = r["table"]
            if isinstance(t, pd.DataFrame):
                st.dataframe(t, hide_index=True, use_container_width=True)
            else:
                st.json(t)
            if r.get("unverified_numbers"):
                st.caption(f"Numbers from the AI that did not match the data: "
                           f"{r['unverified_numbers']}")


def render(data, df):
    st.subheader("💬 Ask the Portfolio")
    st.caption(f"Answers use the current sidebar filters ({len(df):,} loans).")

    with st.expander("ℹ️ How this works: trigger, false positive / negative, human review",
                     expanded=False):
        st.markdown("""
**How it works.** (1) The AI reads your question and picks one approved tool, for example
*default rate by loan type*. It sees **no data** at this step. (2) Python/pandas computes the
numbers. (3) The AI writes a short answer from that small result table only.
(4) Code checks that **every number** in the answer exists in the result. If not, the AI text
is replaced by a code-written answer. Questions the data cannot answer are declined.

**What is sent to the LLM (Google Gemini):** your question + the tool list; then the small
aggregated result table. **No raw customer rows.**

**Trigger:** an analyst types a question.

**False positive:** the assistant answers, but with the wrong tool (e.g. *city tier* when you
meant *region*). The source box shows exactly which tool and table were used.

**False negative:** it declines a question the data could answer. Rephrase, or use the quick
questions below.

**Human-in-the-loop:** the analyst checks the badge and the source table before using a number
in any report or decision.
""")

    if df.empty:
        st.warning("No loans match the filters.")
        return
    an = data["anomalies"][data["anomalies"]["account_id"].isin(df["account_id"])]

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # ---------- Free-text question ----------
    if not llm.has_key():
        st.warning("AI assistant is off: no GEMINI_API_KEY is set. "
                   "Quick questions below still work, straight from the data.")
    st.markdown("**Example questions** (click to ask):")
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(EXAMPLES):
        if cols[i % 3].button(q, key=f"ex_{i}", use_container_width=True):
            clicked = q

    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("Your question", placeholder="e.g. Which region has the lowest default rate?")
        sent = st.form_submit_button("Ask", type="primary")
    question = clicked or (q.strip() if sent and q.strip() else None)

    if question:
        with st.spinner("Thinking..."):
            r = llm.ask(question, df, an, data["metrics"])
        st.session_state.qa_history.insert(0, r)

    # ---------- Quick questions (no AI) ----------
    with st.expander("📐 Quick questions (no AI, always available)"):
        a, b = st.columns([3, 1])
        pick = a.selectbox("Pick one", list(QUICK), label_visibility="collapsed")
        if b.button("Show", use_container_width=True):
            tool, args = QUICK[pick]
            r = llm.answer_without_llm(tool, args, df, an, data["metrics"])
            r["question"] = pick
            st.session_state.qa_history.insert(0, r)

    # ---------- History ----------
    if st.session_state.qa_history:
        st.markdown("---")
        top = st.columns([4, 1])
        top[0].markdown("#### Answers (newest first)")
        if top[1].button("Clear", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()
        for r in st.session_state.qa_history[:10]:
            show_result(r)
            st.markdown("")