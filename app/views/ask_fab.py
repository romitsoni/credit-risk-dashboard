"""Floating 'Ask the Portfolio' button, visible on every tab.
Opens a pop-up that uses the same safe, checked answers as the Ask tab."""
import streamlit as st

import llm_assistant as llm
from views.ask import EXAMPLES, show_result

FAB_CSS = """
<style>
div[class*="st-key-ask_fab"] {
    position: fixed !important;
    bottom: 80px;
    right: 28px;
    z-index: 999999;
    width: auto !important;
}
div[class*="st-key-ask_fab"] button {
    border-radius: 999px;
    padding: 0.7rem 1.3rem;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
</style>
"""


@st.dialog("💬 Ask the Portfolio", width="large")
def ask_dialog(data, df):
    if df.empty:
        st.warning("No loans match the current filters.")
        return
    st.caption(f"Ask a question in plain English. Answers cover the loans currently shown "
               f"({len(df):,}), and every number is checked against our data.")
    if not llm.has_key():
        st.warning("The AI assistant is not set up right now. You can still use the "
                   "ready-made reports in the Ask the Portfolio tab.")
        return
    an = data["anomalies"][data["anomalies"]["account_id"].isin(df["account_id"])]

    pick = st.selectbox("Pick an example, or type your own question below",
                        ["(type my own)"] + EXAMPLES, key="fab_example")
    with st.form("fab_form"):
        q = st.text_input("Your question",
                          value="" if pick.startswith("(") else pick,
                          placeholder="e.g. Which region has the fewest loans going bad?")
        go = st.form_submit_button("Ask", type="primary")

    if go and q.strip():
        with st.spinner("Working it out..."):
            r = llm.ask(q.strip(), df, an, data["metrics"])
        st.session_state.setdefault("qa_history", []).insert(0, r)
        show_result(r)
    st.caption("Your answers are also saved in the Ask the Portfolio tab.")


def render(data, df):
    st.markdown(FAB_CSS, unsafe_allow_html=True)
    with st.container(key="ask_fab"):
        if st.button("💬 Ask the Portfolio", key="fab_button", type="primary"):
            ask_dialog(data, df)