"""Portfolio page: where the risk is in the loan book, in plain language."""
import pandas as pd
import plotly.express as px
import streamlit as st

import analytics
from views.common import SEG_COLORS, money, pct, risk_reasons, typical_text, typical_values

DIMS = {k: v for k, v in analytics.DIMENSIONS.items() if k != "risk_segment"}
GROUP_TEXT = {
    "Low": "under 10% chance of going bad",
    "Medium": "10% to 20% chance",
    "High": "20% to 35% chance",
    "Very High": "over 35% chance",
}


def render(data, df):
    full = data["scored"]
    st.subheader("Portfolio overview")
    st.caption(f"Showing {len(df):,} of {len(full):,} loans. Use the filters on the left "
               "to focus on one part of the book.")
    if df.empty:
        st.warning("No loans match these filters.")
        return

    with st.expander("How to read this page"):
        st.markdown("""
- **Went bad** means the customer stopped paying (a default).
- **Predicted risk** is the model's estimate, from 0% to 100%, of how likely a loan is to go bad.
- **Risk groups** sort loans by predicted risk: **Low** under 10%, **Medium** 10% to 20%, **High** 20% to 35%, **Very High** over 35%.
- **Money lent** is the original amount of the loans.
- Everything on this page follows the filters on the left.
""")

    # ---------- Headline numbers ----------
    k = analytics.portfolio_kpis(df)
    by_type = analytics.profile(df, "loan_type")
    worst = by_type.sort_values("default_rate", ascending=False).iloc[0]
    st.info(f"**{pct(k['high_risk_exposure_share'])} of the money** is in high-risk loans. "
            f"**{worst['loan_type']}** loans go bad most often ({pct(worst['default_rate'])}).")

    c = st.columns(4)
    c[0].metric("Loans", f"{k['accounts']:,}", help=f"Held by {k['customers']:,} customers")
    c[1].metric("Money lent", money(k["total_exposure"]),
                help="Total of the original loan amounts")
    c[2].metric("Loans that went bad", pct(k["default_rate"]),
                help="Share of loans where the customer stopped paying")
    c[3].metric("Money in high-risk loans", pct(k["high_risk_exposure_share"]),
                help="Share of the money lent that sits in the High and Very High risk groups")

    # ---------- Risk groups ----------
    st.markdown("#### Risk groups")
    st.markdown("Each loan is put in a group by its **predicted risk** "
                "(the model's chance that the loan goes bad):")
    boxes = st.columns(4)
    for box, (name, rng, meaning) in zip(boxes, [
            ("Low", "under 10%", "fewer than 10 in 100 go bad"),
            ("Medium", "10% to 20%", "10 to 20 in 100 go bad"),
            ("High", "20% to 35%", "20 to 35 in 100 go bad"),
            ("Very High", "over 35%", "more than 35 in 100 go bad")]):
        box.markdown(
            f"<div style='border-left:6px solid {SEG_COLORS[name]};padding:6px 10px;"
            f"background:rgba(128,128,128,0.08);border-radius:4px'>"
            f"<b style='color:{SEG_COLORS[name]}'>{name}</b><br>"
            f"<span style='font-size:1.15em'><b>{rng}</b></span><br>"
            f"<small>{meaning}</small></div>", unsafe_allow_html=True)
    st.caption("The average loan has about a 16% chance of going bad, so Medium is around average.")
    seg = analytics.profile(df, "risk_segment")
    left, right = st.columns(2)
    fig = px.bar(seg, x="risk_segment", y="exposure", color="risk_segment",
                 color_discrete_map=SEG_COLORS, text=seg["exposure"].map(money),
                 labels={"risk_segment": "Risk group", "exposure": "Money lent"},
                 title="Money lent in each risk group")
    fig.update_layout(showlegend=False, height=340)
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(seg, x="risk_segment", y="default_rate", color="risk_segment",
                 color_discrete_map=SEG_COLORS, text=seg["default_rate"].map(pct),
                 labels={"risk_segment": "Risk group", "default_rate": "Share that went bad"},
                 title="How many loans in each group really went bad")
    fig.update_layout(showlegend=False, height=340, yaxis_tickformat=".0%")
    right.plotly_chart(fig, use_container_width=True)
    st.caption("If the bars on the right rise from Low to Very High, the risk groups are "
               "sorting customers correctly.")

    # ---------- Compare groups ----------
    st.markdown("#### Compare groups")
    dim = st.selectbox("Compare by", list(DIMS), format_func=DIMS.get, key="pf_dim")
    t = analytics.profile(df, dim)
    left, right = st.columns(2)
    fig = px.bar(t, x=dim, y="default_rate", text=t["default_rate"].map(pct),
                 labels={dim: DIMS[dim], "default_rate": "Share that went bad"},
                 title=f"How often loans go bad, by {DIMS[dim].lower()}")
    fig.add_hline(y=k["default_rate"], line_dash="dash",
                  annotation_text=f"Average {pct(k['default_rate'])}")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(t, x=dim, y="exposure_share", text=t["exposure_share"].map(pct),
                 labels={dim: DIMS[dim], "exposure_share": "Share of money lent"},
                 title=f"Where the money is, by {DIMS[dim].lower()}")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    right.plotly_chart(fig, use_container_width=True)
    st.caption("Groups above the dashed line go bad more often than average. "
               "A group that is both above the line and holds a lot of money needs attention first.")

    table = pd.DataFrame({
        DIMS[dim]: t[dim].astype(str),
        "Loans": t["accounts"].map("{:,}".format),
        "Went bad": t["defaults"].map("{:,}".format),
        "Share that went bad": t["default_rate"].map(pct),
        "Money lent": t["exposure"].map(money),
        "Share of money": t["exposure_share"].map(pct),
        "Predicted risk": t["avg_pd"].map(pct),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)

    # ---------- Two things together ----------
    st.markdown("#### Two things together")
    st.caption("Some risks only show up when you combine two things, "
               "for example income and type of job. Darker red means more loans went bad.")
    a, b = st.columns(2)
    row = a.selectbox("Rows", list(DIMS), index=list(DIMS).index("income_band"),
                      format_func=DIMS.get, key="cx_row")
    col = b.selectbox("Columns", list(DIMS), index=list(DIMS).index("employment_type"),
                      format_func=DIMS.get, key="cx_col")
    if row == col:
        st.info("Pick two different things to compare.")
    else:
        x = analytics.cross(df, row, col)
        fig = px.imshow(x.values, x=[str(v) for v in x.columns],
                        y=[str(v) for v in x.index], text_auto=".1%",
                        color_continuous_scale="Reds", aspect="auto",
                        labels={"color": "Share that went bad"})
        fig.update_layout(height=380, xaxis_title=DIMS[col], yaxis_title=DIMS[row])
        st.plotly_chart(fig, use_container_width=True)

    # ---------- By year ----------
    st.markdown("#### By the year the loan was given")
    split = st.selectbox("Split by", ["(none)"] + list(DIMS),
                         format_func=lambda v: "Nothing" if v == "(none)" else DIMS.get(v, v),
                         key="tr_split")
    d = df.assign(year=pd.to_datetime(df["origination_date"]).dt.year)
    keys = ["year"] if split == "(none)" else ["year", split]
    tr = d.groupby(keys, observed=True).agg(
        default_rate=("target_default", "mean"),
        exposure=("loan_amount", "sum")).reset_index()
    left, right = st.columns(2)
    fig = px.line(tr, x="year", y="default_rate", markers=True,
                  color=None if split == "(none)" else split,
                  labels={"year": "Year the loan was given", "default_rate": "Share that went bad"},
                  title="How often loans went bad, by year given")
    fig.update_layout(height=340, yaxis_tickformat=".0%")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(tr, x="year", y="exposure",
                 color=None if split == "(none)" else split,
                 labels={"year": "Year the loan was given", "exposure": "Money lent"},
                 title="Money lent, by year given")
    fig.update_layout(height=340)
    right.plotly_chart(fig, use_container_width=True)
    st.caption("Watch the scale on the left chart: small changes can look large.")

    # ---------- See the loans in a group ----------
    st.markdown("#### See the loans in a group")
    a, b = st.columns(2)
    ddim = a.selectbox("Group by", list(DIMS), format_func=DIMS.get, key="dd_dim")
    vals = [v for v in analytics.profile(df, ddim)[ddim]]
    grp = b.selectbox("Group", vals, key="dd_val")
    acc = df[df[ddim] == grp].sort_values("pd_score", ascending=False)
    top = acc.head(200)
    typical = typical_values(data["scored"])
    cols = {"Account": top["account_id"],
            "Why it is risky": top.apply(lambda r: risk_reasons(r, typical), axis=1),
            "Predicted risk": top["pd_score"].map(pct),
            "Risk group": top["risk_segment"].astype(str),
            "Amount": top["loan_amount"].map(money)}
    if ddim != "loan_type":
        cols["Loan type"] = top["loan_type"]
    view = pd.DataFrame(cols)
    st.caption(f"{len(acc):,} loans in this group, riskiest first (top 200 shown). "
               f"A typical customer has: {typical_text(typical)}. "
               "Open any account in the Account tab to see the full picture.")
    st.dataframe(view, hide_index=True, use_container_width=True,
                 column_config={"Why it is risky": st.column_config.TextColumn(width="large")})

    # ---------- What stands out ----------
    with st.expander("💡 What stands out in the whole book", expanded=True):
        for i, item in enumerate(analytics.insights(full), 1):
            st.markdown(f"**{i}. {item['title']}**  \n{item['finding']}  \n"
                        f"*What to do:* {item['so_what']}")