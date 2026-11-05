"""Portfolio view: KPIs, risk segments, profile by dimension, cross-cut, trend, drill-down."""
import pandas as pd
import plotly.express as px
import streamlit as st

import analytics
from views.common import SEG_COLORS, money, pct, show_table

DIMS = {k: v for k, v in analytics.DIMENSIONS.items() if k != "risk_segment"}


def render(data, df):
    full = data["scored"]
    st.subheader("Portfolio view")
    st.caption(f"Showing {len(df):,} of {len(full):,} loans. Use the sidebar to filter.")
    if df.empty:
        st.warning("No loans match these filters.")
        return

    # ---------- KPIs ----------
    k = analytics.portfolio_kpis(df)
    c = st.columns(5)
    c[0].metric("Loans", f"{k['accounts']:,}", help=f"{k['customers']:,} customers")
    c[1].metric("Exposure", money(k["total_exposure"]), help="Sum of original loan amounts")
    c[2].metric("Observed default rate", pct(k["default_rate"]))
    c[3].metric("Avg model PD", pct(k["avg_pd"]), help="Average predicted probability of default")
    c[4].metric("High-risk exposure", pct(k["high_risk_exposure_share"]),
                help="Share of exposure in High + Very High segments")

    # ---------- Risk segments ----------
    st.markdown("#### Risk segments")
    seg = analytics.profile(df, "risk_segment")
    left, right = st.columns(2)
    fig = px.bar(seg, x="risk_segment", y="exposure", color="risk_segment",
                 color_discrete_map=SEG_COLORS,
                 text=seg["exposure"].map(money),
                 labels={"risk_segment": "Segment", "exposure": "Exposure"},
                 title="Exposure by risk segment")
    fig.update_layout(showlegend=False, height=340)
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(seg, x="risk_segment", y="default_rate", color="risk_segment",
                 color_discrete_map=SEG_COLORS,
                 text=seg["default_rate"].map(pct),
                 labels={"risk_segment": "Segment", "default_rate": "Default rate"},
                 title="Actual default rate by segment (model check)")
    fig.update_layout(showlegend=False, height=340, yaxis_tickformat=".0%")
    right.plotly_chart(fig, use_container_width=True)
    st.caption("Segments: Low < 10% PD · Medium 10-20% · High 20-35% · Very High > 35%.")

    # ---------- Profile by one dimension ----------
    st.markdown("#### Profile by dimension")
    dim = st.selectbox("Dimension", list(DIMS), format_func=DIMS.get, key="pf_dim")
    t = analytics.profile(df, dim)
    left, right = st.columns(2)
    fig = px.bar(t, x=dim, y="default_rate", text=t["default_rate"].map(pct),
                 labels={dim: DIMS[dim], "default_rate": "Default rate"},
                 title=f"Default rate by {DIMS[dim].lower()}")
    fig.add_hline(y=k["default_rate"], line_dash="dash",
                  annotation_text=f"Average {pct(k['default_rate'])}")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(t, x=dim, y="exposure_share", text=t["exposure_share"].map(pct),
                 labels={dim: DIMS[dim], "exposure_share": "Share of exposure"},
                 title=f"Share of exposure by {DIMS[dim].lower()}")
    fig.update_layout(height=360, yaxis_tickformat=".0%")
    right.plotly_chart(fig, use_container_width=True)
    show_table(t)

    # ---------- Cross-cut ----------
    st.markdown("#### Cross-cut: default rate by two dimensions")
    a, b = st.columns(2)
    row = a.selectbox("Rows", list(DIMS), index=list(DIMS).index("income_band"),
                      format_func=DIMS.get, key="cx_row")
    col = b.selectbox("Columns", list(DIMS), index=list(DIMS).index("employment_type"),
                      format_func=DIMS.get, key="cx_col")
    if row == col:
        st.info("Pick two different dimensions.")
    else:
        x = analytics.cross(df, row, col)
        fig = px.imshow(x.values, x=[str(v) for v in x.columns],
                        y=[str(v) for v in x.index], text_auto=".1%",
                        color_continuous_scale="Reds", aspect="auto",
                        labels={"color": "Default rate"})
        fig.update_layout(height=380, xaxis_title=DIMS[col], yaxis_title=DIMS[row])
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Trend ----------
    st.markdown("#### Trend by origination year")
    split = st.selectbox("Split by", ["(none)"] + list(DIMS),
                         format_func=lambda v: DIMS.get(v, v), key="tr_split")
    d = df.assign(orig_year=pd.to_datetime(df["origination_date"]).dt.year)
    keys = ["orig_year"] if split == "(none)" else ["orig_year", split]
    tr = d.groupby(keys, observed=True).agg(
        default_rate=("target_default", "mean"),
        exposure=("loan_amount", "sum")).reset_index()
    left, right = st.columns(2)
    fig = px.line(tr, x="orig_year", y="default_rate", markers=True,
                  color=None if split == "(none)" else split,
                  labels={"orig_year": "Origination year", "default_rate": "Default rate"},
                  title="Default rate by year the loan was made")
    fig.update_layout(height=340, yaxis_tickformat=".0%")
    left.plotly_chart(fig, use_container_width=True)
    fig = px.bar(tr, x="orig_year", y="exposure",
                 color=None if split == "(none)" else split,
                 labels={"orig_year": "Origination year", "exposure": "Exposure"},
                 title="Exposure by year the loan was made")
    fig.update_layout(height=340)
    right.plotly_chart(fig, use_container_width=True)

    # ---------- Drill-down ----------
    st.markdown("#### Drill-down: accounts in a group")
    a, b = st.columns(2)
    ddim = a.selectbox("Group by", list(DIMS), format_func=DIMS.get, key="dd_dim")
    vals = [v for v in analytics.profile(df, ddim)[ddim]]
    grp = b.selectbox("Group", vals, key="dd_val")
    acc = df[df[ddim] == grp].sort_values("pd_score", ascending=False)
    view = acc[["account_id", "loan_type", "loan_amount", "pd_score",
                "risk_segment", "top_drivers"]].head(200).copy()
    view["loan_amount"] = view["loan_amount"].map(money)
    view["pd_score"] = view["pd_score"].map(pct)
    st.caption(f"{len(acc):,} loans in this group, highest risk first (top 200 shown). "
               "Copy an account ID into the Account tab for details.")
    st.dataframe(view, hide_index=True, use_container_width=True)

    # ---------- Insights ----------
    with st.expander("💡 Key portfolio insights (full portfolio)", expanded=True):
        for i, item in enumerate(analytics.insights(full), 1):
            st.markdown(f"**{i}. {item['title']}**  \n{item['finding']}  \n"
                        f"*So what:* {item['so_what']}")