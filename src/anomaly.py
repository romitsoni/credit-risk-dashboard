"""
anomaly.py
Flags accounts with unusual repayment or bureau patterns, compared with
(a) their OWN history and (b) their PEER segment (loan type x income band).
Every flag has a plain-English reason.

Input : data/raw/repayment_history.csv, data/processed/scored_accounts.csv
Output: data/processed/anomalies.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

RAW = Path("data/raw")
DATA = Path("data/processed")
PEER = ["loan_type", "income_band"]


def own_history_features(r):
    """Compare months 1-9 (the account's normal) with months 10-12 (recent)."""
    r = r.copy()
    r["is_missed"] = (r["payment_status"] == "Missed").astype(int)
    early = r[r["payment_month"] <= 9].groupby("account_id").agg(
        early_max_dpd=("days_past_due", "max"),
        early_mean_dpd=("days_past_due", "mean"))
    recent = r[r["payment_month"] >= 10].groupby("account_id").agg(
        recent_mean_dpd=("days_past_due", "mean"),
        recent_missed=("is_missed", "sum"))
    return early.join(recent).reset_index()


def main():
    df = pd.read_csv(DATA / "scored_accounts.csv")
    r = pd.read_csv(RAW / "repayment_history.csv")
    df = df.merge(own_history_features(r), on="account_id", how="left")

    flags = {}    # rule name -> boolean Series
    reasons = {}  # rule name -> Series of reason text

    # ---------- A. Own history ----------
    # A1: clean for 9 months, then missed in the last 3
    m = (df["early_max_dpd"] == 0) & (df["recent_missed"] >= 1)
    flags["sudden_miss"] = m
    reasons["sudden_miss"] = ("No late payment in months 1-9, then missed "
                              + df["recent_missed"].astype(int).astype(str)
                              + " of the last 3 months")

    # A2: average DPD jumped by 30+ days vs own normal
    jump = df["recent_mean_dpd"] - df["early_mean_dpd"]
    m = jump >= 30
    flags["dpd_jump"] = m
    reasons["dpd_jump"] = ("Average DPD rose from "
                           + df["early_mean_dpd"].round(0).astype(int).astype(str)
                           + " to " + df["recent_mean_dpd"].round(0).astype(int).astype(str)
                           + " days (months 1-9 vs 10-12)")

    # ---------- B. Peer segment ----------
    for col, label, fmt in [
        ("bur_utilization", "Bureau utilization", lambda s: (s * 100).round(0).astype(int).astype(str) + "%"),
        ("bur_mean_dpd", "Average bureau DPD", lambda s: s.round(0).astype(int).astype(str) + " days"),
    ]:
        p99 = df.groupby(PEER)[col].transform(lambda s: s.quantile(0.99))
        m = (df[col] > p99) & (df[col] > 0)
        name = f"peer_{col}"
        flags[name] = m
        reasons[name] = (label + " " + fmt(df[col]) + " is above the top 1% of peers ("
                         + df["loan_type"] + ", " + df["income_band"].astype(str)
                         + ": " + fmt(p99) + ")")

    # ---------- C. Unusual combination (Isolation Forest) ----------
    iso_cols = ["rep_mean_dpd", "rep_max_dpd", "rep_missed", "rep_dpd_trend",
                "rep_pay_ratio", "bur_utilization", "bur_mean_dpd", "bur_writeoffs"]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    df["iso_score"] = -iso.fit(df[iso_cols]).score_samples(df[iso_cols])
    flags["isolation_forest"] = pd.Series(iso.predict(df[iso_cols]) == -1, index=df.index)
    reasons["isolation_forest"] = pd.Series(
        "Rare combination of repayment and bureau behaviour (Isolation Forest, top 2%)",
        index=df.index)

    # ---------- Combine ----------
    F = pd.DataFrame(flags).fillna(False)
    df["flag_count"] = F.sum(axis=1)
    df["flags"] = F.apply(lambda row: ", ".join(row.index[row]), axis=1)
    df["reasons"] = [
        " | ".join(reasons[k].iloc[i] for k in F.columns if F[k].iloc[i])
        for i in range(len(df))]

    out = df[df["flag_count"] > 0].sort_values(
        ["flag_count", "pd_score"], ascending=False)
    cols = ["account_id", "customer_id", "loan_type", "income_band",
            "employment_type", "region", "loan_amount", "pd_score",
            "risk_segment", "target_default", "flag_count", "flags", "reasons"]
    out[cols].to_csv(DATA / "anomalies.csv", index=False)

    # ---------- Report ----------
    print("Accounts flagged:", len(out), "of", len(df))
    print("\nFlags per rule:")
    print(F.sum().to_string())
    print("\nDefault rate - flagged vs not flagged (sanity check):")
    print(df.groupby(df["flag_count"] > 0)["target_default"].agg(["count", "mean"]).round(3))
    print("\nTop 5 flagged accounts:")
    print(out[["account_id", "flag_count", "pd_score", "reasons"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()