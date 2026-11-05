"""
collections_analysis.py
Collections strategy (a "C-score" view), separate from the default model:
which actions recover money, and which customers pay when in collections.

Run:  python src/collections_analysis.py > collections_analysis.txt 2>&1
Needs: python src/model.py to have run (uses scored_accounts.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = Path("data/raw")
DATA = Path("data/processed")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

LEVEL = {"Reminder Call": 1, "SMS/Email Notice": 1, "Field Visit": 2,
         "Settlement Offer": 3, "Legal Notice": 4, "Written-Off Referral": 5}


def title(t):
    print("\n" + "=" * 75 + "\n" + t + "\n" + "=" * 75)


def pay_table(df, by):
    t = df.groupby(by, observed=True).agg(
        accounts=("account_id", "count"),
        pay_rate=("any_paid", "mean"),
        avg_recovered=("total_recovered", "mean"),
        default_rate=("target_default", "mean"))
    return t.round(3).sort_values("pay_rate", ascending=False)


def main():
    e = pd.read_csv(RAW / "collections_events.csv")
    s = pd.read_csv(DATA / "scored_accounts.csv")
    e["event_date"] = pd.to_datetime(e["event_date"])
    e["paid"] = (e["amount_recovered"] > 0).astype(int)
    e["level"] = e["action_type"].map(LEVEL)

    # ---------------------------------------------------------------
    title("1. IS THE COLLECTIONS DATA CONSISTENT?")
    print(f"Events: {len(e):,} on {e['account_id'].nunique():,} accounts")
    print("\nEvents per account:")
    print(e.groupby("account_id").size().value_counts().sort_index())
    print("\nOutcome label vs money actually recovered (share of events):")
    ct = pd.crosstab(e["outcome"], e["paid"], normalize="index").round(3)
    ct.columns = ["no money (amount = 0)", "money recovered (> 0)"]
    print(ct)
    pr0 = ((e["outcome"] == "Payment Received") & (e["paid"] == 0)).mean()
    nr1 = ((e["outcome"] == "No Response") & (e["paid"] == 1)).mean()
    print(f"\n'Payment Received' but amount = 0: {pr0:.1%} of all events")
    print(f"'No Response' but money recovered: {nr1:.1%} of all events")

    e = e.sort_values(["account_id", "event_date"])
    multi = e.groupby("account_id").filter(lambda g: len(g) > 1)
    logical = multi.groupby("account_id")["level"].apply(lambda s: s.is_monotonic_increasing)
    after_wo = multi.groupby("account_id").apply(
        lambda g: (g["level"].values[:-1] == 5).any(), include_groups=False)
    print(f"\nAccounts with 2+ actions: {len(logical):,}")
    print(f"  Actions escalate in a logical order: {logical.mean():.1%}")
    print(f"  Another action comes AFTER a write-off referral: {after_wo.mean():.1%}")
    print("-> Source of truth for 'paid' = amount_recovered > 0 (money is a fact;")
    print("   the outcome label is typed by staff). Event order is not reliable.")

    # ---------------------------------------------------------------
    title("2. WHICH ACTION WORKS BEST? (event level)")
    act = e.groupby("action_type").agg(
        events=("collection_event_id", "count"),
        pay_rate=("paid", "mean"),
        avg_when_paid=("amount_recovered", lambda x: x[x > 0].mean()),
        total_recovered=("amount_recovered", "sum")).round(3)
    act["share_of_money"] = (act["total_recovered"] / act["total_recovered"].sum()).round(3)
    print(act.sort_values("pay_rate", ascending=False))
    print("\nOutcome mix by action (share of events):")
    print(pd.crosstab(e["action_type"], e["outcome"], normalize="index").round(3))

    # ---------------------------------------------------------------
    a = e.groupby("account_id").agg(
        events=("collection_event_id", "count"),
        any_paid=("paid", "max"),
        total_recovered=("amount_recovered", "sum"),
        any_promise=("outcome", lambda x: (x == "Promise to Pay").any()),
        max_level=("level", "max")).reset_index()
    d = a.merge(s, on="account_id", how="left")
    d["rec_vs_balance"] = d["total_recovered"] / d["rep_last_balance"]

    title("3. WHICH CUSTOMERS PAY WHEN IN COLLECTIONS? (account level)")
    print(f"Accounts in collections: {len(d):,} | paid something: {d['any_paid'].mean():.1%} | "
          f"total recovered: {d['total_recovered'].sum():,.0f}")
    print(f"Median recovered as share of remaining balance (payers only): "
          f"{d.loc[d['any_paid'] == 1, 'rec_vs_balance'].median():.1%}")
    for col in ["employment_type", "income_band", "age_band", "loan_type",
                "risk_segment", "region"]:
        print(f"\nBy {col}:")
        print(pay_table(d, col))

    # ---------------------------------------------------------------
    title("4. DOES OUR DEFAULT RISK SCORE PREDICT WHO PAYS IN COLLECTIONS?")
    auc = roc_auc_score(d["any_paid"], -d["pd_score"])
    print(f"AUC of (low PD -> pays): {auc:.3f}  (0.5 = no link)")
    d["pd_band"] = pd.qcut(d["pd_score"], 5, labels=["Q1 lowest PD", "Q2", "Q3", "Q4", "Q5 highest PD"])
    print(pay_table(d, "pd_band").sort_index())
    print("-> If AUC is near 0.5, a separate collections score (C-score) is needed.")

    # ---------------------------------------------------------------
    title("5. DOES PAYING IN COLLECTIONS SAVE THE LOAN? (cure check)")
    print(d.groupby("any_paid")["target_default"].agg(["count", "mean"])
          .rename(index={0: "never paid in collections", 1: "paid in collections"}).round(3))
    print("\nBy number of actions on the account:")
    print(d.groupby(d["events"].clip(upper=4))["target_default"].agg(["count", "mean"])
          .rename(index={4: "4+"}).round(3))
    print("\nBy highest action reached (1 = reminder ... 5 = write-off referral):")
    print(d.groupby("max_level")[["any_paid", "target_default"]].mean().round(3))

    # ---------------------------------------------------------------
    title("6. BEST ACTION FOR EACH TYPE OF CUSTOMER (pay rate per event)")
    ev = e.merge(s[["account_id", "risk_segment", "employment_type", "income_band"]],
                 on="account_id", how="left")
    for col in ["risk_segment", "employment_type", "income_band"]:
        t = pd.crosstab(ev[col], ev["action_type"], values=ev["paid"], aggfunc="mean").round(3)
        t["best_action"] = t.idxmax(axis=1)
        print(f"\nBy {col}:")
        print(t)
    print("\nNote: groups with few events are noisy; check the counts before acting.")


if __name__ == "__main__":
    main()