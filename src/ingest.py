"""
ingest.py
Reads the 5 raw CSVs, cleans them, builds features, and joins them
into ONE row per loan account.

Input : data/raw/*.csv        (never changed)
Output: data/processed/analytic_dataset.csv
        data/processed/data_quality_log.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DQ = []  # data-quality log: every decision we make


def log(table, issue, action, rows):
    DQ.append({"table": table, "issue": issue,
               "action": action, "rows_affected": int(rows)})


# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
def load():
    names = ["customers", "loan_accounts", "bureau_credit",
             "repayment_history", "collections_events"]
    return {n: pd.read_csv(RAW / f"{n}.csv") for n in names}


# ------------------------------------------------------------------
# 2. CLEAN CUSTOMERS
# ------------------------------------------------------------------
def clean_customers(c):
    c = c.copy()

    # Gender: M/F -> Male/Female
    bad = c["gender"].isin(["M", "F"]).sum()
    c["gender"] = c["gender"].replace({"M": "Male", "F": "Female"})
    log("customers", "gender coded as M/F", "mapped to Male/Female", bad)

    # years_employed cannot be more than (age - 16)
    cap = (c["age"] - 16).clip(lower=0)
    too_high = c["years_employed"] > cap
    c["years_employed_capped"] = too_high.astype(int)
    c.loc[too_high, "years_employed"] = cap[too_high]
    log("customers", "years_employed > age - 16",
        "capped at age - 16 + capped flag", too_high.sum())

    # Missing values: fill with median of same employment type + flag
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        n = c[col].isna().sum()
        c[f"{col}_missing"] = c[col].isna().astype(int)
        c[col] = c[col].fillna(
            c.groupby("employment_type")[col].transform("median"))
        log("customers", f"{col} missing",
            "filled with employment-type median + missing flag", n)

    # Filled values must also respect the age cap
    c["years_employed"] = np.minimum(c["years_employed"], cap)

    c["customer_since"] = pd.to_datetime(c["customer_since"])

    # Bands for analytics
    c["age_band"] = pd.cut(c["age"], [0, 30, 40, 50, 60, 200], right=False,
                           labels=["<30", "30-39", "40-49", "50-59", "60+"])
    c["income_band"] = pd.qcut(c["annual_income"], 4,
                               labels=["Q1 Low", "Q2", "Q3", "Q4 High"])
    return c


# ------------------------------------------------------------------
# 3. CLEAN LOANS
# ------------------------------------------------------------------
def clean_loans(l):
    l = l.copy()
    n = l["interest_rate_pct"].isna().sum()
    l["interest_rate_missing"] = l["interest_rate_pct"].isna().astype(int)
    l["interest_rate_pct"] = l["interest_rate_pct"].fillna(
        l.groupby("loan_type")["interest_rate_pct"].transform("median"))
    log("loan_accounts", "interest_rate_pct missing",
        "filled with loan-type median + missing flag", n)
    l["origination_date"] = pd.to_datetime(l["origination_date"])
    return l


# ------------------------------------------------------------------
# 4. BUREAU FEATURES (many lines -> 1 row per customer)
# ------------------------------------------------------------------
def bureau_features(b):
    b = b.copy()
    log("bureau_credit", "date_closed missing",
        "kept as null: these are Active lines", b["date_closed"].isna().sum())
    log("bureau_credit", "days_past_due missing",
        "left null in max/mean, counted per customer",
        b["days_past_due"].isna().sum())

    b["is_active"] = (b["account_status"] == "Active").astype(int)
    b["is_writeoff"] = (b["account_status"] == "Written-Off").astype(int)
    b["dpd_missing"] = b["days_past_due"].isna().astype(int)
    b["is_nbfc_fintech"] = b["lender_category"].isin(
        ["NBFC", "Fintech"]).astype(int)

    g = b.groupby("customer_id").agg(
        bur_lines=("bureau_id", "count"),
        bur_active=("is_active", "sum"),
        bur_writeoffs=("is_writeoff", "sum"),
        bur_sanctioned=("sanctioned_amount", "sum"),
        bur_outstanding=("outstanding_amount", "sum"),
        bur_max_dpd=("days_past_due", "max"),
        bur_mean_dpd=("days_past_due", "mean"),
        bur_dpd_missing=("dpd_missing", "sum"),
        bur_nbfc_fintech_share=("is_nbfc_fintech", "mean"),
    ).reset_index()

    g["bur_utilization"] = np.where(
        g["bur_sanctioned"] > 0, g["bur_outstanding"] / g["bur_sanctioned"], 0)
    g["bur_has_writeoff"] = (g["bur_writeoffs"] > 0).astype(int)
    return g


# ------------------------------------------------------------------
# 5. REPAYMENT FEATURES (12 months -> 1 row per loan)
# ------------------------------------------------------------------
def repayment_features(r):
    r = r.sort_values(["account_id", "payment_month"]).copy()
    r["is_late"] = (r["payment_status"] == "Late").astype(int)
    r["is_missed"] = (r["payment_status"] == "Missed").astype(int)
    r["dpd30"] = (r["days_past_due"] >= 30).astype(int)
    r["dpd60"] = (r["days_past_due"] >= 60).astype(int)
    r["from_end"] = r.groupby("account_id")["payment_month"].rank(
        ascending=False)

    g = r.groupby("account_id").agg(
        rep_months=("payment_month", "count"),
        rep_late=("is_late", "sum"),
        rep_missed=("is_missed", "sum"),
        rep_max_dpd=("days_past_due", "max"),
        rep_mean_dpd=("days_past_due", "mean"),
        rep_ever_30dpd=("dpd30", "max"),
        rep_ever_60dpd=("dpd60", "max"),
        rep_due=("emi_due", "sum"),
        rep_paid=("emi_paid", "sum"),
        rep_last_balance=("outstanding_balance", "last"),
    ).reset_index()
    g["rep_pay_ratio"] = g["rep_paid"] / g["rep_due"]

    first3 = r[r["payment_month"] <= 3].groupby("account_id")["days_past_due"].mean()
    last3 = r[r["from_end"] <= 3].groupby("account_id")["days_past_due"].mean()
    missed_last3 = r[r["from_end"] <= 3].groupby("account_id")["is_missed"].sum()

    g["rep_dpd_first3"] = g["account_id"].map(first3)
    g["rep_dpd_last3"] = g["account_id"].map(last3)
    g["rep_dpd_trend"] = g["rep_dpd_last3"] - g["rep_dpd_first3"]
    g["rep_missed_last3"] = g["account_id"].map(missed_last3)
    return g.drop(columns=["rep_due", "rep_paid"])


# ------------------------------------------------------------------
# 6. COLLECTIONS FEATURES (dashboard / anomaly ONLY - not in model)
# ------------------------------------------------------------------
def collections_features(e):
    e = e.copy()
    e["event_date"] = pd.to_datetime(e["event_date"])
    g = e.groupby("account_id").agg(
        coll_events=("collection_event_id", "count"),
        coll_recovered=("amount_recovered", "sum"),
        coll_last_date=("event_date", "max"),
    ).reset_index()
    last = e.sort_values("event_date").groupby("account_id").tail(1)
    g = g.merge(
        last[["account_id", "action_type", "outcome"]].rename(
            columns={"action_type": "coll_last_action",
                     "outcome": "coll_last_outcome"}),
        on="account_id", how="left")
    return g


# ------------------------------------------------------------------
# 7. JOIN EVERYTHING -> 1 row per loan
# ------------------------------------------------------------------
def build():
    t = load()
    c = clean_customers(t["customers"])
    l = clean_loans(t["loan_accounts"])
    b = bureau_features(t["bureau_credit"])
    r = repayment_features(t["repayment_history"])
    e = collections_features(t["collections_events"])

    # Base table = loans (it holds the target). Left join keeps every loan.
    df = l.merge(c, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(b, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(r, on="account_id", how="left", validate="one_to_one")
    df = df.merge(e, on="account_id", how="left", validate="one_to_one")

    # Loan dated before customer_since -> field is unreliable
    before = (df["origination_date"] < df["customer_since"]).sum()
    log("joined", "loan origination before customer_since",
        "customer_since not used as a model feature", before)

    # Customers with no bureau record
    no_bur = df["bur_lines"].isna().sum()
    df["bur_no_history"] = df["bur_lines"].isna().astype(int)
    bur_cols = [x for x in b.columns if x != "customer_id"]
    df[bur_cols] = df[bur_cols].fillna(0)
    log("joined", "loan with no bureau record",
        "bureau features set to 0 + no-history flag", no_bur)

    # Loans with no collections
    df["coll_events"] = df["coll_events"].fillna(0)
    df["coll_recovered"] = df["coll_recovered"].fillna(0)
    df["has_collections"] = (df["coll_events"] > 0).astype(int)

    # Extra features
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["customer_loans"] = df.groupby("customer_id")["account_id"].transform("count")

    # Safety check: joins must not add or lose loans
    assert len(df) == len(l), "row count changed after joins"

    df.to_csv(OUT / "analytic_dataset.csv", index=False)
    pd.DataFrame(DQ).to_csv(OUT / "data_quality_log.csv", index=False)

    print("Final shape:", df.shape)
    print("Default rate:", round(df["target_default"].mean(), 4))
    print("\nColumns with nulls left (expected: collections fields only):")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nData-quality log:")
    print(pd.DataFrame(DQ).to_string(index=False))
    return df


if __name__ == "__main__":
    build()