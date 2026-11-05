"""Data ingestion: one row per loan, clean values, documented decisions."""
import numpy as np
import pandas as pd

import ingest


# ---------- On the real data ----------
def test_one_row_per_loan(analytic, raw_loans):
    assert len(analytic) == len(raw_loans) == 14685
    assert analytic["account_id"].is_unique


def test_every_loan_has_its_target(analytic):
    assert analytic["target_default"].isin([0, 1]).all()
    assert abs(analytic["target_default"].mean() - 0.161) < 0.005


def test_gender_labels_are_clean(analytic):
    assert set(analytic["gender"].unique()) <= {"Male", "Female"}


def test_years_employed_never_impossible(analytic):
    cap = (analytic["age"] - 16).clip(lower=0)
    assert (analytic["years_employed"] <= cap + 1e-9).all()


def test_model_features_have_no_missing_values(analytic):
    import model as M
    missing = analytic[M.NUM].isna().sum()
    assert missing.sum() == 0, missing[missing > 0]


def test_missing_flags_are_zero_or_one(analytic):
    # bur_dpd_missing is a COUNT per customer (lines with missing DPD), not a 0/1 flag
    flags = [c for c in analytic.columns
             if c.endswith("_missing") and c != "bur_dpd_missing"]
    assert flags, "missing-value flags should exist"
    for c in flags:
        assert analytic[c].isin([0, 1]).all(), c
    counts = analytic["bur_dpd_missing"]
    assert (counts >= 0).all() and (counts == counts.round()).all()


def test_data_quality_log_is_written(processed):
    log = pd.read_csv(processed / "data_quality_log.csv")
    assert len(log) >= 8
    assert (log["rows_affected"] >= 0).all()
    assert log["issue"].str.contains("gender").any()


# ---------- Edge cases on tiny made-up data ----------
def tiny_customers():
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(8)],
        "gender": ["M", "F", "Male", "Female", "M", "F", "Male", "Female"],
        "age": [22, 30, 40, 50, 25, 35, 45, 55],
        "years_employed": [10.0, np.nan, 5.0, 20.0, 1.0, 3.0, np.nan, 8.0],
        "num_dependents": [0, 1, np.nan, 2, 0, 1, 2, 3],
        "years_at_residence": [1.0, 2.0, 3.0, np.nan, 1.0, 2.0, 3.0, 4.0],
        "employment_type": ["Salaried"] * 4 + ["Contract"] * 4,
        "annual_income": [20000, 40000, 60000, 80000, 30000, 50000, 70000, 90000],
        "customer_since": ["2020-01-01"] * 8,
    })


def test_clean_customers_fixes_labels_caps_and_fills():
    c = ingest.clean_customers(tiny_customers())
    assert set(c["gender"]) == {"Male", "Female"}
    # age 22 with 10 years employed is impossible -> capped at 6, and flagged
    assert c.loc[0, "years_employed"] == 6
    assert c.loc[0, "years_employed_capped"] == 1
    # missing values are filled AND flagged, no rows dropped
    assert len(c) == 8
    assert c[["years_employed", "num_dependents", "years_at_residence"]].notna().all().all()
    assert c.loc[1, "years_employed_missing"] == 1
    assert c.loc[2, "num_dependents_missing"] == 1


def test_repayment_summary_for_one_loan():
    status = ["On-Time"] * 9 + ["Missed", "Late", "Missed"]
    dpd = [0] * 9 + [45, 10, 60]
    r = pd.DataFrame({
        "account_id": ["A1"] * 12, "payment_month": range(1, 13),
        "emi_due": [100.0] * 12,
        "emi_paid": [100.0] * 9 + [0.0, 100.0, 0.0],
        "days_past_due": dpd, "payment_status": status,
        "outstanding_balance": np.linspace(1200, 100, 12),
    })
    g = ingest.repayment_features(r).iloc[0]
    assert len(ingest.repayment_features(r)) == 1          # 12 rows -> 1 row
    assert g["rep_months"] == 12
    assert g["rep_missed"] == 2 and g["rep_late"] == 1
    assert g["rep_max_dpd"] == 60
    assert g["rep_ever_60dpd"] == 1
    assert abs(g["rep_pay_ratio"] - 10 / 12) < 1e-9
    assert g["rep_dpd_trend"] > 0                          # getting worse
    assert g["rep_missed_last3"] == 2