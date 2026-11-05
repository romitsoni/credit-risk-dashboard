"""Anomaly detection: every flag has a reason, rules are known, flags find risk."""
import pandas as pd

import anomaly

RULES = {"sudden_miss", "dpd_jump", "peer_bur_utilization",
         "peer_bur_mean_dpd", "isolation_forest"}


def test_every_flagged_account_has_a_reason(anomalies):
    assert len(anomalies) > 0
    assert (anomalies["flag_count"] >= 1).all()
    assert anomalies["reasons"].str.len().gt(0).all()


def test_flag_count_matches_flags_and_rules_are_known(anomalies):
    for flags, n in zip(anomalies["flags"], anomalies["flag_count"]):
        names = flags.split(", ")
        assert len(names) == n
        assert set(names) <= RULES


def test_review_list_is_sorted_worst_first(anomalies):
    assert anomalies["flag_count"].is_monotonic_decreasing


def test_flags_point_to_higher_risk(anomalies, scored):
    rest = scored[~scored["account_id"].isin(anomalies["account_id"])]
    assert anomalies["target_default"].mean() > rest["target_default"].mean()


def test_own_history_split_months_1_9_vs_10_12():
    r = pd.DataFrame({
        "account_id": ["A1"] * 12, "payment_month": range(1, 13),
        "days_past_due": [0] * 9 + [40, 0, 50],
        "payment_status": ["On-Time"] * 9 + ["Missed", "On-Time", "Missed"],
    })
    h = anomaly.own_history_features(r).iloc[0]
    assert h["early_max_dpd"] == 0          # clean history ...
    assert h["recent_missed"] == 2          # ... then 2 misses -> "sudden miss" rule
    assert h["recent_mean_dpd"] - h["early_mean_dpd"] >= 30   # "DPD jump" rule