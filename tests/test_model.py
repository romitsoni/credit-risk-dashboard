"""Model: no leakage, fair inputs, sane metrics, correct risk segments."""
import pandas as pd
import pytest

import model as M

LEAKY_OR_PROTECTED = {"gender", "marital_status", "target_default", "account_id",
                      "customer_id", "customer_since", "has_collections",
                      "origination_date"}


def test_inputs_exclude_leakage_ids_and_protected_attributes():
    for f in M.NUM + M.CAT:
        assert f not in LEAKY_OR_PROTECTED, f
        assert not f.startswith("coll_"), f"collections data is leakage: {f}"


def test_design_matrix_has_no_protected_or_collections_columns(analytic):
    X = M.make_X(analytic.head(500))
    bad = [c for c in X.columns
           if c.startswith(("gender", "marital", "coll_")) or c == "target_default"]
    assert not bad, bad
    assert X.isna().sum().sum() == 0


def test_metrics_function_on_a_hand_checked_example():
    # 2 bad loans (0.7, 0.3) vs 2 good loans (0.4, 0.1): 3 of 4 pairs ranked right
    m = M.metrics([1, 1, 0, 0], [0.7, 0.3, 0.4, 0.1])
    assert m["AUC"] == pytest.approx(0.75)
    assert m["Gini"] == pytest.approx(0.5)


def test_segment_boundaries():
    s = M.segment(pd.Series([0.0, 0.10, 0.1001, 0.20, 0.35, 0.3501, 1.0])).astype(str).tolist()
    assert s == ["Low", "Low", "Medium", "Medium", "High", "Very High", "Very High"]


def test_reported_metrics_are_believable(metrics):
    for name in ["logistic_regression", "xgboost"]:
        m = metrics[name]
        assert 0.55 < m["AUC"] < 0.90, f"{name}: AUC above 0.90 would suggest leakage"
        assert m["Gini"] == pytest.approx(2 * m["AUC"] - 1, abs=1e-3)
        assert 0 < m["KS"] < 1


def test_scores_are_probabilities(scored):
    assert scored["pd_score"].between(0, 1).all()
    assert scored["pd_score_lr"].between(0, 1).all()
    assert scored["risk_segment"].notna().all()


def test_no_customer_in_both_train_and_test(scored):
    sides = scored.groupby("customer_id")["in_test_set"].nunique()
    assert (sides == 1).all()


def test_segments_rank_in_order_on_unseen_data(scored):
    test = scored[scored["in_test_set"] == 1]
    rate = test.groupby("risk_segment")["target_default"].mean()
    order = ["Low", "Medium", "High", "Very High"]
    values = [rate[s] for s in order if s in rate.index]
    assert values == sorted(values)


def test_average_pd_close_to_actual_default_rate(scored):
    assert abs(scored["pd_score"].mean() - scored["target_default"].mean()) < 0.02