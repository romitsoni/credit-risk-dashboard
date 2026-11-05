"""Analytics: the numbers behind the dashboard and the AI assistant add up."""
import pytest

import analytics


def test_profile_adds_up(scored):
    t = analytics.profile(scored, "loan_type")
    assert t["accounts"].sum() == len(scored)
    assert t["exposure_share"].sum() == pytest.approx(1.0)
    assert t["default_rate"].between(0, 1).all()


def test_profile_works_on_a_filtered_subset(scored):
    sub = scored[scored["loan_type"] == "Mortgage"]
    t = analytics.profile(sub, "region")
    assert t["accounts"].sum() == len(sub)


def test_ordered_dimensions_keep_their_order(scored):
    t = analytics.profile(scored, "income_band")
    assert list(t["income_band"].astype(str)) == analytics.ORDER["income_band"]


def test_cross_cut_has_expected_shape(scored):
    x = analytics.cross(scored, "income_band", "employment_type")
    assert list(x.index) == analytics.ORDER["income_band"]
    assert x.shape[1] == scored["employment_type"].nunique()
    assert ((x >= 0) & (x <= 1)).all().all()


def test_kpis(scored):
    k = analytics.portfolio_kpis(scored)
    assert k["accounts"] == len(scored)
    assert k["customers"] == scored["customer_id"].nunique()
    assert 0 <= k["high_risk_exposure_share"] <= 1


def test_insights_have_finding_and_so_what(scored):
    ins = analytics.insights(scored)
    assert 4 <= len(ins) <= 5
    for i in ins:
        assert i["title"] and i["finding"] and i["so_what"]