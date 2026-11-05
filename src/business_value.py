"""
business_value.py
Shows what the model means in money and in daily decisions.

 1. Expected loss = chance of default x share of money lost x loan amount
 2. Where the loss is: how much of it sits in the riskiest loans
 3. Cut-off: if we review every loan above a risk level, what do we catch?

Assumptions (you can change them below):
 - Loan amount = the money at risk (the data has no current balance).
 - Share of money lost when a loan defaults:
     45% for loans with no collateral (a common banking reference value),
     25% for Mortgage and Auto loans, because the house or car can be sold.

Run:  python src/business_value.py > reports/business_value.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed")
LOSS_NO_COLLATERAL = 0.45
LOSS_WITH_COLLATERAL = 0.25
CUTOFFS = [0.10, 0.15, 0.20, 0.25, 0.35]
ORDER = ["Low", "Medium", "High", "Very High"]


def title(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def m(v):
    return f"{v / 1e6:,.1f}M"


def main():
    s = pd.read_csv(DATA / "scored_accounts.csv")
    s["loss_share"] = np.where(s["collateral_flag"] == 1,
                               LOSS_WITH_COLLATERAL, LOSS_NO_COLLATERAL)
    s["expected_loss"] = s["pd_score"] * s["loss_share"] * s["loan_amount"]
    s["actual_loss"] = s["target_default"] * s["loss_share"] * s["loan_amount"]

    # ---------- 1. Expected loss ----------
    title("1. EXPECTED LOSS")
    total = s["loan_amount"].sum()
    el = s["expected_loss"].sum()
    print(f"Total money lent:              {m(total)}")
    print(f"Expected loss (from the model): {m(el)}  ({el / total:.2%} of money lent)")
    print(f"Loss from the real defaults:    {m(s['actual_loss'].sum())}")

    seg = s.groupby("risk_segment").agg(
        loans=("account_id", "count"), money_lent=("loan_amount", "sum"),
        expected_loss=("expected_loss", "sum"),
        real_loss=("actual_loss", "sum")).reindex(ORDER)
    seg["share_of_expected_loss"] = (seg["expected_loss"] / el).map("{:.1%}".format)
    for c in ["money_lent", "expected_loss", "real_loss"]:
        seg[c] = seg[c].map(m)
    print("\nBy risk segment:")
    print(seg)

    lt = s.groupby("loan_type").agg(money_lent=("loan_amount", "sum"),
                                    expected_loss=("expected_loss", "sum"))
    lt = lt.sort_values("expected_loss", ascending=False)
    lt["share_of_expected_loss"] = (lt["expected_loss"] / el).map("{:.1%}".format)
    lt["money_lent"] = lt["money_lent"].map(m)
    lt["expected_loss"] = lt["expected_loss"].map(m)
    print("\nBy loan type:")
    print(lt)

    # ---------- 2. Where the loss is ----------
    title("2. WHERE THE LOSS IS")
    s = s.sort_values("pd_score", ascending=False).reset_index(drop=True)
    for top in [0.10, 0.20, 0.30]:
        k = int(len(s) * top)
        e = s.loc[:k - 1, "expected_loss"].sum() / el
        a = s.loc[:k - 1, "actual_loss"].sum() / s["actual_loss"].sum()
        print(f"The riskiest {top:.0%} of loans hold {e:.1%} of expected loss "
              f"and {a:.1%} of the real loss")

    # ---------- 3. Cut-off ----------
    title("3. CUT-OFF: REVIEW EVERY LOAN ABOVE A RISK LEVEL (test set only)")
    t = s[s["in_test_set"] == 1]
    bad = t["target_default"] == 1
    rows = []
    for c in CUTOFFS:
        flag = t["pd_score"] >= c
        caught = int((flag & bad).sum())
        false_alarm = int((flag & ~bad).sum())
        missed = int((~flag & bad).sum())
        rows.append({
            "review_if_risk_at_least": f"{c:.0%}",
            "loans_to_review": f"{flag.mean():.1%}",
            "bad_loans_caught": caught,
            "false_alarms": false_alarm,
            "bad_loans_missed": missed,
            "share_of_reviewed_that_are_bad": f"{caught / max(caught + false_alarm, 1):.1%}",
            "share_of_bad_loans_caught": f"{caught / max(caught + missed, 1):.1%}",
            "share_of_money_lost_caught": f"{t.loc[flag, 'actual_loss'].sum() / t['actual_loss'].sum():.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nIn plain words:")
    print(" - bad_loans_caught: loans that really defaulted and were flagged (good).")
    print(" - false_alarms: good loans that were flagged anyway (extra review work).")
    print(" - bad_loans_missed: loans that defaulted but were not flagged (lost money).")
    print(" A missed bad loan costs much more than one extra review.")


if __name__ == "__main__":
    main()