"""
deep_analysis.py
Deeper data understanding + tests of new feature ideas.
Ideas from the Kaggle Home Credit winners (recency, change over time,
worst behaviour across accounts) and bank scorecards (Information Value).

Run:  python src/deep_analysis.py > deep_analysis.txt 2>&1
Needs: python src/ingest.py to have run first.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # reuse the same feature list and make_X()

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
RAW = Path("data/raw")
DATA = Path("data/processed")


def title(t):
    print("\n" + "=" * 75 + "\n" + t + "\n" + "=" * 75)


# ------------------------------------------------------------------
# Helpers: Information Value and single-feature AUC
# ------------------------------------------------------------------
def info_value(x, y, bins=10):
    """IV = sum((%good - %bad) * ln(%good / %bad)) over bins."""
    d = pd.DataFrame({"x": x, "y": y})
    if pd.api.types.is_numeric_dtype(d["x"]) and d["x"].nunique() > bins:
        d["b"] = pd.qcut(d["x"].fillna(-999).rank(method="first"), bins, labels=False)
    else:
        d["b"] = d["x"].astype(str)
    g = d.groupby("b")["y"].agg(["count", "sum"])
    g["good"] = g["count"] - g["sum"]
    pb = (g["sum"] + 0.5) / (g["sum"].sum() + 0.5)
    pg = (g["good"] + 0.5) / (g["good"].sum() + 0.5)
    return float(((pg - pb) * np.log(pg / pb)).sum())


def iv_label(v):
    if v < 0.02: return "useless"
    if v < 0.10: return "weak"
    if v < 0.30: return "medium"
    if v < 0.50: return "strong"
    return "SUSPICIOUS (check leakage)"


def single_auc(x, y):
    x = pd.Series(x).astype(float)
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


# ------------------------------------------------------------------
# New feature ideas
# ------------------------------------------------------------------
def repayment_extra(r):
    r = r.sort_values(["account_id", "payment_month"]).copy()
    r["late"] = (r["days_past_due"] > 0).astype(int)
    r["missed"] = (r["payment_status"] == "Missed").astype(int)
    g = r.groupby("account_id")

    prev = g["late"].shift()
    r["blk"] = ((r["late"] != prev) | prev.isna()).cumsum()
    streak = r[r["late"] == 1].groupby(["account_id", "blk"]).size().groupby(level=0).max()
    first_late = r[r["late"] == 1].groupby("account_id")["payment_month"].min()
    last_miss = r[r["missed"] == 1].groupby("account_id")["payment_month"].max()
    changes = (r["payment_status"] != g["payment_status"].shift()).groupby(r["account_id"]).sum() - 1
    recency = ((r["days_past_due"] * r["payment_month"]).groupby(r["account_id"]).sum()
               / r["payment_month"].groupby(r["account_id"]).sum())
    emi = g["emi_due"].mean()
    bal_first = r[r["payment_month"] == 1].set_index("account_id")["outstanding_balance"]
    bal_last = g["outstanding_balance"].last()

    out = pd.DataFrame({
        "x_late_streak": streak, "x_first_late_month": first_late,
        "x_last_miss": last_miss, "x_status_changes": changes,
        "x_recency_dpd": recency, "x_emi": emi,
        "x_bal_first": bal_first, "x_bal_last": bal_last,
    })
    out["x_late_streak"] = out["x_late_streak"].fillna(0)
    out["x_first_late_month"] = out["x_first_late_month"].fillna(13)
    out["x_months_since_miss"] = (12 - out["x_last_miss"]).fillna(12)
    out["x_paydown"] = 1 - out["x_bal_last"] / out["x_bal_first"]
    out = out.drop(columns=["x_last_miss", "x_bal_first", "x_bal_last"])
    return out.rename_axis("account_id").reset_index()


def bureau_timing(l, b):
    """Bureau lines compared with the date OUR loan was given."""
    l = l[["account_id", "customer_id", "origination_date"]].copy()
    l["origination_date"] = pd.to_datetime(l["origination_date"])
    b = b.copy()
    b["date_opened"] = pd.to_datetime(b["date_opened"])
    b["date_closed"] = pd.to_datetime(b["date_closed"])
    m = l.merge(b, on="customer_id", how="inner")
    m["opened_after"] = (m["date_opened"] > m["origination_date"]).astype(int)
    days = (m["origination_date"] - m["date_opened"]).dt.days
    m["recent_12m"] = days.between(0, 365).astype(int)
    m["active_at_orig"] = ((m["date_opened"] <= m["origination_date"])
                           & (m["date_closed"].isna()
                              | (m["date_closed"] > m["origination_date"]))).astype(int)
    m["wo_amount"] = np.where(m["account_status"] == "Written-Off", m["sanctioned_amount"], 0)
    m["active_dpd"] = np.where(m["account_status"] == "Active", m["days_past_due"], np.nan)
    agg = m.groupby("account_id").agg(
        x_bur_opened_after=("opened_after", "sum"),
        x_bur_recent_12m=("recent_12m", "sum"),
        x_bur_active_at_orig=("active_at_orig", "sum"),
        x_bur_wo_amount=("wo_amount", "sum"),
        x_bur_active_max_dpd=("active_dpd", "max"),
    ).reset_index()
    return agg, m


def add_new_features(df, r, l, b):
    rep = repayment_extra(r)
    bur, pairs = bureau_timing(l, b)
    d = df.merge(rep, on="account_id", how="left").merge(bur, on="account_id", how="left")
    for c in ["x_bur_opened_after", "x_bur_recent_12m", "x_bur_active_at_orig",
              "x_bur_wo_amount", "x_bur_active_max_dpd"]:
        d[c] = d[c].fillna(0)
    d["x_emi_to_income"] = d["x_emi"] * 12 / d["annual_income"]
    d["x_debt_to_income"] = (d["bur_outstanding"] + d["rep_last_balance"]) / d["annual_income"]
    d["x_bur_writeoff_share"] = np.where(d["bur_lines"] > 0, d["bur_writeoffs"] / d["bur_lines"], 0)
    cust_missed = d.groupby("customer_id")["rep_missed"].transform("sum")
    d["x_other_loans_missed"] = cust_missed - d["rep_missed"]
    d = d.drop(columns=["x_emi"])
    return d, pairs


NEW = ["x_late_streak", "x_first_late_month", "x_months_since_miss", "x_recency_dpd",
       "x_status_changes", "x_paydown", "x_emi_to_income", "x_debt_to_income",
       "x_bur_writeoff_share", "x_bur_wo_amount", "x_bur_active_max_dpd",
       "x_bur_recent_12m", "x_bur_active_at_orig", "x_bur_opened_after",
       "x_other_loans_missed"]


# ------------------------------------------------------------------
# Cross-validation (5 folds, grouped by customer)
# ------------------------------------------------------------------
def cv_auc(X, y, groups, kind):
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, oof = [], np.zeros(len(y))
    for tr, te in sg.split(X, y, groups):
        if kind == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        else:
            m = xgb.XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.02,
                                  subsample=0.8, colsample_bytree=0.7,
                                  min_child_weight=20, reg_lambda=5, random_state=42)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


# ------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA / "analytic_dataset.csv")
    r = pd.read_csv(RAW / "repayment_history.csv")
    l = pd.read_csv(RAW / "loan_accounts.csv")
    b = pd.read_csv(RAW / "bureau_credit.csv")
    e = pd.read_csv(RAW / "collections_events.csv")
    y = df["target_default"].values
    base = y.mean()

    d, pairs = add_new_features(df, r, l, b)

    title("1. DO A CUSTOMER'S LOANS DEFAULT TOGETHER?")
    multi = d[d["customer_loans"] > 1].copy()
    multi["other_defaults"] = (multi.groupby("customer_id")["target_default"].transform("sum")
                               - multi["target_default"])
    t = multi.groupby(multi["other_defaults"] > 0)["target_default"].agg(["count", "mean"])
    t.index = ["no other loan defaulted", "another loan defaulted"]
    print(t.round(3))
    print(f"Portfolio average: {base:.3f}")
    print("-> If 'another loan defaulted' is much higher, risk is at CUSTOMER level.")
    print("   (Other loans' TARGET is not used as a feature: that would be leakage.)")

    title("2. TIMING CHECKS (hidden leakage)")
    print(f"Bureau lines linked to a loan: {len(pairs):,}")
    print(f"Bureau lines opened AFTER our loan was given: {pairs['opened_after'].mean():.1%}")
    print("-> If high, the bureau file is a CURRENT snapshot, not 'as at approval'.")
    lo = l[["account_id", "origination_date"]].copy()
    lo["origination_date"] = pd.to_datetime(lo["origination_date"])
    ce = e.merge(lo, on="account_id")
    ce["months_after"] = (pd.to_datetime(ce["event_date"]) - ce["origination_date"]).dt.days / 30.4
    print("\nCollections event timing (months after loan start):")
    print(ce["months_after"].describe().round(1))
    print(f"Events BEFORE the loan started: {(ce['months_after'] < 0).mean():.1%}")
    print(f"Events within the 12 repayment months: {ce['months_after'].between(0, 12).mean():.1%}")
    print(f"Events after month 12: {(ce['months_after'] > 12).mean():.1%}")

    title("3. DEFAULT RATE BY TENURE")
    print(d.groupby("tenure_months").agg(loans=("account_id", "count"),
                                         default_rate=("target_default", "mean")).round(3))

    title("4. IS INTEREST RATE LINKED TO RISK? (risk-based pricing)")
    print(f"AUC of interest rate alone: {single_auc(d['interest_rate_pct'], y):.3f}")
    print(d.groupby(["loan_type", "target_default"])["interest_rate_pct"].mean()
          .unstack().round(2).rename(columns={0: "rate_good", 1: "rate_bad"}))

    title("5. FEATURE RANKING: Information Value + single-feature AUC")
    feats = M.NUM + [c for c in M.CAT] + NEW
    rows = []
    for f in feats:
        iv = info_value(d[f], y)
        auc = single_auc(d[f], y) if pd.api.types.is_numeric_dtype(d[f]) else np.nan
        rows.append((f, "NEW" if f in NEW else "", round(iv, 4), iv_label(iv),
                     round(auc, 3) if auc == auc else None))
    rank = pd.DataFrame(rows, columns=["feature", "new", "IV", "strength", "single_AUC"])
    rank = rank.sort_values("IV", ascending=False)
    print(rank.to_string(index=False))
    leak = rank[(rank["IV"] > 0.5) | (rank["single_AUC"].fillna(0) > 0.85)]
    print("\nPossible leakage (IV > 0.5 or single AUC > 0.85):",
          "none" if leak.empty else list(leak["feature"]))
    rank.to_csv(DATA / "feature_ranking.csv", index=False)

    title("6. DOES RISK RISE STEADILY? (default rate by decile of top numeric features)")
    top_num = [f for f in rank["feature"] if pd.api.types.is_numeric_dtype(d[f])
               and d[f].nunique() > 10][:6]
    for f in top_num:
        q = pd.qcut(d[f].rank(method="first"), 10, labels=list(range(1, 11)))
        s = d.groupby(q, observed=True)["target_default"].mean().round(3)
        print(f"\n{f} (decile 1 = lowest value):")
        print(" ".join(f"{v:.3f}" for v in s.values))

    title("7. NEAR-DUPLICATE FEATURES (|correlation| > 0.85)")
    num = d[[f for f in M.NUM + NEW if pd.api.types.is_numeric_dtype(d[f])]]
    c = num.corr().abs()
    pairs_c = [(a, b2, round(c.loc[a, b2], 3)) for i, a in enumerate(c.columns)
               for b2 in c.columns[i + 1:] if c.loc[a, b2] > 0.85]
    for a, b2, v in sorted(pairs_c, key=lambda x: -x[2]):
        print(f"{a:28s} ~ {b2:28s} {v}")
    print("-> Overlapping features split their effect in LR, which can flip signs.")

    title("8. COMBINED EFFECTS (hidden risky groups)")
    d["lti_band"] = pd.qcut(d["loan_to_income"], 4, labels=["LTI Q1 low", "Q2", "Q3", "Q4 high"])
    print("Default rate: loan type x loan-to-income band")
    print(pd.crosstab(d["loan_type"], d["lti_band"], values=d["target_default"],
                      aggfunc="mean").round(3))
    d["any_missed"] = np.where(d["rep_missed"] > 0, "missed >= 1", "never missed")
    print("\nDefault rate: employment type x missed payments")
    print(pd.crosstab(d["employment_type"], d["any_missed"], values=d["target_default"],
                      aggfunc="mean").round(3))
    d["bur_wo"] = np.where(d["bur_has_writeoff"] == 1, "bureau write-off", "no write-off")
    print("\nDefault rate: income band x bureau write-off")
    print(pd.crosstab(d["income_band"], d["bur_wo"], values=d["target_default"],
                      aggfunc="mean").round(3))

    title("9. CAN NEW FEATURES IMPROVE THE MODEL? (5-fold CV, grouped by customer)")
    X_base = M.make_X(d)
    X_new = pd.concat([X_base, d[NEW].fillna(0).reset_index(drop=True)], axis=1)
    for col in ["x_bur_wo_amount", "x_debt_to_income", "x_emi_to_income"]:
        X_new[col] = np.log1p(X_new[col].clip(lower=0))
    groups = d["customer_id"]
    res = {}
    for name, X in [("base", X_base), ("base + new", X_new)]:
        for kind in ["lr", "xgb"]:
            mean, std, oof = cv_auc(X, y, groups, kind)
            res[(name, kind)] = (mean, std, oof)
            print(f"{name:12s} {kind:4s}  AUC {mean:.4f} +/- {std:.4f}")
    gain = res[("base + new", "xgb")][0] - res[("base", "xgb")][0]
    print(f"\nXGBoost gain from new features: {gain:+.4f} AUC")
    print("-> The +/- shows how much AUC moves between folds. A gain smaller than that is noise.")

    title("10. WHERE DOES THE MODEL FAIL? (out-of-fold, XGBoost base + new)")
    d["oof"] = res[("base + new", "xgb")][2]
    print("AUC and calibration inside each loan type:")
    for lt, g in d.groupby("loan_type"):
        print(f"  {lt:14s} loans {len(g):5d}  AUC {roc_auc_score(g['target_default'], g['oof']):.3f}"
              f"  predicted {g['oof'].mean():.3f}  actual {g['target_default'].mean():.3f}")
    cut = d["oof"].quantile(0.30)
    missed = d[(d["target_default"] == 1) & (d["oof"] <= cut)]
    caught = d[(d["target_default"] == 1) & (d["oof"] > d["oof"].quantile(0.80))]
    print(f"\nDefaults the model MISSED (scored in the safest 30%): {len(missed):,} "
          f"of {int(y.sum()):,} defaults")
    cols = ["annual_income", "age", "years_employed", "rep_missed", "rep_mean_dpd",
            "bur_writeoffs", "bur_mean_dpd", "loan_to_income"]
    print(pd.DataFrame({"missed defaults": missed[cols].mean(),
                        "caught defaults (top 20%)": caught[cols].mean(),
                        "all loans": d[cols].mean()}).round(2))
    print("-> Missed defaults look like good customers in our data: the signal is not in"
          " these tables (e.g. sudden job loss, health event).")


if __name__ == "__main__":
    main()