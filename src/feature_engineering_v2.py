"""
feature_engineering_v2.py
Tests 5 new feature ideas with the same fair test as the benchmark
(5-fold cross-validation, split by customer). Only real gains count.

 A. Bureau by credit type   (unsecured vs secured write-offs, card utilization...)
 B. Bureau by lender type   (lateness at fintech/NBFC vs banks)
 C. Combined features       (e.g. unemployed x low income) for logistic regression
 D. Clean scorecard         (drop exact duplicates, keep only useful features)
 E. WoE binning             (classic bank scorecard: 10 risk groups per feature)

Run:  python src/feature_engineering_v2.py > reports/feature_engineering_v2.txt
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
import model as M

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = Path("data/processed")
RAW = Path("data/raw")
DUPLICATES = ["rep_pay_ratio", "rep_last_balance", "collateral_flag", "bur_has_writeoff"]


def title(t):
    print("\n" + "=" * 90 + "\n" + t + "\n" + "=" * 90)


# ------------------------------------------------------------------
# Information Value (same definition as deep_analysis.py)
# ------------------------------------------------------------------
def info_value(x, y, bins=10):
    d = pd.DataFrame({"x": x, "y": y})
    if pd.api.types.is_numeric_dtype(d["x"]) and d["x"].nunique() > bins:
        d["b"] = pd.qcut(d["x"].fillna(-999).rank(method="first"), bins, labels=False)
    else:
        d["b"] = d["x"].astype(str)
    g = d.groupby("b")["y"].agg(["count", "sum"])
    pb = (g["sum"] + 0.5) / (g["sum"].sum() + 0.5)
    pg = (g["count"] - g["sum"] + 0.5) / ((g["count"] - g["sum"]).sum() + 0.5)
    return float(((pg - pb) * np.log(pg / pb)).sum())


# ------------------------------------------------------------------
# A + B. Bureau features by credit type and lender type
# ------------------------------------------------------------------
def bureau_v2(b):
    b = b.copy()
    b["unsecured"] = ~b["credit_type"].isin(["Mortgage", "Auto Loan"])
    b["wo"] = b["account_status"] == "Written-Off"
    b["active"] = b["account_status"] == "Active"
    b["nonbank"] = b["lender_category"].isin(["NBFC", "Fintech"])
    b["revolving"] = b["credit_type"].isin(["Credit Card", "Overdraft"])
    b["wo_unsec"] = (b["wo"] & b["unsecured"]).astype(int)
    b["wo_sec"] = (b["wo"] & ~b["unsecured"]).astype(int)
    b["wo_amt"] = np.where(b["wo"], b["sanctioned_amount"], 0)
    b["active_dpd30"] = (b["active"] & (b["days_past_due"] >= 30)).astype(int)
    b["dpd_nonbank"] = np.where(b["nonbank"], b["days_past_due"], np.nan)
    b["dpd_bank"] = np.where(~b["nonbank"], b["days_past_due"], np.nan)
    b["rev_out"] = np.where(b["revolving"] & b["active"], b["outstanding_amount"], 0)
    b["rev_lim"] = np.where(b["revolving"] & b["active"], b["sanctioned_amount"], 0)
    b["closed"] = (b["account_status"] == "Closed").astype(int)

    g = b.groupby("customer_id").agg(
        v2_wo_unsecured=("wo_unsec", "sum"),
        v2_wo_secured=("wo_sec", "sum"),
        v2_wo_amount=("wo_amt", "sum"),
        v2_sanctioned=("sanctioned_amount", "sum"),
        v2_active_dpd30=("active_dpd30", "sum"),
        v2_nonbank_max_dpd=("dpd_nonbank", "max"),
        v2_bank_max_dpd=("dpd_bank", "max"),
        v2_rev_out=("rev_out", "sum"),
        v2_rev_lim=("rev_lim", "sum"),
        v2_closed=("closed", "sum"),
        v2_lines=("bureau_id", "count"),
        v2_credit_types=("credit_type", "nunique"),
    ).reset_index()
    g["v2_wo_amount_share"] = np.where(g["v2_sanctioned"] > 0, g["v2_wo_amount"] / g["v2_sanctioned"], 0)
    g["v2_revolving_util"] = np.where(g["v2_rev_lim"] > 0, g["v2_rev_out"] / g["v2_rev_lim"], 0)
    g["v2_closed_share"] = g["v2_closed"] / g["v2_lines"]
    return g.drop(columns=["v2_wo_amount", "v2_sanctioned", "v2_rev_out", "v2_rev_lim",
                           "v2_closed", "v2_lines"])


BUREAU_TYPE = ["v2_wo_unsecured", "v2_wo_secured", "v2_wo_amount_share", "v2_active_dpd30",
               "v2_revolving_util", "v2_closed_share", "v2_credit_types"]
BUREAU_LENDER = ["v2_nonbank_max_dpd", "v2_bank_max_dpd"]


# ------------------------------------------------------------------
# C. Combined features
# ------------------------------------------------------------------
def combined(df):
    unemp = (df["employment_type"] == "Unemployed").astype(int)
    contract = (df["employment_type"] == "Contract").astype(int)
    low = (df["income_band"] == "Q1 Low").astype(int)
    young = (df["age"] < 30).astype(int)
    mortgage = (df["loan_type"] == "Mortgage").astype(int)
    return pd.DataFrame({
        "c_unemployed_x_low_income": unemp * low,
        "c_contract_x_low_income": contract * low,
        "c_young_x_low_income": young * low,
        "c_mortgage_x_loan_to_income": mortgage * np.log1p(df["loan_to_income"]),
        "c_unemployed_x_missed": unemp * df["rep_missed"],
    }, index=df.index)


COMBINED = ["c_unemployed_x_low_income", "c_contract_x_low_income", "c_young_x_low_income",
            "c_mortgage_x_loan_to_income", "c_unemployed_x_missed"]


# ------------------------------------------------------------------
# E. WoE binning (fitted on the training fold only, so no leakage)
# ------------------------------------------------------------------
class WoE:
    def fit(self, X, y):
        self.maps = {}
        y = np.asarray(y)
        for c in X.columns:
            x = X[c]
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
                edges = np.unique(np.quantile(x.fillna(x.median()), np.linspace(0, 1, 11))[1:-1])
                key = np.digitize(x.fillna(x.median()), edges)
                kind = ("num", edges, x.median())
            else:
                key = x.astype(str).values
                kind = ("cat", None, None)
            d = pd.DataFrame({"k": key, "y": y})
            g = d.groupby("k")["y"].agg(["sum", "count"])
            bad = g["sum"] + 0.5
            good = g["count"] - g["sum"] + 0.5
            self.maps[c] = (kind, np.log((good / good.sum()) / (bad / bad.sum())).to_dict())
        return self

    def transform(self, X):
        out = {}
        for c, (kind, w) in self.maps.items():
            x = X[c]
            if kind[0] == "num":
                key = np.digitize(x.fillna(kind[2]), kind[1])
            else:
                key = x.astype(str).values
            out[c] = pd.Series(key).map(w).fillna(0).values
        return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------------------
# Fair test
# ------------------------------------------------------------------
def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def xg():
    return xgb.XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
                             reg_lambda=5, random_state=42)


def cv(X, y, folds, make):
    aucs = []
    for tr, te in folds:
        m = make()
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def cv_woe(R, y, folds):
    aucs = []
    for tr, te in folds:
        w = WoE().fit(R.iloc[tr], y[tr])
        m = lr()
        m.fit(w.transform(R.iloc[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(w.transform(R.iloc[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    df = pd.read_csv(DATA / "analytic_dataset.csv")
    b = pd.read_csv(RAW / "bureau_credit.csv")
    y = df["target_default"].values
    df = df.merge(bureau_v2(b), on="customer_id", how="left")
    for c in BUREAU_TYPE + BUREAU_LENDER:
        df[c] = df[c].fillna(0)
    df = pd.concat([df, combined(df)], axis=1)

    # ---------- IV of the new features ----------
    title("1. HOW STRONG ARE THE NEW FEATURES ON THEIR OWN? (Information Value)")
    print("IV: < 0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong")
    new = BUREAU_TYPE + BUREAU_LENDER + COMBINED
    ivs = pd.DataFrame({"feature": new, "IV": [info_value(df[f], y) for f in new]})
    ivs["group"] = ["A: bureau type"] * len(BUREAU_TYPE) + ["B: bureau lender"] * len(BUREAU_LENDER) \
        + ["C: combined"] * len(COMBINED)
    print(ivs.sort_values("IV", ascending=False).round(4).to_string(index=False))
    print("\nFor comparison, the strongest existing features: annual_income 0.17, rep_mean_dpd 0.17")

    # ---------- Build design matrices ----------
    X = M.make_X(df).reset_index(drop=True)
    extra_b = df[BUREAU_TYPE + BUREAU_LENDER].reset_index(drop=True)
    extra_c = df[COMBINED].reset_index(drop=True)
    for c in ["v2_nonbank_max_dpd", "v2_bank_max_dpd"]:
        extra_b[c] = extra_b[c].astype(float)

    rank = pd.read_csv(DATA / "feature_ranking.csv") if (DATA / "feature_ranking.csv").exists() else None
    if rank is not None:
        keep = set(rank.loc[rank["IV"] >= 0.02, "feature"])
        useful = [c for c in X.columns
                  if c in keep or any(c.startswith(k + "_") for k in keep if k in M.CAT)]
    else:
        useful = list(X.columns)
    clean = [c for c in useful if c not in DUPLICATES]

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(X, y, df["customer_id"].values))

    title("2. DOES EACH IDEA IMPROVE THE MODEL? (5-fold CV, split by customer)")
    tests = [
        ("LR: all current features (baseline)", X, lr),
        ("LR: only useful features (IV >= 0.02)", X[useful], lr),
        ("D. LR: useful features, duplicates removed", X[clean], lr),
        ("A+B. LR: clean + bureau by type and lender", pd.concat([X[clean], extra_b], axis=1), lr),
        ("C. LR: clean + combined features", pd.concat([X[clean], extra_c], axis=1), lr),
        ("A+B+C. LR: clean + all new features", pd.concat([X[clean], extra_b, extra_c], axis=1), lr),
        ("XGBoost: all current features", X, xg),
        ("A+B. XGBoost: + bureau by type and lender", pd.concat([X, extra_b], axis=1), xg),
        ("A+B+C. XGBoost: + all new features", pd.concat([X, extra_b, extra_c], axis=1), xg),
    ]
    rows = []
    for name, Xt, make in tests:
        a, s = cv(Xt, y, folds, make)
        rows.append((name, Xt.shape[1], a, s))
        print(f"done: {name}  AUC {a:.4f} +/- {s:.4f}")

    raw_cols = [c for c in M.NUM + M.CAT]
    R = df[raw_cols].reset_index(drop=True)
    a, s = cv_woe(R, y, folds)
    rows.append(("E. LR on WoE bins (all features)", R.shape[1], a, s))
    print(f"done: E. WoE  AUC {a:.4f} +/- {s:.4f}")
    if rank is not None:
        Ru = R[[c for c in raw_cols if c in keep and c not in DUPLICATES]]
        a, s = cv_woe(Ru, y, folds)
        rows.append(("E. LR on WoE bins (useful features only)", Ru.shape[1], a, s))
        print(f"done: E. WoE useful  AUC {a:.4f} +/- {s:.4f}")

    res = pd.DataFrame(rows, columns=["test", "features", "AUC", "spread"])
    base = res.loc[0, "AUC"]
    noise = res["spread"].max()
    res["gain_vs_baseline"] = res["AUC"] - base
    res["real_gain?"] = np.where(res["gain_vs_baseline"] > noise, "yes",
                                 np.where(res["gain_vs_baseline"] < -noise, "worse", "no (noise)"))
    title("3. RESULTS")
    print(res.round(4).to_string(index=False))
    print(f"\nNoise level (largest fold-to-fold spread): {noise:.4f}")
    if rank is not None:
        print(f"\nThe clean scorecard uses {len(clean)} columns:")
        print(", ".join(clean))


if __name__ == "__main__":
    main()