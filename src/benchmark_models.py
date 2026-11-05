"""
benchmark_models.py
Is there a better algorithm for this data? Tests the main credit-scoring
algorithms on the SAME features with the SAME fair test:
5-fold cross-validation, split by customer (no customer in train and test).

Includes the winners of the large Lessmann et al. credit-scoring benchmark
(Random Forest and simple averaged ensembles).

Run:  python src/benchmark_models.py > reports/benchmark_models.txt
Optional extra models: pip install lightgbm catboost  (skipped if not installed)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # same features as the real model

warnings.filterwarnings("ignore")
DATA = Path("data/processed")


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def models():
    m = {
        "Logistic regression (our baseline)": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)),
        "Logistic regression, strong regularization": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=42),
        "Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=42),
        "HistGradientBoosting (LightGBM-style, sklearn)": lambda: HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=3, max_iter=400, l2_regularization=5,
            random_state=42),
        "XGBoost (our model)": lambda: xgb.XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=20, reg_lambda=5, random_state=42),
    }
    try:
        import lightgbm as lgb
        m["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=8, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
            random_state=42, verbose=-1)
    except ImportError:
        print("(LightGBM not installed - skipped)")
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = lambda: CatBoostClassifier(
            iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5,
            random_seed=42, verbose=0)
    except ImportError:
        print("(CatBoost not installed - skipped)")
    return m


def iv_filtered_columns(X):
    """Keep only features with Information Value >= 0.02 (bank scorecard practice)."""
    path = DATA / "feature_ranking.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    keep = set(r.loc[r["IV"] >= 0.02, "feature"])
    cols = [c for c in X.columns
            if c in keep or any(c.startswith(k + "_") for k in keep if k in M.CAT)]
    return cols


def main():
    df = pd.read_csv(DATA / "analytic_dataset.csv")
    y = df["target_default"].values
    groups = df["customer_id"].values
    X = M.make_X(df)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(X, y, groups))
    print(f"Loans: {len(y):,} | features: {X.shape[1]} | default rate: {y.mean():.3f}")
    print("Test: 5-fold cross-validation, split by customer. Same features for every model.\n")

    makers = models()
    oof = {}
    rows = []
    for name, make in makers.items():
        t0 = time.time()
        p_all = np.zeros(len(y))
        aucs, kss = [], []
        for tr, te in folds:
            mdl = make()
            mdl.fit(X.iloc[tr], y[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            p_all[te] = p
            aucs.append(roc_auc_score(y[te], p))
            kss.append(ks(y[te], p))
        oof[name] = p_all
        rows.append({"model": name, "AUC": np.mean(aucs), "AUC_spread": np.std(aucs),
                     "KS": np.mean(kss), "seconds": time.time() - t0})
        print(f"done: {name}  AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Simple averaged ensembles (the kind that ranked well in the Lessmann benchmark)
    def ens(names, label):
        names = [n for n in names if n in oof]
        if len(names) < 2:
            return
        p = np.mean([pd.Series(oof[n]).rank(pct=True).values for n in names], axis=0)
        aucs = [roc_auc_score(y[te], p[te]) for _, te in folds]
        kss = [ks(y[te], p[te]) for _, te in folds]
        rows.append({"model": label, "AUC": np.mean(aucs), "AUC_spread": np.std(aucs),
                     "KS": np.mean(kss), "seconds": np.nan})

    ens(["Logistic regression (our baseline)", "XGBoost (our model)"],
        "Ensemble: average of LR + XGBoost")
    ens(["Logistic regression (our baseline)", "Random Forest", "XGBoost (our model)"],
        "Ensemble: average of LR + RF + XGBoost")
    ens(list(oof), "Ensemble: average of all models")

    # Feature selection test: drop features banks would call useless (IV < 0.02)
    cols = iv_filtered_columns(X)
    if cols:
        aucs = []
        for tr, te in folds:
            mdl = makers["Logistic regression (our baseline)"]()
            mdl.fit(X.iloc[tr][cols], y[tr])
            aucs.append(roc_auc_score(y[te], mdl.predict_proba(X.iloc[te][cols])[:, 1]))
        rows.append({"model": f"LR with only useful features (IV >= 0.02, {len(cols)} cols)",
                     "AUC": np.mean(aucs), "AUC_spread": np.std(aucs), "KS": np.nan,
                     "seconds": np.nan})

    res = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    base = res.loc[res["model"] == "Logistic regression (our baseline)", "AUC"].iloc[0]
    res["gain_vs_LR"] = res["AUC"] - base
    res["real_gain?"] = np.where(res["gain_vs_LR"].abs() > res["AUC_spread"].max(),
                                 "yes", "no (within noise)")
    print("\n" + "=" * 100)
    print("RESULTS (sorted by AUC). 'AUC_spread' = how much AUC moves between folds.")
    print("A gain smaller than the spread is just noise.")
    print("=" * 100)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()