"""
model.py
Trains a logistic regression baseline and an XGBoost model to predict
target_default. Reports AUC, KS, Gini. Explains drivers with SHAP.

Input : data/processed/analytic_dataset.csv
Output: models/*.joblib, data/processed/scored_accounts.csv,
        data/processed/model_metrics.json, data/processed/feature_importance.csv
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)
TARGET = "target_default"

NUM = [
    # loan
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag",
    "interest_rate_missing", "loan_to_income", "customer_loans",
    # customer
    "age", "years_employed", "annual_income", "num_dependents",
    "years_at_residence", "years_employed_missing", "num_dependents_missing",
    "years_at_residence_missing", "years_employed_capped",
    # bureau
    "bur_lines", "bur_active", "bur_writeoffs", "bur_sanctioned",
    "bur_outstanding", "bur_max_dpd", "bur_mean_dpd", "bur_dpd_missing",
    "bur_nbfc_fintech_share", "bur_utilization", "bur_has_writeoff",
    "bur_no_history",
    # repayment
    "rep_late", "rep_missed", "rep_max_dpd", "rep_mean_dpd", "rep_ever_30dpd",
    "rep_ever_60dpd", "rep_last_balance", "rep_pay_ratio", "rep_dpd_first3",
    "rep_dpd_last3", "rep_dpd_trend", "rep_missed_last3",
]
CAT = ["loan_type", "employment_type", "region", "city_tier", "housing_type",
       "education_level", "disbursal_channel",
       "loan_purpose", "occupation_sector"]
LOG_COLS = ["loan_amount", "annual_income", "bur_sanctioned",
            "bur_outstanding", "rep_last_balance", "loan_to_income"]
# NOT used: IDs, dates, customer_since (unreliable), coll_* (leakage)


def make_X(df):
    X = df[NUM + CAT].copy()
    for col in LOG_COLS:
        X[col] = np.log1p(X[col])
    X = pd.get_dummies(X, columns=CAT, drop_first=True, dtype=int)
    return X


def metrics(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"AUC": round(auc, 4),
            "KS": round(float(np.max(tpr - fpr)), 4),
            "Gini": round(2 * auc - 1, 4)}


def segment(pd_score):
    return pd.cut(pd_score, [0, 0.10, 0.20, 0.35, 1.0], include_lowest=True,
                  labels=["Low", "Medium", "High", "Very High"])


def main():
    df = pd.read_csv(DATA / "analytic_dataset.csv")
    X = make_X(df)
    y = df[TARGET].values

    # Split by customer: the same customer is never in both train and test
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    tr, te = next(sgkf.split(X, y, groups=df["customer_id"]))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    print(f"Train {len(tr)} | Test {len(te)} | "
          f"test default rate {yte.mean():.3f}")

    # Baseline: logistic regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # ML model: XGBoost (shallow trees + early stopping to limit overfitting)
    Xfit, Xval = Xtr.iloc[:-2000], Xtr.iloc[-2000:]
    yfit, yval = ytr[:-2000], ytr[-2000:]
    xg = xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.7,
                           min_child_weight=20, reg_lambda=5,
                           eval_metric="auc", early_stopping_rounds=50,
                           random_state=42)
    xg.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    print("XGBoost best number of trees:", xg.best_iteration)
    p_xg = xg.predict_proba(Xte)[:, 1]

    results = {"logistic_regression": metrics(yte, p_lr),
               "xgboost": metrics(yte, p_xg),
               "train_rows": int(len(tr)), "test_rows": int(len(te)),
               "split": "StratifiedGroupKFold by customer_id, 75/25"}
    print(json.dumps(results, indent=2))

    # Decile table on the test set (for KS view and monitoring baseline)
    dec = pd.DataFrame({"p": p_xg, "y": yte})
    dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10,
                            labels=list(range(10, 0, -1)))
    dec_tab = dec.groupby("decile", observed=True).agg(
        accounts=("y", "count"), defaults=("y", "sum"),
        avg_pd=("p", "mean")).sort_index(ascending=False)
    dec_tab["default_rate"] = dec_tab["defaults"] / dec_tab["accounts"]
    dec_tab.to_csv(DATA / "decile_table.csv")

    # SHAP values (XGBoost native TreeSHAP) for every account
    booster = xg.get_booster()
    contrib = booster.predict(
        xgb.DMatrix(X), pred_contribs=True,
        iteration_range=(0, xg.best_iteration + 1))[:, :-1]
    shap_df = pd.DataFrame(contrib, columns=X.columns)

    imp = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(contrib).mean(axis=0),
        "lr_coef": lr.named_steps["logisticregression"].coef_[0],
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(DATA / "feature_importance.csv", index=False)
    print("\nTop 15 drivers (SHAP):")
    print(imp.head(15).to_string(index=False))

    # Score every account + top 3 drivers that push risk UP
    df["pd_score"] = xg.predict_proba(X)[:, 1]
    df["pd_score_lr"] = lr.predict_proba(X)[:, 1]
    df["risk_segment"] = segment(df["pd_score"])
    df["in_test_set"] = 0
    df.loc[df.index[te], "in_test_set"] = 1
    top = shap_df.apply(lambda row: row.nlargest(3), axis=1)
    df["top_drivers"] = [
        "; ".join(f"{k} (+{v:.2f})" for k, v in r.dropna().items() if v > 0)
        for _, r in top.iterrows()]

    df.to_csv(DATA / "scored_accounts.csv", index=False)
    shap_df.insert(0, "account_id", df["account_id"])
    shap_df.to_csv(DATA / "shap_values.csv", index=False)

    joblib.dump(lr, MODELS / "logreg.joblib")
    joblib.dump(xg, MODELS / "xgb.joblib")
    joblib.dump(list(X.columns), MODELS / "feature_columns.joblib")
    with open(DATA / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nRisk segments (TEST set only - honest view):")
    print(df[df["in_test_set"] == 1].groupby("risk_segment", observed=True).agg(
        accounts=("account_id", "count"),
        default_rate=("target_default", "mean"),
        exposure=("loan_amount", "sum")).round(3))


if __name__ == "__main__":
    main()