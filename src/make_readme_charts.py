"""
make_readme_charts.py
Creates the charts used in README.md and saves them as PNG files in assets/.

Run:  python src/make_readme_charts.py
Needs: python src/build.py to have run first (uses data/processed/).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DATA = Path("data/processed")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN, ORANGE = "#1565c0", "#c62828", "#9e9e9e", "#2e7d32", "#ef6c00"
SEG_COLORS = {"Low": GREEN, "Medium": "#f9a825", "High": ORANGE, "Very High": RED}
ORDER = {
    "age_band": ["<30", "30-39", "40-49", "50-59", "60+"],
    "income_band": ["Q1 Low", "Q2", "Q3", "Q4 High"],
    "risk_segment": ["Low", "Medium", "High", "Very High"],
}
LOSS_NO_COLLATERAL, LOSS_WITH_COLLATERAL = 0.45, 0.25
CUTOFF = 0.20

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def rate_by(df, col):
    t = df.groupby(col, observed=True)["target_default"].agg(["mean", "count"])
    if col in ORDER:
        t = t.reindex([x for x in ORDER[col] if x in t.index])
    else:
        t = t.sort_values("mean", ascending=False)
    return t


def bar_rate(df, col, title, name, xlabel):
    t = rate_by(df, col)
    avg = df["target_default"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [RED if v > avg * 1.15 else (GREEN if v < avg * 0.85 else BLUE) for v in t["mean"]]
    bars = ax.bar([str(i) for i in t.index], t["mean"], color=colors)
    ax.axhline(avg, ls="--", color=GREY, label=f"Average {avg:.1%}")
    ax.legend(loc="upper right", frameon=False)
    for b, v, n in zip(bars, t["mean"], t["count"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}\n({n:,})", ha="center",
                va="bottom", fontsize=8.5)
    ax.set_ylim(0, t["mean"].max() * 1.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Default rate")
    save(fig, name)


def confusion(ax, y, flag, title):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap="Blues")
    labels = [["Good loan,\nnot flagged", "Good loan,\nflagged (false alarm)"],
              ["Bad loan,\nmissed", "Bad loan,\ncaught"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n{labels[i][j]}", ha="center", va="center",
                    fontsize=8.5, color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Good (0)", "Defaulted (1)"])
    ax.set_xlabel(f"Model decision (risk >= {CUTOFF:.0%})")
    ax.set_ylabel("What really happened")
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    ax.set_title(f"{title}\ncaught {rec:.0%} of bad loans, {prec:.0%} of flags are bad",
                 fontweight="bold", fontsize=10)
    ax.grid(False)


def main():
    s = pd.read_csv(DATA / "scored_accounts.csv")
    an = pd.read_csv(DATA / "anomalies.csv")
    imp = pd.read_csv(DATA / "feature_importance.csv")
    y = s["target_default"]

    # 1. Default split
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = y.value_counts().sort_index()
    bars = ax.bar(["Paid fine (0)", "Defaulted (1)"], counts.values, color=[BLUE, RED])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(y):.1%})",
                ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("How many loans defaulted?", fontweight="bold")
    ax.set_ylabel("Loans")
    save(fig, "01_default_split.png")

    # 2-7. One chart per idea tested
    bar_rate(s, "age_band", "Idea 1: younger people default more (TRUE)", "02_idea_age.png", "Age")
    bar_rate(s, "income_band", "Idea 2: lower income means more defaults (TRUE)",
             "03_idea_income.png", "Income group (Q1 = lowest)")
    bar_rate(s, "employment_type", "Idea 3: unemployed people default more (TRUE)",
             "04_idea_employment.png", "Employment type")
    bar_rate(s, "loan_type", "Idea 8: loans with collateral are safer (FALSE: mortgages are riskiest)",
             "05_idea_loan_type.png", "Loan type")
    bar_rate(s, "region", "Idea 9: region affects risk (FALSE)", "06_idea_region.png", "Region")
    s["missed_months"] = s["rep_missed"].clip(upper=4).astype(int).astype(str).replace({"4": "4+"})
    ORDER["missed_months"] = ["0", "1", "2", "3", "4+"]
    bar_rate(s, "missed_months", "Idea 5: missed payments warn of default (TRUE)",
             "07_idea_missed.png", "Months missed in the last 12")

    # 8. Interest rate: good vs bad within each loan type
    g = s.groupby(["loan_type", "target_default"])["interest_rate_pct"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g[0], 0.4, label="Paid fine", color=BLUE)
    ax.bar(x + 0.2, g[1], 0.4, label="Defaulted", color=RED)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("Average interest rate (%)")
    ax.set_title("Idea 11: riskier people pay a higher rate (FALSE: almost the same)",
                 fontweight="bold")
    ax.legend()
    save(fig, "08_idea_interest_rate.png")

    # 9. Heatmap income x employment
    ct = pd.crosstab(s["income_band"], s["employment_type"], values=y, aggfunc="mean")
    ct = ct.reindex([r for r in ORDER["income_band"] if r in ct.index])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(ct.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(ct.shape[1]), ct.columns)
    ax.set_yticks(range(ct.shape[0]), ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                    color="white" if v > np.nanmax(ct.values) * 0.7 else "black")
    ax.set_title("Cross-cut: default rate by income group and employment type", fontweight="bold")
    ax.grid(False)
    fig.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    save(fig, "09_heatmap_income_employment.png")

    # Model charts use the TEST set only (customers the model never saw)
    t = s[s["in_test_set"] == 1]
    yt = t["target_default"]

    # 10. Confusion matrices at the 20% cut-off
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    confusion(axes[0], yt, t["pd_score_lr"] >= CUTOFF, "Logistic regression")
    confusion(axes[1], yt, t["pd_score"] >= CUTOFF, "XGBoost")
    save(fig, "10_confusion_matrices.png")

    # 11. ROC curves
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for col, name, c in [("pd_score_lr", "Logistic regression", ORANGE), ("pd_score", "XGBoost", BLUE)]:
        fpr, tpr, _ = roc_curve(yt, t[col])
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(yt, t[col]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Coin flip (AUC 0.5)")
    ax.set_xlabel("Good loans wrongly flagged")
    ax.set_ylabel("Bad loans caught")
    ax.set_title("ROC curve (test set)", fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "11_roc_curves.png")

    # 12. Lift by decile
    d = t.assign(decile=pd.qcut(t["pd_score"].rank(method="first", ascending=False), 10,
                                labels=list(range(1, 11))))
    lt = d.groupby("decile", observed=True).agg(rate=("target_default", "mean"),
                                                bad=("target_default", "sum"))
    lt["cum"] = lt["bad"].cumsum() / lt["bad"].sum()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(lt.index.astype(str), lt["rate"], color=BLUE)
    for b, v in zip(bars, lt["rate"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(yt.mean(), ls="--", color=GREY)
    ax.set_ylabel("Default rate")
    ax.set_xlabel("Score group (1 = riskiest 10% of loans)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2 = ax.twinx()
    ax2.plot(lt.index.astype(str), lt["cum"], color=RED, marker="o")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Share of all defaults caught so far", color=RED)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.grid(False)
    ax.set_title(f"Lift: the riskiest 20% of loans hold {lt['cum'].iloc[1]:.0%} of defaults",
                 fontweight="bold")
    save(fig, "12_lift.png")

    # 13. Calibration
    c = t.assign(bin=pd.qcut(t["pd_score"], 10, duplicates="drop")).groupby(
        "bin", observed=True).agg(pred=("pd_score", "mean"), real=("target_default", "mean"))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mx = max(c["pred"].max(), c["real"].max()) * 1.1
    ax.plot([0, mx], [0, mx], "--", color=GREY, label="Perfect")
    ax.plot(c["pred"], c["real"], marker="o", color=BLUE, label="Model")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Real default rate")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"Calibration: predicted {t['pd_score'].mean():.1%} vs real {yt.mean():.1%}",
                 fontweight="bold")
    ax.legend()
    save(fig, "13_calibration.png")

    # 14. Top drivers (SHAP)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE)
    ax.set_xlabel("Average impact on the risk score (SHAP)")
    ax.set_title("What drives the risk score most", fontweight="bold")
    save(fig, "14_top_drivers.png")

    # 15. Risk segments on the test set
    seg = t.groupby("risk_segment")["target_default"].mean().reindex(ORDER["risk_segment"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(seg.index, seg.values, color=[SEG_COLORS[k] for k in seg.index])
    for b, v in zip(bars, seg.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Real default rate")
    ax.set_title("Risk groups rank in the right order (test set)", fontweight="bold")
    save(fig, "15_risk_segments.png")

    # 16. Loss concentration
    s["loss_share"] = np.where(s["collateral_flag"] == 1, LOSS_WITH_COLLATERAL, LOSS_NO_COLLATERAL)
    s["real_loss"] = y * s["loss_share"] * s["loan_amount"]
    o = s.sort_values("pd_score", ascending=False)
    share_loans = np.arange(1, len(o) + 1) / len(o)
    share_loss = o["real_loss"].cumsum() / o["real_loss"].sum()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(share_loans, share_loss, color=RED, label="Sorted by our risk score")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random order")
    k = int(len(o) * 0.2)
    ax.scatter([0.2], [share_loss.iloc[k - 1]], color=RED, zorder=5)
    ax.annotate(f"riskiest 20% of loans\n= {share_loss.iloc[k - 1]:.0%} of the money lost",
                (0.2, share_loss.iloc[k - 1]), xytext=(0.32, 0.45),
                arrowprops=dict(arrowstyle="->"))
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of loans, riskiest first")
    ax.set_ylabel("Share of money lost")
    ax.set_title("Most of the loss sits in the riskiest loans", fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "16_loss_concentration.png")

    # 17. Anomalies
    names = {"sudden_miss": "Sudden miss", "dpd_jump": "Lateness jump",
             "peer_bur_utilization": "High credit use", "peer_bur_mean_dpd": "Late at other banks",
             "isolation_forest": "Rare mix"}
    rules = an["flags"].str.split(", ").explode().map(names).value_counts()
    rest = s[~s["account_id"].isin(an["account_id"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    axes[0].barh(rules.index[::-1], rules.values[::-1], color=ORANGE)
    axes[0].set_xlabel("Accounts flagged")
    axes[0].set_title(f"{len(an):,} accounts flagged, by rule", fontweight="bold")
    vals = [an["target_default"].mean(), rest["target_default"].mean()]
    bars = axes[1].bar(["Flagged", "Not flagged"], vals, color=[RED, BLUE])
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_title("Flagged accounts default more", fontweight="bold")
    save(fig, "17_anomalies.png")


def extra_charts():
    """Charts 18-22: feature strength, near-copies, algorithm comparison and feature
    engineering round 2. The numbers are copied from the reports/ text files
    (deep_analysis, benchmark_models, feature_engineering_v2), because those
    experiments take minutes to re-run."""
    from matplotlib.patches import Patch

    # 18. Feature strength (Information Value) - reports/deep_analysis.txt
    iv = {"annual_income": .1723, "rep_mean_dpd": .1667, "rep_late": .1246, "rep_max_dpd": .1244,
          "years_employed": .1169, "rep_missed": .1132, "rep_pay_ratio": .1056, "bur_mean_dpd": .0938,
          "loan_to_income": .0866, "bur_max_dpd": .0849, "employment_type": .0804,
          "rep_ever_30dpd": .0779, "age": .0765, "rep_dpd_last3": .0671, "rep_dpd_trend": .0585,
          "rep_dpd_first3": .0570, "rep_ever_60dpd": .0471, "tenure_months": .0456,
          "rep_missed_last3": .0421, "loan_amount": .0421, "loan_type": .0415,
          "rep_last_balance": .0413, "bur_writeoffs": .0236, "education_level": .0180,
          "bur_has_writeoff": .0150, "years_at_residence": .0080, "interest_rate_pct": .0061,
          "bur_active": .0061, "bur_lines": .0053, "bur_utilization": .0044, "bur_sanctioned": .0043,
          "collateral_flag": .0043, "bur_outstanding": .0040, "num_dependents": .0033,
          "bur_nbfc_fintech_share": .0032, "occupation_sector": .0024, "region": .0020,
          "loan_purpose": .0019, "disbursal_channel": .0018, "housing_type": .0016,
          "bur_no_history": .0014, "city_tier": .0009, "customer_loans": .0008}
    s = pd.Series(iv).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(s.index, s.values, color=[GREEN if v >= .1 else (BLUE if v >= .02 else GREY) for v in s])
    ax.axvline(.02, ls="--", color=GREY)
    ax.axvline(.1, ls="--", color=GREEN)
    ax.text(.021, -1.3, "0.02: useful", color="#555")
    ax.text(.101, -1.3, "0.10: medium", color=GREEN)
    ax.set_xlabel("Information Value (how well one feature alone separates good and bad loans)")
    ax.set_title("Feature strength: green = medium, blue = weak, grey = useless", fontweight="bold")
    save(fig, "18_feature_strength.png")

    # 19. Near-copies (correlation) - reports/deep_analysis.txt
    pairs = [("rep_missed", "rep_pay_ratio", 1.0), ("loan_amount", "rep_last_balance", .998),
             ("loan_to_income", "debt_to_income", .958), ("bur_writeoffs", "bur_has_writeoff", .908),
             ("rep_mean_dpd", "recency_dpd", .902), ("rep_missed", "rep_mean_dpd", .894),
             ("rep_mean_dpd", "rep_pay_ratio", .894), ("rep_dpd_last3", "rep_missed_last3", .881),
             ("rep_max_dpd", "rep_ever_30dpd", .874), ("tenure_months", "rep_last_balance", .872),
             ("rep_max_dpd", "rep_mean_dpd", .872), ("bur_max_dpd", "bur_mean_dpd", .86),
             ("loan_amount", "tenure_months", .853), ("rep_missed", "rep_ever_30dpd", .853)][::-1]
    colors = [ORANGE if (a.startswith("rep") and b.startswith(("rep", "rec")))
              else (BLUE if "bur" in a else RED) for a, b, _ in pairs]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([f"{a}  ~  {b}" for a, b, _ in pairs], [v for *_, v in pairs], color=colors)
    ax.set_xlim(.8, 1.01)
    ax.set_xlabel("Correlation")
    ax.set_title("Features that are near copies of each other", fontweight="bold")
    ax.legend(handles=[Patch(color=ORANGE, label="payment behaviour"), Patch(color=RED, label="loan size"),
                       Patch(color=BLUE, label="other banks")], loc="lower right")
    save(fig, "19_correlated_features.png")

    def auc_bars(rows, colour, xlim, xlabel, title, name, ref, width=8):
        names = [r[0] for r in rows][::-1]
        aucs = np.array([r[1] for r in rows][::-1])
        errs = np.array([r[2] for r in rows][::-1])
        fig, ax = plt.subplots(figsize=(width, 5.5))
        ax.barh(names, aucs, xerr=errs, color=[colour(n) for n in names], capsize=3)
        ax.set_xlim(*xlim)
        ax.axvline(ref, ls="--", color=GREY)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        for i, v in enumerate(aucs):
            ax.text(v + (xlim[1] - xlim[0]) * 0.22, i, f"{v:.4f}", va="center", fontsize=8.5)
        save(fig, name)

    # 20. Algorithm comparison - reports/benchmark_models.txt
    auc_bars([("Average of all models", .6778, .0070), ("Average of LR + XGBoost", .6773, .0073),
              ("LR, 29 useful features", .6771, .0073), ("XGBoost (dashboard)", .6762, .0065),
              ("CatBoost", .6759, .0073), ("LightGBM", .6743, .0066),
              ("HistGradientBoosting", .6740, .0072), ("LR, strong regularization", .6734, .0082),
              ("LR, all 81 features", .6724, .0083), ("Random Forest", .6707, .0083),
              ("Extra Trees", .6636, .0070)],
             lambda n: GREEN if "XGBoost (dash" in n or "29" in n else (GREY if "Extra" in n else BLUE),
             (.645, .69), "AUC (5-fold cross-validation, split by customer). Lines = fold-to-fold spread",
             "Every algorithm reaches the same level", "20_algorithm_benchmark.png", .6724)

    # 21. Feature engineering round 2 - reports/feature_engineering_v2.txt
    auc_bars([("LR, all 81 features", .6734, .0082), ("LR, 29 useful features", .6774, .0073),
              ("LR, clean scorecard (27)", .6775, .0074),
              ("LR, clean + other-bank by type/lender (36)", .6781, .0066),
              ("LR, clean + combined features (32)", .6773, .0073),
              ("LR, clean + all new features (41)", .6779, .0066),
              ("XGBoost, all 81 features", .6762, .0065),
              ("XGBoost + other-bank by type/lender (90)", .6767, .0046),
              ("XGBoost + all new features (95)", .6773, .0051),
              ("WoE scorecard, all features (49)", .6686, .0105),
              ("WoE scorecard, useful features (21)", .6774, .0103)],
             lambda n: GREEN if "clean scorecard" in n else
             (ORANGE if "new" in n or "type/lender" in n or "combined" in n else BLUE),
             (.65, .695), "AUC (5-fold cross-validation). Orange = new features tested",
             "New features do not beat the noise; 27 columns are enough",
             "21_feature_engineering_v2.png", .6734, width=8.5)

    # 22. Strength of the new features - reports/feature_engineering_v2.txt
    nf = {"unemployed x low income": .0744, "unemployed x missed": .0739, "late at fintech/NBFC": .0679,
          "young x low income": .0648, "mortgage x loan-to-income": .0440, "late at a bank": .0388,
          "active lines 30+ days late": .0277, "unsecured write-offs": .0244,
          "share of money written off": .0221, "contract x low income": .0216,
          "share of lines closed": .0080, "card utilization": .0045, "secured write-offs": .0026,
          "number of credit types": .0024}
    s = pd.Series(nf).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index, s.values, color=[BLUE if v >= .02 else GREY for v in s])
    ax.axvline(.1723, ls="--", color=GREEN)
    ax.text(.1723, -0.9, " annual income (best existing)", color=GREEN, ha="right")
    ax.axvline(.02, ls="--", color=GREY)
    ax.set_xlim(0, .19)
    ax.set_xlabel("Information Value")
    ax.set_title("New features are weak next to the best existing one", fontweight="bold")
    save(fig, "22_new_feature_strength.png")


if __name__ == "__main__":
    main()
    extra_charts()