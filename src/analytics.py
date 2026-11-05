"""
analytics.py
Portfolio profiling functions. Used by the dashboard AND the LLM assistant,
so both always show the same numbers (one source of truth).
"""

import json
import pandas as pd
from pathlib import Path

DATA = Path("data/processed")

DIMENSIONS = {
    "loan_type": "Loan type", "employment_type": "Employment type",
    "region": "Region", "income_band": "Income band", "age_band": "Age band",
    "city_tier": "City tier", "housing_type": "Housing type",
    "education_level": "Education", "disbursal_channel": "Channel",
    "loan_purpose": "Loan purpose", "risk_segment": "Risk segment",
}
ORDER = {
    "age_band": ["<30", "30-39", "40-49", "50-59", "60+"],
    "income_band": ["Q1 Low", "Q2", "Q3", "Q4 High"],
    "risk_segment": ["Low", "Medium", "High", "Very High"],
    "city_tier": ["Tier 1", "Tier 2", "Tier 3"],
}


def load_scored():
    return pd.read_csv(DATA / "scored_accounts.csv")


def _ordered(t, dim):
    if dim in ORDER:
        t[dim] = pd.Categorical(t[dim], ORDER[dim], ordered=True)
        return t.sort_values(dim)
    return t.sort_values("default_rate", ascending=False)


def profile(df, dim):
    """Accounts, defaults, default rate, exposure and avg model PD by one dimension."""
    t = df.groupby(dim, observed=True).agg(
        accounts=("account_id", "count"),
        defaults=("target_default", "sum"),
        default_rate=("target_default", "mean"),
        exposure=("loan_amount", "sum"),
        avg_pd=("pd_score", "mean"),
    ).reset_index()
    t["exposure_share"] = t["exposure"] / t["exposure"].sum()
    return _ordered(t, dim).reset_index(drop=True)


def cross(df, row, col, value="target_default"):
    """Cross-dimensional cut: mean of value for row x col."""
    t = pd.crosstab(df[row], df[col], values=df[value], aggfunc="mean")
    if row in ORDER:
        t = t.reindex([x for x in ORDER[row] if x in t.index])
    if col in ORDER:
        t = t[[x for x in ORDER[col] if x in t.columns]]
    return t


def portfolio_kpis(df):
    high = df["risk_segment"].isin(["High", "Very High"])
    return {
        "accounts": int(len(df)),
        "customers": int(df["customer_id"].nunique()),
        "total_exposure": float(df["loan_amount"].sum()),
        "default_rate": float(df["target_default"].mean()),
        "avg_pd": float(df["pd_score"].mean()),
        "high_risk_exposure_share": float(df.loc[high, "loan_amount"].sum()
                                          / df["loan_amount"].sum()),
    }


def insights(df):
    """4-5 insights, each with numbers computed from the data and a 'so what'."""
    avg = df["target_default"].mean()
    lt = profile(df, "loan_type").set_index("loan_type")
    inc = profile(df, "income_band").set_index("income_band")
    age = profile(df, "age_band").set_index("age_band")
    emp = profile(df, "employment_type").set_index("employment_type")
    reg = profile(df, "region").set_index("region")
    x = cross(df, "income_band", "employment_type")

    return [
        {"title": "Mortgages are the riskiest AND the biggest book",
         "finding": (f"Mortgages default at {lt.loc['Mortgage', 'default_rate']:.1%} "
                     f"vs {avg:.1%} portfolio average, and hold "
                     f"{lt.loc['Mortgage', 'exposure_share']:.0%} of total exposure."),
         "so_what": ("Secured loans should be safer. Review mortgage underwriting "
                     "first: this is where most of the money is at risk.")},
        {"title": "Default falls steadily as income rises",
         "finding": (f"Lowest income band defaults at {inc.loc['Q1 Low', 'default_rate']:.1%}, "
                     f"highest at {inc.loc['Q4 High', 'default_rate']:.1%}."),
         "so_what": ("Income is the strongest single driver. Use loan-to-income limits, "
                     "and verify income more strictly for the lowest band.")},
        {"title": "Age matters a lot; region barely matters",
         "finding": (f"Under-30s default at {age.loc['<30', 'default_rate']:.1%} vs "
                     f"{age.loc['60+', 'default_rate']:.1%} for 60+. Regions only range "
                     f"{reg['default_rate'].min():.1%}-{reg['default_rate'].max():.1%}."),
         "so_what": ("Focus risk rules on borrower profile, not geography. "
                     "Regional pricing would add little.")},
        {"title": "Unemployed borrowers are a very high-risk group",
         "finding": (f"Unemployed borrowers default at {emp.loc['Unemployed', 'default_rate']:.1%}, "
                     f"{emp.loc['Unemployed', 'default_rate'] / avg:.1f}x the portfolio average."),
         "so_what": ("Require proof of other income or collateral before lending "
                     "to this group.")},
        {"title": "Cross-cut: income x employment",
         "finding": (f"Low-income contract workers default at {x.loc['Q1 Low', 'Contract']:.1%}, "
                     f"but only {x.loc['Q2', 'Contract']:.1%} one band up. Unemployed borrowers "
                     f"stay at {x.loc['Q4 High', 'Unemployed']:.1%} even in the top income band."),
         "so_what": ("For contract workers, income is the key check. For the unemployed, "
                     "high stated income does not remove the risk.")},
    ]


def main():
    df = load_scored()
    k = portfolio_kpis(df)
    print("KPIs:", json.dumps(k, indent=2))
    ins = insights(df)
    for i, item in enumerate(ins, 1):
        print(f"\n{i}. {item['title']}\n   {item['finding']}\n   So what: {item['so_what']}")
    with open(DATA / "insights.json", "w") as f:
        json.dump({"kpis": k, "insights": ins}, f, indent=2)


if __name__ == "__main__":
    main()