import pandas as pd
import numpy as np

d = "data/raw/"
c = pd.read_csv(d + "customers.csv")
l = pd.read_csv(d + "loan_accounts.csv")
b = pd.read_csv(d + "bureau_credit.csv")
r = pd.read_csv(d + "repayment_history.csv")
e = pd.read_csv(d + "collections_events.csv")
c["gender"] = c["gender"].replace({"M": "Male", "F": "Female"})

def title(t): print("\n" + "=" * 60 + "\n" + t + "\n" + "=" * 60)

title("1. RELATIONSHIPS")
print("Loans per customer:\n", l.customer_id.value_counts().value_counts().sort_index())
print("Customers with no loan:", (~c.customer_id.isin(l.customer_id)).sum())
print("Customers with no bureau:", (~c.customer_id.isin(b.customer_id)).sum())
print("Bureau lines per customer:\n", b.customer_id.value_counts().describe())
print("Months per account:\n", r.account_id.value_counts().value_counts())
print("Accounts with collections:", e.account_id.nunique())

title("2. LOGIC CHECKS (should be 0 or explainable)")
print("Collateral flag vs loan type:\n", pd.crosstab(l.loan_type, l.collateral_flag))
print("emi_paid > emi_due:", (r.emi_paid > r.emi_due).sum())
print("Missed but paid > 0:", ((r.payment_status == "Missed") & (r.emi_paid > 0)).sum())
print("On-Time but DPD > 0:", ((r.payment_status == "On-Time") & (r.days_past_due > 0)).sum())
print("DPD by status:\n", r.groupby("payment_status").days_past_due.describe())
print("Bureau outstanding > sanctioned:", (b.outstanding_amount > b.sanctioned_amount).sum())
print("Closed/Written-Off with no close date:",
      ((b.account_status != "Active") & b.date_closed.isna()).sum())
print("Bureau close date before open date:",
      (pd.to_datetime(b.date_closed) < pd.to_datetime(b.date_opened)).sum())
lc = l.merge(c[["customer_id", "customer_since"]], on="customer_id")
print("Loan before customer_since:",
      (pd.to_datetime(lc.origination_date) < pd.to_datetime(lc.customer_since)).sum())
print("Unemployed with years_employed > 0:",
      ((c.employment_type == "Unemployed") & (c.years_employed > 0)).sum())
print("Years employed > age - 16:", (c.years_employed > c.age - 16).sum())
print("Income by employment type:\n", c.groupby("employment_type").annual_income.describe())
print("Origination date range:", l.origination_date.min(), "to", l.origination_date.max())

df = l.merge(c, on="customer_id")
df["age_band"] = pd.cut(df.age, [0, 30, 40, 50, 60, 200], right=False,
                        labels=["<30", "30-39", "40-49", "50-59", "60+"])
df["income_band"] = pd.qcut(df.annual_income, 4, labels=["Q1 Low", "Q2", "Q3", "Q4 High"])

title("3. IS MISSING DATA RANDOM? (default rate missing vs not)")
for col in ["years_employed", "num_dependents", "years_at_residence", "interest_rate_pct"]:
    print(col, df.groupby(df[col].isna()).target_default.mean().round(3).to_dict())
print("Missing years_employed by employment type:\n",
      df[df.years_employed.isna()].employment_type.value_counts())

title("4. DEFAULT RATE BY DIMENSION")
for col in ["loan_type", "employment_type", "region", "city_tier", "age_band",
            "income_band", "housing_type", "education_level", "disbursal_channel",
            "loan_purpose", "marital_status", "gender"]:
    t = df.groupby(col, observed=True).agg(loans=("account_id", "count"),
                                           default_rate=("target_default", "mean"),
                                           exposure=("loan_amount", "sum"))
    print("\n", t.round(3).sort_values("default_rate", ascending=False))

title("5. CROSS CUT: income band x employment type (default rate)")
print(pd.crosstab(df.income_band, df.employment_type,
                  values=df.target_default, aggfunc="mean").round(3))

title("6. BUREAU SIGNAL")
bw = b.groupby("customer_id").account_status.apply(lambda s: (s == "Written-Off").any())
df["bur_writeoff"] = df.customer_id.map(bw).fillna(False)
print(df.groupby("bur_writeoff").target_default.agg(["count", "mean"]).round(3))

title("7. REPAYMENT SIGNAL + LEAKAGE CHECK")
rm = r.groupby("account_id").agg(missed=("payment_status", lambda s: (s == "Missed").sum()))
df = df.merge(rm, on="account_id")
print("Default rate by missed months:\n",
      df.groupby("missed").target_default.agg(["count", "mean"]).round(3))
df["has_coll"] = df.account_id.isin(e.account_id)
print("\nCollections vs default (leakage check):\n",
      pd.crosstab(df.has_coll, df.target_default, normalize="index").round(3))

title("8. OUTLIERS")
print(df.groupby("loan_type").loan_amount.describe().round(0))
df["lti"] = df.loan_amount / df.annual_income
print("\nLoan-to-income:\n", df.lti.describe().round(2))