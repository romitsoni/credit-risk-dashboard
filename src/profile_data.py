import pandas as pd

d = "data/raw/"
names = ["customers", "loan_accounts", "bureau_credit",
         "repayment_history", "collections_events"]
t = {n: pd.read_csv(d + n + ".csv") for n in names}

for n, df in t.items():
    print("=" * 60)
    print(n, "shape:", df.shape)
    print("duplicate rows:", df.duplicated().sum())
    print("missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])
    for c in df.select_dtypes("object").columns:
        if df[c].nunique() <= 15:
            print("\n", c)
            print(df[c].value_counts(dropna=False))
    print("\nnumbers:")
    print(df.describe().T.to_string())

print("=" * 60)
c, l, b, r, e = (t[n] for n in names)
print("duplicate customer_id:", c.customer_id.duplicated().sum())
print("duplicate account_id:", l.account_id.duplicated().sum())
print("loans with unknown customer:", (~l.customer_id.isin(c.customer_id)).sum())
print("bureau with unknown customer:", (~b.customer_id.isin(c.customer_id)).sum())
print("repayment with unknown account:", (~r.account_id.isin(l.account_id)).sum())
print("collections with unknown account:", (~e.account_id.isin(l.account_id)).sum())
print("duplicate account+month in repayment:", r.duplicated(["account_id", "payment_month"]).sum())
print("default rate:", l.target_default.mean())