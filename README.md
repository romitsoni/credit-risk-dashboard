# Credit Risk Monitor

A credit-risk monitoring dashboard built on 5 synthetic bank tables (10,000 customers,
14,685 loans), with two embedded AI features: an LLM **"Ask the Portfolio"** assistant
and **anomaly detection** with human review.

- **Live app:** (add link)
- **Demo video:** (add link)

## Python version
Python **3.12.6**

## Run it (3 commands)
```
pip install -r requirements.txt
python src/build.py
streamlit run app/app.py
```
`build.py` cleans and joins the data, trains both models, flags anomalies and computes
the insights. Outputs go to `data/processed/` and `models/`. The raw files in
`data/raw/` are never changed.

## LLM API key
The assistant uses **Google Gemini** (free tier works).

- **Local:** copy `.env.example` to `.env` and set `GEMINI_API_KEY=<your key>`
  (get one at aistudio.google.com). `GEMINI_MODEL` is optional.
- **Streamlit Cloud:** add `GEMINI_API_KEY = "<your key>"` in the app's **Secrets**.

**If no key is set:** the app still runs fully. The Ask tab shows "AI assistant is off",
and the **Quick questions** still answer straight from the data, without the LLM.
If the API is busy or rate-limited, the app tries backup models, then shows
"AI unavailable"; the rest of the dashboard keeps working.

## Where each part of the brief lives

| Brief item | Where |
|---|---|
| 1. Data ingestion, join logic, data-quality decisions | `src/ingest.py` â†’ `data/processed/analytic_dataset.csv`, `data_quality_log.csv`; see "Join logic" below |
| 2. Portfolio analytics + 5 insights with "so what" | `src/analytics.py`; dashboard **Portfolio** tab |
| 3. Default model (LR baseline + XGBoost), AUC / KS / Gini, SHAP | `src/model.py` â†’ `data/processed/model_metrics.json`, `feature_importance.csv` |
| 3. Model monitoring plan | Dashboard **Model & Monitoring** tab (PSI, calibration, fairness, plan) |
| 4. Dashboard: portfolio view, account view, filters, drill-down | `app/app.py`, `app/views/portfolio.py`, `app/views/account.py` |
| AI feature 1: Ask the Portfolio | `src/llm_assistant.py`, `app/views/ask.py` |
| AI feature 2: Anomaly detection + review decisions | `src/anomaly.py`, `app/views/anomalies.py` |
| Trigger / false positive / false negative / human step | "How this works" boxes in the **Ask** and **Anomaly Review** tabs |
| Architecture diagram | `docs/architecture.pdf` |
| Solution overview | `docs/solution_overview.pdf` |
| Extra: collections strategy | `src/collections_analysis.py`, `app/views/collections.py` |
| Extra: data checks and research | `src/profile_data.py`, `src/explore_data.py`, `src/deep_analysis.py` |

## Join logic
- **Base table = `loan_accounts`** (one row per loan; it holds `target_default`).
- **Repayment** (12 rows per loan) is summarized to **one row per loan** first
  (late/missed months, max and mean DPD, DPD trend, pay ratio), then joined on `account_id`.
- **Bureau** (many lines per customer) is summarized to **one row per customer**
  (lines, write-offs, DPD, utilization), then joined on `customer_id`.
- **Customers** joined on `customer_id`.
- **Collections** summarized per loan and joined for the dashboard only; **not a model
  input** (it is the bank's reaction to non-payment, so it would leak the outcome).
- All joins are **left joins** with `validate=` checks, and an `assert` confirms the result
  is still **14,685 rows**. A raw join without summarizing would create 176,220+ rows and
  count each default many times.

## Data-quality decisions
| Issue | Rows | Decision |
|---|---|---|
| Gender coded "M"/"F" | 600 | Mapped to Male/Female |
| `years_employed` > age âˆ’ 16 | 726 | Capped at age âˆ’ 16, plus a flag |
| `years_employed`, `num_dependents`, `years_at_residence` missing | 500 each | Employment-type median, plus a missing flag (no rows dropped) |
| `interest_rate_pct` missing | 294 | Loan-type median, plus a missing flag |
| Bureau `date_closed` missing | 11,243 | Kept null: these are Active lines |
| Bureau `days_past_due` missing | 689 | Left out of max/mean; count kept per customer |
| Loan dated before `customer_since` | 7,293 | `customer_since` not used as a feature |
| Customer with no bureau record | 1,462 loans | Bureau features = 0, plus a no-history flag |

## Model summary
| Model | AUC | KS | Gini |
|---|---|---|---|
| Logistic regression (baseline) | 0.6625 | 0.2563 | 0.3249 |
| XGBoost (used in the dashboard) | 0.6611 | 0.2434 | 0.3223 |

- Test set = 25% of loans, **split by customer** (no customer in both train and test).
- 5-fold cross-validation: **0.672 (LR) / 0.676 (XGBoost)**.
- Riskiest 20% of loans hold **36.7%** of all defaults. Calibration: predicted 15.7% vs actual 16.1%.
- **Not used as inputs:** IDs, dates, `customer_since`, collections data, gender and marital status (fair lending).
- Features from Kaggle Home Credit solutions were tested with 5-fold CV and added only
  +0.002 AUC (noise), so the simpler model was kept (`src/deep_analysis.py`).

## Key assumptions
- The model is a **behaviour score** of existing loans that have 12 months of history.
- `target_default` was observed after the 12 repayment months.
- Dates do not line up across tables (e.g. many collection events before the loan
  start), so no date-based features are used.
- Exposure = original loan amount.

## Notes for reviewers
- Anomaly decisions are saved to a CSV (`data/decisions/`). On Streamlit Cloud this resets
  when the app restarts; in production it would be a database with login and audit trail.
- Free Streamlit apps sleep after a period without visitors; click "wake up" (no login needed).
- The LLM free tier may use inputs to improve Google products; the data here is synthetic.
  Only the question and small aggregated tables are sent, never raw customer rows.

## Repository layout
```
README.md
requirements.txt
.env.example
app/app.py                 dashboard entry point
app/views/                 one file per dashboard tab
src/                       ingestion, analytics, model, anomaly, LLM assistant, checks
data/raw/                  the 5 CSVs as provided
data/DATA_DICTIONARY.md
data/processed/            outputs of build.py
models/                    trained models
docs/architecture.pdf
docs/solution_overview.pdf
```
