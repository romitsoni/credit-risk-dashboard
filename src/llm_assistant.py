"""
llm_assistant.py
"Ask the Portfolio": natural-language Q&A grounded in the real data.

Design (no hallucinated numbers):
 1. LLM call 1 (planner) sees ONLY the question + tool list, never data.
    It returns JSON: which tool to run, or "decline".
 2. Our code validates the tool and computes the numbers with pandas.
 3. LLM call 2 (writer) sees the question + the small result table only.
 4. Our code checks every number in the answer exists in the result.
    If any number fails, the LLM text is replaced by a code-written answer.
"""

import json
import time
import os
import re
import numbers
import pandas as pd
from dotenv import load_dotenv

import analytics

load_dotenv()
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
DIMS = analytics.DIMENSIONS
SEGMENTS = ["Low", "Medium", "High", "Very High"]


class ToolError(Exception):
    pass


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
PLANNER = """You route questions for a credit-risk portfolio assistant.
You NEVER answer the question yourself and NEVER write numbers about the data.
Pick exactly ONE tool that can answer the question from the portfolio data.

Tools:
- kpis: portfolio totals (loans, customers, total exposure, observed default rate, average PD, high-risk exposure share). args: {{}}
  Use kpis also for "expected" or "predicted" default rate of the CURRENT portfolio (that is the average PD from the model).- profile: per-group accounts, defaults, default_rate, exposure, exposure_share, avg_pd. args: {{"dim": one of {dims}}}
- cross: default rate for every row x column pair of two dimensions. args: {{"row": dim, "col": dim}}
- account: one loan's PD score, risk segment, top drivers and anomaly flags. args: {{"account_id": "ACC2xxxxx"}}
- top_accounts: highest-PD accounts. args: {{"n": 1-20, "loan_type": optional, "segment": optional one of {segs}}}
- anomalies: number of flagged accounts, flags by rule, default rate flagged vs not. args: {{}}
- model_metrics: AUC, KS, Gini of the logistic regression and XGBoost models. args: {{}}

Loan types: Personal Loan, Auto Loan, Credit Card, Mortgage, Business Loan.
Employment types: Salaried, Self-Employed, Business Owner, Contract, Unemployed.

If the data cannot answer it (future forecasts, other banks, market data, opinions,
individual customer personal details, anything not in these tools), use:
{{"tool": "decline", "args": {{"reason": "<short reason>"}}}}

Return JSON only, like {{"tool": "profile", "args": {{"dim": "loan_type"}}}}

Question: {question}"""

WRITER = """You are a credit-risk analyst assistant. Answer the question in 1-3 short,
plain-English sentences using ONLY the numbers in DATA.
Rules:
- Do not calculate new numbers (no sums, differences or ratios) unless they are in DATA.
- Show rates and shares as percentages with 1 decimal (0.2472 -> 24.7%).
- Show AUC, KS and Gini exactly as given, as decimals (0.6611 stays 0.661). Never as %.
- Money: 1 million or more -> millions with 1 decimal (314473817 -> 314.5M);
  below 1 million -> thousands with 1 decimal (98500 -> 98.5K). No currency symbol.
- When you mention a driver, say it raises the risk; do not repeat the +numbers.
- Do not add a source line; the app adds it.

Question: {question}
DATA: {data}"""


# ------------------------------------------------------------------
# LLM call (Gemini) with model fallback
# ------------------------------------------------------------------
def has_key():
    return bool(os.getenv("GEMINI_API_KEY"))


def _models():
    first = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    backup = ["gemini-3.5-flash", "gemini-3.5-flash-lite",
              "gemini-flash-latest", "gemini-flash-lite-latest"]
    return [first] + [m for m in backup if m != first]


def _llm(prompt, json_mode=False):
    """Try each model in order, with one retry. Any server error (5xx),
    rate limit (429) or retired model (404) moves to the next model."""
    from google import genai
    from google.genai import types, errors
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if json_mode else "text/plain")
    last_err = None
    for model in _models():
        for attempt in range(2):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt, config=cfg)
                return (r.text or "").strip()
            except errors.ServerError as e:      # 5xx: busy / temporary
                last_err = e
                time.sleep(1.5)
            except errors.ClientError as e:      # 4xx
                last_err = e
                if getattr(e, "code", None) in (404, 429):
                    break                        # go to next model
                raise                            # bad key etc.: stop
    raise last_err


def _parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# ------------------------------------------------------------------
# Tools: the ONLY way numbers are produced
# ------------------------------------------------------------------
def run_tool(tool, args, df, anomalies, metrics):
    if tool == "kpis":
        return analytics.portfolio_kpis(df), "analytics.portfolio_kpis()"

    if tool == "profile":
        dim = args.get("dim")
        if dim not in DIMS:
            raise ToolError(f"'{dim}' is not a dimension in this data.")
        t = analytics.profile(df, dim)[[dim, "accounts", "defaults", "default_rate",
                                        "exposure", "exposure_share"]]
        return t.round(4), f"analytics.profile('{dim}')"

    if tool == "cross":
        row, col = args.get("row"), args.get("col")
        if row not in DIMS or col not in DIMS or row == col:
            raise ToolError("Those two dimensions cannot be crossed in this data.")
        t = analytics.cross(df, row, col).round(4).reset_index()
        t.columns = [str(c) for c in t.columns]
        return t, f"analytics.cross('{row}', '{col}') - default rate"

    if tool == "account":
        aid = str(args.get("account_id", "")).strip().upper()
        r = df[df["account_id"] == aid]
        if r.empty:
            raise ToolError(f"Account {aid} was not found.")
        r = r.iloc[0]
        a = anomalies[anomalies["account_id"] == aid]
        return {"account_id": aid, "loan_type": r["loan_type"],
                "loan_amount": round(float(r["loan_amount"]), 2),
                "pd_score": round(float(r["pd_score"]), 4),
                "risk_segment": r["risk_segment"],
                "top_drivers": r["top_drivers"],
                "anomaly_flags": a["reasons"].iloc[0] if len(a) else "none"
                }, "scored_accounts.csv + anomalies.csv"

    if tool == "top_accounts":
        n = max(1, min(int(args.get("n") or 10), 20))
        t = df
        lt, seg = args.get("loan_type"), args.get("segment")
        if lt:
            if lt not in df["loan_type"].unique():
                raise ToolError(f"'{lt}' is not a loan type in this data.")
            t = t[t["loan_type"] == lt]
        if seg:
            if seg not in SEGMENTS:
                raise ToolError(f"'{seg}' is not a risk segment.")
            t = t[t["risk_segment"] == seg]
        t = t.nlargest(n, "pd_score")[["account_id", "loan_type", "loan_amount",
                                        "pd_score", "risk_segment"]].round(4)
        return t, f"scored_accounts.csv - top {n} by pd_score"

    if tool == "anomalies":
        by_rule = anomalies["flags"].str.split(", ").explode().value_counts()
        rest = df[~df["account_id"].isin(anomalies["account_id"])]
        return {"flagged_accounts": int(len(anomalies)),
                "total_accounts": int(len(df)),
                "flags_by_rule": {k: int(v) for k, v in by_rule.items()},
                "flagged_default_rate": round(float(anomalies["target_default"].mean()), 4),
                "not_flagged_default_rate": round(float(rest["target_default"].mean()), 4),
                }, "anomalies.csv"

    if tool == "model_metrics":
        return {k: metrics[k] for k in ["logistic_regression", "xgboost"]}, \
            "model_metrics.json (test set)"

    raise ToolError("That question is outside what this assistant can answer.")


# ------------------------------------------------------------------
# Safe code-written answers (used when LLM fails the check or no key)
# ------------------------------------------------------------------
def _money(v):
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.1f}K"


def template_answer(tool, args, result):
    if tool == "kpis":
        k = result
        return (f"The portfolio has {k['accounts']:,} loans for {k['customers']:,} customers, "
                f"total exposure {_money(k['total_exposure'])}, and an observed default "
                f"rate of {k['default_rate']:.1%}.")
    if tool == "profile":
        dim = args["dim"]
        t = result.sort_values("default_rate", ascending=False)
        hi, lo = t.iloc[0], t.iloc[-1]
        return (f"By {DIMS[dim].lower()}, {hi[dim]} has the highest default rate "
                f"({hi['default_rate']:.1%}) and {lo[dim]} the lowest ({lo['default_rate']:.1%}).")
    if tool == "cross":
        s = result.set_index(result.columns[0]).stack()
        (r, c), v = s.idxmax(), s.max()
        return f"The highest default rate is {r} x {c} at {v:.1%}."
    if tool == "account":
        return (f"{result['account_id']} ({result['loan_type']}, {_money(result['loan_amount'])}) "
                f"has a PD of {result['pd_score']:.1%} ({result['risk_segment']} risk). "
                f"Main drivers: {result['top_drivers'] or 'none above baseline'}.")
    if tool == "top_accounts":
        ids = ", ".join(f"{r.account_id} ({r.pd_score:.1%})" for r in result.itertuples())
        return f"Highest-risk accounts: {ids}."
    if tool == "anomalies":
        return (f"{result['flagged_accounts']:,} of {result['total_accounts']:,} accounts are "
                f"flagged. Flagged accounts default at {result['flagged_default_rate']:.1%} "
                f"vs {result['not_flagged_default_rate']:.1%} for the rest.")
    if tool == "model_metrics":
        x, l = result["xgboost"], result["logistic_regression"]
        return (f"On the test set, XGBoost has AUC {x['AUC']:.3f}, KS {x['KS']:.3f}, "
                f"Gini {x['Gini']:.3f}; logistic regression has AUC {l['AUC']:.3f}, "
                f"KS {l['KS']:.3f}, Gini {l['Gini']:.3f}.")
    return "No answer."


# ------------------------------------------------------------------
# Number check: every number in the answer must come from the result
# ------------------------------------------------------------------
def _to_jsonable(result):
    if isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    return result


def _collect_numbers(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_numbers(k, out)
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, numbers.Number):
        if not pd.isna(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(x.replace(",", "")) for x in NUM_RE.findall(obj))


def verify(answer, data, question):
    allowed = []
    _collect_numbers(data, allowed)
    _collect_numbers(question, allowed)
    variants = [v * f for v in allowed for f in (1, 100, 1e-3, 1e-6, 1e-9)]
    bad = []
    for s in NUM_RE.findall(answer):
        a = float(s.replace(",", ""))
        if not any(abs(a - b) <= max(0.051, 0.006 * abs(b)) for b in variants):
            bad.append(s)
    return len(bad) == 0, bad


# ------------------------------------------------------------------
# Main entry point used by the dashboard
# ------------------------------------------------------------------
def ask(question, df, anomalies, metrics):
    out = {"question": question, "tool": None, "args": {}, "table": None,
           "source": None, "unverified_numbers": []}

    if not has_key():
        out.update(status="no_key", answer=(
            "The AI assistant is off because no GEMINI_API_KEY is set. "
            "Use the quick questions below; they are answered directly from the data."))
        return out

    try:
        plan = _parse_json(_llm(PLANNER.format(
            question=question, dims=list(DIMS), segs=SEGMENTS), json_mode=True))
    except Exception as e:
        out.update(status="error", answer=(
            f"The AI service is not available right now "
            f"({type(e).__name__} {getattr(e, 'code', '')}). "
            "The rest of the dashboard still works."))
        return out

    tool, args = plan.get("tool"), plan.get("args") or {}
    out.update(tool=tool, args=args)

    if tool == "decline":
        out.update(status="declined", answer=(
            "I can't answer that from this portfolio data. " + args.get("reason", "")))
        return out

    try:
        result, source = run_tool(tool, args, df, anomalies, metrics)
    except (ToolError, ValueError, TypeError) as e:
        out.update(status="declined", answer=f"I can't answer that from this data. {e}")
        return out

    data = _to_jsonable(result)
    out.update(source=source, table=result)

    try:
        answer = _llm(WRITER.format(question=question,
                                    data=json.dumps(data, default=str)))
    except Exception:
        answer = ""

    ok, bad = verify(answer, data, question) if answer else (False, [])
    if ok:
        out.update(status="verified", answer=answer)
    else:
        out.update(status="fallback", answer=template_answer(tool, args, result),
                   unverified_numbers=bad)
    return out


def answer_without_llm(tool, args, df, anomalies, metrics):
    """Quick questions: no LLM at all. Used when there is no key."""
    result, source = run_tool(tool, args, df, anomalies, metrics)
    return {"status": "direct", "answer": template_answer(tool, args, result),
            "tool": tool, "args": args, "table": result, "source": source,
            "unverified_numbers": []}


if __name__ == "__main__":
    df = analytics.load_scored()
    an = pd.read_csv(analytics.DATA / "anomalies.csv")
    with open(analytics.DATA / "model_metrics.json") as f:
        mt = json.load(f)
    for q in ["Which loan type has the highest default rate?",
              "What is the default rate for unemployed people in the lowest income band?",
              "Tell me about account ACC202884",
              "How good is the model?",
              "What will the default rate be next year?"]:
        r = ask(q, df, an, mt)
        print(f"\nQ: {q}\n[{r['status']}] tool={r['tool']} args={r['args']}\nA: {r['answer']}")
        if r["unverified_numbers"]:
            print("   (LLM numbers that failed the check:", r["unverified_numbers"], ")")