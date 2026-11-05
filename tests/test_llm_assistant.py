"""Ask the Portfolio: grounded answers, caught hallucinations, safe failures.

Gemini is replaced by a fake function, so these tests need no API key
and no internet, and give the same result every time.
"""
import pytest

import analytics
import llm_assistant as llm


# ---------- Number check ----------
DATA = [{"loan_type": "Mortgage", "default_rate": 0.2472, "exposure": 314473817.0},
        {"loan_type": "Auto Loan", "default_rate": 0.1388, "exposure": 78795810.0}]


def test_verify_accepts_correctly_formatted_numbers():
    ok, bad = llm.verify("Mortgage is highest at 24.7% with 314.5M exposure; "
                         "Auto Loan is 13.9%.", DATA, "which is highest?")
    assert ok and bad == []


def test_verify_catches_a_made_up_number():
    ok, bad = llm.verify("Mortgage defaults at 31.2%.", DATA, "which is highest?")
    assert not ok and "31.2" in bad


def test_verify_allows_numbers_from_the_question():
    ok, _ = llm.verify("Here are the top 5 accounts.", DATA, "show the top 5")
    assert ok


def test_parse_json_handles_code_fences():
    assert llm._parse_json('```json\n{"tool": "kpis", "args": {}}\n```')["tool"] == "kpis"


# ---------- Tools refuse anything outside the data ----------
def test_unknown_dimension_is_refused(scored, anomalies, metrics):
    with pytest.raises(llm.ToolError):
        llm.run_tool("profile", {"dim": "favourite_colour"}, scored, anomalies, metrics)


def test_unknown_tool_is_refused(scored, anomalies, metrics):
    with pytest.raises(llm.ToolError):
        llm.run_tool("delete_everything", {}, scored, anomalies, metrics)


def test_unknown_account_is_refused(scored, anomalies, metrics):
    with pytest.raises(llm.ToolError):
        llm.run_tool("account", {"account_id": "ACC999999"}, scored, anomalies, metrics)


def test_top_accounts_is_capped_at_20(scored, anomalies, metrics):
    t, _ = llm.run_tool("top_accounts", {"n": 999}, scored, anomalies, metrics)
    assert len(t) == 20
    assert t["pd_score"].is_monotonic_decreasing


def test_bad_loan_type_is_refused(scored, anomalies, metrics):
    with pytest.raises(llm.ToolError):
        llm.run_tool("top_accounts", {"loan_type": "Spaceship Loan"}, scored, anomalies, metrics)


# ---------- End-to-end with a fake Gemini ----------
def fake_gemini(plan, answer):
    def _fake(prompt, json_mode=False):
        return plan if json_mode else answer
    return _fake


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_no_key_does_not_crash(monkeypatch, scored, anomalies, metrics):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = llm.ask("Which loan type is riskiest?", scored, anomalies, metrics)
    assert r["status"] == "no_key"


def test_api_down_does_not_crash(monkeypatch, key, scored, anomalies, metrics):
    def boom(prompt, json_mode=False):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(llm, "_llm", boom)
    r = llm.ask("Which loan type is riskiest?", scored, anomalies, metrics)
    assert r["status"] == "error"


def test_off_topic_question_is_declined(monkeypatch, key, scored, anomalies, metrics):
    monkeypatch.setattr(llm, "_llm", fake_gemini(
        '{"tool": "decline", "args": {"reason": "Not about the portfolio."}}', ""))
    r = llm.ask("Which AI tool is best?", scored, anomalies, metrics)
    assert r["status"] == "declined"


def test_correct_answer_is_verified_with_source(monkeypatch, key, scored, anomalies, metrics):
    t = analytics.profile(scored, "loan_type")
    top = t.sort_values("default_rate", ascending=False).iloc[0]
    answer = f"{top['loan_type']} has the highest default rate at {top['default_rate']:.1%}."
    monkeypatch.setattr(llm, "_llm", fake_gemini(
        '{"tool": "profile", "args": {"dim": "loan_type"}}', answer))
    r = llm.ask("Which loan type has the highest default rate?", scored, anomalies, metrics)
    assert r["status"] == "verified"
    assert r["source"] and r["table"] is not None


def test_hallucinated_number_is_replaced_by_safe_answer(monkeypatch, key, scored, anomalies, metrics):
    monkeypatch.setattr(llm, "_llm", fake_gemini(
        '{"tool": "profile", "args": {"dim": "loan_type"}}',
        "Mortgage has the highest default rate at 99.9%."))
    r = llm.ask("Which loan type has the highest default rate?", scored, anomalies, metrics)
    assert r["status"] == "fallback"
    assert "99.9" in r["unverified_numbers"]
    assert "99.9" not in r["answer"]


def test_planner_picking_a_fake_dimension_is_declined(monkeypatch, key, scored, anomalies, metrics):
    monkeypatch.setattr(llm, "_llm", fake_gemini(
        '{"tool": "profile", "args": {"dim": "zodiac_sign"}}', ""))
    r = llm.ask("Default rate by zodiac sign?", scored, anomalies, metrics)
    assert r["status"] == "declined"