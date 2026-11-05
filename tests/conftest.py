"""Shared setup for all tests.

The tests read the outputs of `python src/build.py` (data/processed/).
LLM tests use a fake Gemini, so no API key or internet is needed.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # the pipeline uses paths relative to the project root

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def processed():
    if not (PROCESSED / "scored_accounts.csv").exists():
        pytest.skip("Run `python src/build.py` first")
    return PROCESSED


@pytest.fixture(scope="session")
def analytic(processed):
    return pd.read_csv(processed / "analytic_dataset.csv")


@pytest.fixture(scope="session")
def scored(processed):
    return pd.read_csv(processed / "scored_accounts.csv")


@pytest.fixture(scope="session")
def anomalies(processed):
    return pd.read_csv(processed / "anomalies.csv")


@pytest.fixture(scope="session")
def metrics(processed):
    with open(processed / "model_metrics.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def raw_loans():
    return pd.read_csv(RAW / "loan_accounts.csv")