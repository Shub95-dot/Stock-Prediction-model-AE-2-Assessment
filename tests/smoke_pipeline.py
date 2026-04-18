"""
tests/smoke_pipeline.py
=======================
Smoke test for the DataPipeline feature engineering.

Downloads ~300 days of MSFT data and verifies the feature matrix is
non-empty, has no all-NaN columns, and contains the expected columns.

Run:  pytest tests/smoke_pipeline.py -v
"""

import datetime
import os
import sys

import pytest

# Suppress TF log noise before any import of barometer_core
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from barometer_core import DataPipeline  # noqa: E402

# ── Shared fixture — one download for all tests in this module ─────────────────

@pytest.fixture(scope="module")
def msft_df():
    """Download 300 days of MSFT data and engineer features once per session."""
    end   = datetime.datetime.today().strftime("%Y-%m-%d")
    start = (datetime.datetime.today() - datetime.timedelta(days=300)).strftime(
        "%Y-%m-%d"
    )
    pipe = DataPipeline(tickers=["MSFT"], start=start, end=end, window=60)
    pipe.download()
    pipe.prepare_all()
    return pipe.feature_data["MSFT"]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_non_empty(msft_df):
    """Feature DataFrame must have at least 50 rows and 30 columns."""
    assert msft_df.shape[0] > 50,  f"Too few rows: {msft_df.shape[0]}"
    assert msft_df.shape[1] > 30,  f"Too few feature columns: {msft_df.shape[1]}"


def test_no_all_nan_columns(msft_df):
    """No column should be entirely NaN — that signals a broken indicator."""
    all_nan_cols = [c for c in msft_df.columns if msft_df[c].isna().all()]
    assert not all_nan_cols, f"All-NaN columns detected: {all_nan_cols}"


@pytest.mark.parametrize("col", [
    "close", "rsi_14", "macd", "bb_upper", "bb_lower",
    "ema_50", "sma_200",
    "target_1d", "target_5d", "target_21d", "target_63d",
    "dir_1d", "dir_5d", "dir_21d", "dir_63d",   # all four direction columns
])
def test_required_columns_present(msft_df, col):
    """Core columns (price, indicators, targets, direction flags) must exist."""
    assert col in msft_df.columns, f"Missing required column: {col}"
