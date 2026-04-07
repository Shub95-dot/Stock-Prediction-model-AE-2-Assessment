"""
tests/smoke_pipeline.py
=======================
Fast smoke test for the DataPipeline feature engineering.
Downloads 90 days of MSFT data and verifies the feature matrix is
non-empty, has no all-NaN columns, and has the expected shape.
 
Run manually: python tests/smoke_pipeline.py
Run in CI:    called by .github/workflows/ci.yml
"""
 
import os
import sys
import datetime
 
# Suppress TF log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
from barometer_core import DataPipeline
 
 
def run():
    print("=" * 55)
    print("  DataPipeline Smoke Test")
    print("=" * 55)
 
    end   = datetime.datetime.today().strftime("%Y-%m-%d")
    start = (datetime.datetime.today() - datetime.timedelta(days=300)).strftime("%Y-%m-%d")
 
    print(f"\n  Downloading MSFT  {start} → {end} ...")
    pipe = DataPipeline(tickers=["MSFT"], start=start, end=end, window=60)
    pipe.download()
    pipe.prepare_all()
 
    df = pipe.feature_data["MSFT"]
    print(f"  Shape: {df.shape}")
 
    assert df.shape[0] > 50,  f"Too few rows: {df.shape[0]}"
    assert df.shape[1] > 30,  f"Too few feature columns: {df.shape[1]}"
 
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    assert not all_nan_cols, f"All-NaN columns detected: {all_nan_cols}"
 
    required = ["close", "rsi_14", "macd", "bb_upper", "bb_lower",
                "ema_50", "sma_200", "target_1d", "target_5d"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
 
    print(f"  Columns: {list(df.columns[:8])} ...")
    print(f"\n  PASS — Pipeline produces {df.shape[1]} features over {df.shape[0]} rows.")
    print("=" * 55)
 
 
if __name__ == "__main__":