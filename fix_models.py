import os
import joblib
import datetime
import dotenv
import numpy as np
import pandas as pd
from barometer_core import DataPipeline, BarometerSystem

dotenv.load_dotenv()

def fix():
    start = "2020-01-01"
    end = datetime.datetime.today().strftime("%Y-%m-%d")
    
    tickers = ["MSFT", "AMZN", "AMGN", "NVDA"]
    # We need to re-download a bit of history to correctly fit/estimate HMM params
    pipeline = DataPipeline(tickers=tickers, start=start, end=end)
    pipeline.download()
    pipeline.prepare_all()
    
    # We also need VIX and SPY for HMM fitting
    vix = pipeline.raw["Close"]["^VIX"].ffill().bfill()
    
    for t in tickers:
        df = pipeline.feature_data[t]
        cols = BarometerSystem._feature_cols_from(df)
        path = f"barometer_saved/{t}"
        
        if os.path.exists(path):
            # 1. Save features
            joblib.dump(cols, f"{path}/features.pkl")
            
            # 2. Estimate HMM meta-params
            # We don't need to re-fit the HMM model (we already have hmm.pkl), 
            # but we need the fitted=True flag and the vix mean/std used during its training.
            # Since we used the last 3-10 years for training, we'll re-estimate those stats.
            gate_meta = {
                "fitted": True,
                "vix_mean": float(vix.mean()),
                "vix_std": float(vix.std())
            }
            joblib.dump(gate_meta, f"{path}/gate_meta.pkl")
            print(f"Repaired model at {path}")
        else:
            print(f"Path not found: {path}")

if __name__ == "__main__":
    fix()
