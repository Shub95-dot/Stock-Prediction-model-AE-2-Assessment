import sys
from barometer_core import BarometerSystem
import pandas as pd
import yfinance as yf


def diagnose():
    print("Loading NVDA...")
    sys_obj = BarometerSystem("NVDA", window=60)
    sys_obj.load("barometer_saved/NVDA")

    print("\nFetching last 100 days of NVDA data...")
    df = yf.download("NVDA", period="100d", auto_adjust=True)

    # We use predict_next to see what the models spit out
    # Actually wait, test_inference.py does this perfectly.
    # Let's just output raw base model predictions.


diagnose()
