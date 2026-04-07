import datetime
import logging
import os

import pandas as pd
import yfinance as yf

from barometer_core import BarometerSystem, DataPipeline

# Configure logging to see the barometer actions
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("InferenceTest")


def run_test():
    ticker = "MSFT"
    model_path = f"barometer_saved/{ticker}"

    print("\n" + "═" * 72)
    print(f"  SOLiGence — LOCAL INFERENCE TEST: {ticker}")
    print("═" * 72)

    # 1. Reconstruct and Load the System
    print(f"\n  [1/4] Loading pre-trained system from {model_path}...")
    system = BarometerSystem(ticker=ticker, window=60)
    system.load(model_path)

    # 2. Fetch Latest Market Data
    # We need enough data for the 60-day window + indicators (like SMA-200)
    # Using 2 years (730 days) to be safe.
    print(f"  [2/4] Fetching latest live data for {ticker}...")
    start_date = (datetime.datetime.today() - datetime.timedelta(days=730)).strftime(
        "%Y-%m-%d"
    )
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")

    pipeline = DataPipeline(tickers=[ticker], start=start_date, end=end_date)
    pipeline.download()
    pipeline.prepare_all()

    df = pipeline.feature_data[ticker]
    vix = pipeline.raw["Close"]["^VIX"].reindex(df.index).ffill().bfill()
    spy = pipeline.raw["Close"]["SPY"].pct_change().reindex(df.index).ffill().bfill()

    # 3. Generate Signals
    print(f"  [3/4] Generating real-time signals...")
    # Inject latest data into system memory for what-if
    system._last_df = df
    system._last_vix = vix
    system._last_spy = spy

    signals = system.generate_signal(conf_threshold=0.55)

    print(f"\n  📡 LATEST SIGNALS — {ticker}")
    print(f"  {'─' * 40}")
    for h, sig in signals.items():
        print(
            f"  {h.upper():4s}: {sig['signal']} | Conf: {sig['confidence']:.1%} | Pred: ${sig['price_pred']:.2f}"
        )

    # 4. Stress Test (What-If Analysis)
    print(f"\n  [4/4] Executing Stress Test (Simulation)...")
    print(f"  Scenario: Immediate -5% Price Crash + 10 Index Volatility Points")

    wif = system.what_if(price_shock=-0.05, volume_shock=0.20, vix_shock=10.0)

    print(f"\n  🔮 STRESS TEST RESULTS")
    print(f"  {'─' * 40}")
    for h in ["t1", "t5", "t21"]:
        bp = wif["base"][h]["price"]
        sp = wif["shocked"][h]["price"]
        delta = sp - bp
        impact = "PROTECTIVE" if delta > -2 else "FOLLOWING"
        print(
            f"  {h.upper():4s}: Base ${bp:.2f} → Shocked ${sp:.2f} (Δ ${delta:+.2f}) [{impact}]"
        )

    print("\n" + "═" * 72)
    print("  ✅ LOCAL TEST COMPLETE")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    run_test()
