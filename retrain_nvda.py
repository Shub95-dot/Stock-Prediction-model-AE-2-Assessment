import os

from barometer_core import BarometerSystem, DataPipeline


def retrain_nvda():
    ticker = "NVDA"
    print(f"--- Full Retrain for {ticker} ---")

    # We download ~5 years of data for training
    pipe = DataPipeline(tickers=[ticker], start="2020-01-01", end="2026-03-29")
    pipe.download()
    pipe.prepare_all()

    df = pipe.feature_data[ticker]
    vix = pipe.raw["Close"]["^VIX"].reindex(df.index).ffill().bfill()
    spy = pipe.raw["Close"]["SPY"].pct_change().reindex(df.index).ffill().bfill()

    sys_obj = BarometerSystem(ticker)

    # Completely fit the system
    import time

    t0 = time.time()
    sys_obj.fit(df, vix, spy)

    # Save over the old models
    path = f"barometer_saved/{ticker}"
    os.makedirs(path, exist_ok=True)
    sys_obj.save(path)
    print(
        f"[{ticker}] Retraining complete. Saved to {path}. Time = {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    retrain_nvda()
