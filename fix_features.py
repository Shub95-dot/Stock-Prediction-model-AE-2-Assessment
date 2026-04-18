import datetime
import os

import dotenv
import joblib

from barometer_core import BarometerSystem, DataPipeline

dotenv.load_dotenv()


def fix():
    start = "2020-01-01"
    end = datetime.datetime.today().strftime("%Y-%m-%d")

    # We only need one ticker to get the common feature set
    tickers = ["MSFT", "AMZN", "AMGN", "NVDA"]
    pipeline = DataPipeline(tickers=tickers, start=start, end=end)
    pipeline.download()
    pipeline.prepare_all()

    for t in tickers:
        df = pipeline.feature_data[t]
        cols = BarometerSystem._feature_cols_from(df)
        path = f"barometer_saved/{t}"
        if os.path.exists(path):
            joblib.dump(cols, f"{path}/features.pkl")
            print(f"Saved features.pkl to {path}")
        else:
            print(f"Path not found: {path}")


if __name__ == "__main__":
    fix()
