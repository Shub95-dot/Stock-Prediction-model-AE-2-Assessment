import os

import lightgbm as lgb
import numpy as np

from barometer_core import BarometerSystem, DataPipeline


def train_missing_t63_meta():
    tickers = ["MSFT", "AMZN", "AMGN", "NVDA"]
    # Download 2 years to quickly train a small meta learner,
    # or maybe whatever data is sufficient.
    start = "2020-01-01"

    pipe = DataPipeline(tickers=tickers, start=start, end="2026-03-29")
    pipe.download()
    pipe.prepare_all()
    vix = pipe.raw["Close"]["^VIX"].ffill().bfill()
    spy = pipe.raw["Close"]["SPY"].pct_change().ffill().bfill()

    for t in tickers:
        print(f"Training T+63 Meta Learner for {t}")
        path = f"barometer_saved/{t}"
        if not os.path.exists(path):
            continue

        sys_obj = BarometerSystem(t)
        sys_obj.load(path)

        df = pipe.feature_data[t]
        vix_t = vix.reindex(df.index).ffill().bfill()
        spy_t = spy.reindex(df.index).ffill().bfill()

        # Build sequences
        sys_obj._feature_cols = sys_obj._feature_cols_from(df)
        X_seq, y63, sc = sys_obj._sequences(df, "target_63d", scaler=sys_obj.scaler)
        n = len(X_seq)
        df_a = df.iloc[-n:]
        vix_a = vix_t.values[-n:]
        spy_a = spy_t.values[-n:]

        regime = sys_obj.gate.compute_regime_vector(df_a, vix_a, spy_a)

        all_preds = {nm: b.predict(X_seq) for nm, b in sys_obj.barometers.items()}

        # Now fit ONLY t63
        meta_features = sys_obj.meta._assemble_meta_features(all_preds, regime)

        reg_params = sys_obj.meta.base_params
        reg = lgb.LGBMRegressor(**reg_params)
        reg.fit(meta_features, y63)
        sys_obj.meta.reg_models["t63"] = reg

        clf_params = {**reg_params, "objective": "binary", "metric": "auc"}
        clf = lgb.LGBMClassifier(**clf_params)
        y_dir = (y63 > np.median(y63)).astype(int)
        clf.fit(meta_features, y_dir)
        sys_obj.meta.clf_models["t63"] = clf

        # Save them as .lgb to match others
        reg.booster_.save_model(f"{path}/meta_reg_t63.lgb")
        clf.booster_.save_model(f"{path}/meta_clf_t63.lgb")
        print(f"[{t}] T+63 Meta Learner saved.")


if __name__ == "__main__":
    train_missing_t63_meta()
