import os
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np

from barometer_core import BarometerSystem, DataPipeline


def train_missing_t63_meta():
    tickers = ["MSFT", "AMZN", "AMGN", "NVDA"]
    # Download 5 years to give the T+63 meta-learner enough training signal.
    start = "2020-01-01"
    end = datetime.today().strftime("%Y-%m-%d")  # always use today

    pipe = DataPipeline(tickers=tickers, start=start, end=end)
    pipe.download()
    pipe.prepare_all()
    vix = pipe.raw["Close"]["^VIX"].ffill().bfill()
    spy = pipe.raw["Close"]["SPY"].pct_change().ffill().bfill()

    for t in tickers:
        print(f"\nTraining T+63 Meta Learner for {t}")
        path = f"barometer_saved/{t}"
        if not os.path.exists(path):
            print(f"  [{t}] No saved model directory at {path} — skipping.")
            continue

        # ── Load the existing trained system ──────────────────────────────────
        sys_obj = BarometerSystem(t)
        sys_obj.load(path)

        df = pipe.feature_data[t]
        vix_t = vix.reindex(df.index).ffill().bfill()
        spy_t = spy.reindex(df.index).ffill().bfill()

        # ── Build sequences ───────────────────────────────────────────────────
        sys_obj._feature_cols = sys_obj._feature_cols_from(df)
        X_seq, y63, _ = sys_obj._sequences(df, "target_63d", scaler=sys_obj.scaler)
        n = len(X_seq)
        df_a = df.iloc[-n:]
        vix_a = vix_t.values[-n:]
        spy_a = spy_t.values[-n:]

        regime = sys_obj.gate.compute_regime_vector(df_a, vix_a, spy_a)
        all_preds = {nm: b.predict(X_seq) for nm, b in sys_obj.barometers.items()}
        meta_features = sys_obj.meta._assemble_meta_features(all_preds, regime)

        # ── Fit T+63 regression meta-model ───────────────────────────────────
        reg_params = sys_obj.meta.base_params
        reg = lgb.LGBMRegressor(**reg_params)
        reg.fit(meta_features, y63)

        # ── Fit T+63 classification meta-model ───────────────────────────────
        # FIX: use current close as direction benchmark (matches Fix 2 in fit())
        clf_params = {**reg_params, "objective": "binary", "metric": "auc"}
        clf = lgb.LGBMClassifier(**clf_params)
        current_close = df["close"].values[-n:]
        y_dir = (y63 > current_close).astype(int)
        clf.fit(meta_features, y_dir)

        # ── Persist: update the .pkl dicts (primary load path) ───────────────
        # load() checks meta_reg.pkl first; if it exists, individual .lgb files
        # are ignored entirely. We MUST update the .pkl so t63 survives any
        # subsequent save() call that rewrites these files from the in-memory dict.
        reg_pkl_path = f"{path}/meta_reg.pkl"
        clf_pkl_path = f"{path}/meta_clf.pkl"

        reg_models = joblib.load(reg_pkl_path) if os.path.exists(reg_pkl_path) else {}
        clf_models = joblib.load(clf_pkl_path) if os.path.exists(clf_pkl_path) else {}

        reg_models["t63"] = reg
        clf_models["t63"] = clf

        joblib.dump(reg_models, reg_pkl_path)
        joblib.dump(clf_models, clf_pkl_path)
        print(f"  [{t}] meta_reg.pkl and meta_clf.pkl updated with t63 key ✓")

        # ── Also save .lgb as a lightweight backup (optional, for inspection) ─
        reg.booster_.save_model(f"{path}/meta_reg_t63.lgb")
        clf.booster_.save_model(f"{path}/meta_clf_t63.lgb")
        print(f"  [{t}] .lgb backup files written ✓")


if __name__ == "__main__":
    train_missing_t63_meta()
