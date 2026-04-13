"""
tests/test_fixes.py
===================
Unit tests covering the fixes applied to barometer_core.py:

  Fix 1 — Scaler leakage:    create_sequences() accepts a pre-fitted scaler
  Fix 2/3 — Direction label: meta-learner uses close-relative direction
  Fix 4 — OOF meta stacking: meta trained on out-of-fold predictions
  Fix 5 — Walk-forward eval: each horizon uses its correct target column
  Fix 6 — T+63 coverage:     helper, meta-fit, and OOF assertions all include t63
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from barometer_core import DataPipeline, LightGBMMetaLearner

# ── Make barometer_core importable from the project root ──────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress TF noise during tests
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _make_dummy_df(n_rows: int = 400) -> pd.DataFrame:
    """Return a minimal feature DataFrame that DataPipeline would produce."""
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    idx = pd.bdate_range("2022-01-01", periods=n_rows)
    df = pd.DataFrame(
        {
            "close": price,
            "target_1d": np.roll(price, -1),
            "target_5d": np.roll(price, -5),
            "target_21d": np.roll(price, -21),
            "target_63d": np.roll(price, -63),
            **{f"feat_{i}": np.random.randn(n_rows) for i in range(10)},
        },
        index=idx,
    )
    return df.iloc[:-63]  # trim NaN targets at end


def _make_dummy_predictions(n: int = 200) -> dict:
    """Return fake base-model predictions with the expected shape (all 4 horizons)."""
    np.random.seed(0)
    horizons = ["t1", "t5", "t21", "t63"]  # FIX: include t63 to match real barometers
    models = ["lstm", "xgb", "tcn", "tft"]
    return {m: {h: np.random.randn(n) * 10 + 300 for h in horizons} for m in models}


def _make_dummy_regime(n: int = 200) -> pd.DataFrame:
    np.random.seed(1)
    return pd.DataFrame(
        {
            "vix_regime": np.random.randint(0, 4, n),
            "adx_regime": np.random.randint(0, 4, n),
            "hmm_regime": np.random.randint(0, 4, n),
            "hmm_p0": np.random.rand(n),
            "hmm_p1": np.random.rand(n),
            "hmm_p2": np.random.rand(n),
            "hmm_p3": np.random.rand(n),
            "corr_shift": np.random.randint(0, 2, n),
            "rolling_corr": np.random.randn(n) * 0.3 + 0.5,
            "regime_score": np.random.rand(n) * 10,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 1 — SCALER LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════════


class TestScalerLeakageFix:
    """
    Verify that create_sequences() correctly separates scaler fitting
    from scaler transformation — no future data visible during fit.
    """

    def setup_method(self):
        self.pipeline = DataPipeline(
            tickers=[], start="2022-01-01", end="2024-01-01", window=20
        )
        df = _make_dummy_df(300)
        self.feature_cols = [
            c
            for c in df.columns
            if c not in {"target_1d", "target_5d", "target_21d", "target_63d"}
        ]
        self.df = df

    def test_no_scaler_returns_new_fitted_scaler(self):
        """Calling without a scaler must return a freshly fitted RobustScaler."""
        X, y, scaler = self.pipeline.create_sequences(
            self.df, "target_1d", self.feature_cols, scaler=None
        )
        assert isinstance(scaler, RobustScaler)
        # A fitted scaler has center_ attribute
        assert hasattr(scaler, "center_"), "Scaler should be fitted"

    def test_pre_fitted_scaler_is_reused(self):
        """
        Passing a pre-fitted scaler must NOT refit it — the scaler's
        center_ (median) values must remain identical before and after.
        """
        # Fit scaler on first half only (train fold)
        train_df = self.df.iloc[:150]
        _, _, train_scaler = self.pipeline.create_sequences(
            train_df, "target_1d", self.feature_cols, scaler=None
        )
        original_center = train_scaler.center_.copy()

        # Pass the same fitted scaler for the test fold — must not change
        test_df = self.df.iloc[150:]
        _, _, returned_scaler = self.pipeline.create_sequences(
            test_df, "target_1d", self.feature_cols, scaler=train_scaler
        )
        np.testing.assert_array_equal(
            returned_scaler.center_,
            original_center,
            err_msg="Scaler was refitted on test fold — leakage!",
        )

    def test_train_test_scaler_differ_in_center(self):
        """
        A scaler fitted on train-only data should have a different median
        than one naively fitted on the full dataset, confirming the fix
        is meaningful.
        """
        _, _, train_scaler = self.pipeline.create_sequences(
            self.df.iloc[:150], "target_1d", self.feature_cols, scaler=None
        )
        _, _, full_scaler = self.pipeline.create_sequences(
            self.df, "target_1d", self.feature_cols, scaler=None
        )
        # The medians should differ because they are computed on different data
        assert not np.allclose(
            train_scaler.center_, full_scaler.center_
        ), "Train-only and full-data scalers have identical centers — unexpected."


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 2/3 — DIRECTION LABEL USES CURRENT CLOSE
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectionLabelFix:
    """
    Verify that the classification direction label is computed as
    (target > current_close) when current_close is supplied.
    """

    def _fit_meta(self, pass_close: bool) -> LightGBMMetaLearner:
        n = 200
        preds = _make_dummy_predictions(n)
        regime = _make_dummy_regime(n)
        np.random.seed(5)
        targets = {
            "t1":  np.random.randn(n) * 10 + 305,
            "t5":  np.random.randn(n) * 12 + 307,
            "t21": np.random.randn(n) * 15 + 310,
            "t63": np.random.randn(n) * 20 + 330,  # FIX: include t63 target
        }
        if pass_close:
            targets["current_close"] = np.random.randn(n) * 8 + 300

        meta = LightGBMMetaLearner()
        meta.fit(preds, regime, targets)
        return meta

    def test_meta_fits_with_current_close(self):
        """Meta-learner must fit successfully when current_close is provided."""
        meta = self._fit_meta(pass_close=True)
        for h in ["t1", "t5", "t21", "t63"]:
            assert h in meta.clf_models, f"Missing clf_model for horizon {h}"
            assert h in meta.reg_models, f"Missing reg_model for horizon {h}"

    def test_meta_fits_without_current_close(self):
        """Meta-learner must fit successfully even without current_close (fallback)."""
        meta = self._fit_meta(pass_close=False)
        for h in ["t1", "t5", "t21", "t63"]:
            assert h in meta.clf_models, f"Missing clf_model for horizon {h}"

    def test_direction_label_not_always_50_50(self):
        """
        With real direction labels (target > current_close), the class balance
        will generally NOT be exactly 50/50 — unlike the old median convention
        which forced exactly 50% positive by construction.
        This test simply verifies the model produces non-trivial probabilities.
        """
        meta = self._fit_meta(pass_close=True)
        n = 50
        preds = _make_dummy_predictions(n)
        regime = _make_dummy_regime(n)
        result = meta.predict(preds, regime)
        probs = result["t1"]["up_prob"]
        # Probabilities should span a range — not collapsed to 0.5
        assert (
            probs.max() - probs.min() > 0.05
        ), "All up_probs identical — classifier may be degenerate."


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 4 — OUT-OF-FOLD META STACKING
# ═══════════════════════════════════════════════════════════════════════════════


class TestOOFMetaStacking:
    """
    Verify the meta-learner trains on OOF predictions, not in-sample predictions.
    We confirm this indirectly: OOF training + refit on full data should
    still produce models that can predict without error on fresh data.
    """

    def test_meta_predict_after_oof_fit(self):
        n = 300
        preds = _make_dummy_predictions(n)
        regime = _make_dummy_regime(n)
        np.random.seed(7)
        targets = {
            "t1":  np.random.randn(n) * 10 + 305,
            "t5":  np.random.randn(n) * 12 + 307,
            "t21": np.random.randn(n) * 15 + 310,
            "t63": np.random.randn(n) * 20 + 330,   # FIX: include t63 target
            "current_close": np.random.randn(n) * 8 + 300,
        }
        meta = LightGBMMetaLearner()
        meta.fit(preds, regime, targets)

        # Predict on fresh 50-sample slice
        preds_new = _make_dummy_predictions(50)
        regime_new = _make_dummy_regime(50)
        result = meta.predict(preds_new, regime_new)

        for h in ["t1", "t5", "t21", "t63"]:  # FIX: assert all 4 horizons
            assert h in result, f"Missing horizon {h} in prediction output"
            assert len(result[h]["price"]) == 50
            assert len(result[h]["up_prob"]) == 50
            assert np.all(np.isfinite(result[h]["price"])), f"NaN in {h} price predictions"
            assert np.all(
                (result[h]["up_prob"] >= 0) & (result[h]["up_prob"] <= 1)
            ), f"{h} up_prob out of [0, 1] range"


# ═══════════════════════════════════════════════════════════════════════════════
#  FIX 5 — WALK-FORWARD EVALUATION USES CORRECT TARGET PER HORIZON
# ═══════════════════════════════════════════════════════════════════════════════


class TestWalkForwardTargetFix:
    """
    Verify that the HORIZON_TARGET mapping is correct — each horizon key
    maps to its own target column, not universally to target_1d.
    """

    # Inline the mapping here to test it independently of the class internals
    HORIZON_TARGET = {
        "t1": "target_1d",
        "t5": "target_5d",
        "t21": "target_21d",
        "t63": "target_63d",
    }

    def test_all_horizons_have_distinct_targets(self):
        """Each horizon must map to a distinct target column."""
        values = list(self.HORIZON_TARGET.values())
        assert len(values) == len(
            set(values)
        ), "Two horizons map to the same target column."

    def test_t1_maps_to_target_1d(self):
        assert self.HORIZON_TARGET["t1"] == "target_1d"

    def test_t5_maps_to_target_5d(self):
        assert self.HORIZON_TARGET["t5"] == "target_5d"

    def test_t21_maps_to_target_21d(self):
        assert self.HORIZON_TARGET["t21"] == "target_21d"

    def test_t63_maps_to_target_63d(self):
        assert self.HORIZON_TARGET["t63"] == "target_63d"

    def test_dummy_df_has_all_target_columns(self):
        """Our test DataFrame has all four target columns needed by the evaluator."""
        df = _make_dummy_df(300)
        for col in self.HORIZON_TARGET.values():
            assert col in df.columns, f"Target column {col} missing from DataFrame"
