"""
apply_fixes.py
==============
Run this ONCE from your project root to patch barometer_core.py
with all four fixes. Safe to run multiple times (idempotent).

Usage:
    python apply_fixes.py
"""

import shutil
import sys
from pathlib import Path

TARGET = Path("barometer_core.py")
if not TARGET.exists():
    sys.exit("ERROR: barometer_core.py not found. Run from your project root.")

src = TARGET.read_text(encoding="utf-8")
original = src
applied = []
warnings = []

# ── Fix 1: create_sequences scaler parameter (leakage fix) ──────────────────
F1_OLD = (
    "def create_sequences(self, df: pd.DataFrame, target_col: str,\n"
    "                         feature_cols: list) -> tuple:\n"
    '        """\n'
    "        Sliding window \u2192 3D tensor (n_samples, window, n_features).\n"
    "        Used by LSTM, TCN, and TFT barometers.\n"
    "        Scaler is fitted on the training fold only to prevent data leakage.\n"
    '        """\n'
    "        scaler   = RobustScaler()\n"
    "        X_sc     = scaler.fit_transform(df[feature_cols].values)"
)
F1_NEW = (
    "def create_sequences(self, df: pd.DataFrame, target_col: str,\n"
    "                         feature_cols: list,\n"
    "                         scaler: RobustScaler = None) -> tuple:\n"
    '        """\n'
    "        Sliding window -> 3D tensor (n_samples, window, n_features).\n"
    "        Used by LSTM, TCN, and TFT barometers.\n"
    "\n"
    "        FIX (scaler leakage): pass a pre-fitted scaler to call transform()\n"
    "        only on test folds. Pass scaler=None on train fold to fit a new one.\n"
    '        """\n'
    "        if scaler is None:\n"
    "            scaler = RobustScaler()\n"
    "            X_sc = scaler.fit_transform(df[feature_cols].values)\n"
    "        else:\n"
    "            X_sc = scaler.transform(df[feature_cols].values)"
)
if F1_OLD in src:
    src = src.replace(F1_OLD, F1_NEW)
    applied.append("Fix 1: create_sequences() scaler= parameter added")
elif "scaler: RobustScaler = None" in src:
    applied.append("Fix 1: already applied (skipped)")
else:
    warnings.append("Fix 1: pattern not matched. Apply manually to create_sequences().")

# ── Fix 2: direction label uses current_close not median ────────────────────
F2_OLD = (
    "            # Direction label: 1 if target > median of target (relative move)\n"
    "            y_dir = (y > np.median(y)).astype(int)\n"
    "            clf.fit(meta, y_dir)\n"
    "            self.clf_models[horizon] = clf"
)
F2_NEW = (
    "            # FIX: direction = 1 if target > today's close (not median)\n"
    '            current_close = targets.get("current_close", None)\n'
    "            if current_close is not None:\n"
    "                y_dir = (y > current_close).astype(int)\n"
    "            else:\n"
    "                y_dir = (y > np.median(y)).astype(int)\n"
    "            clf.fit(meta, y_dir)\n"
    "            self.clf_models[horizon] = clf"
)
if F2_OLD in src:
    src = src.replace(F2_OLD, F2_NEW)
    applied.append("Fix 2: direction label uses current_close")
elif "current_close = targets.get" in src:
    applied.append("Fix 2: already applied (skipped)")
else:
    warnings.append(
        "Fix 2: pattern not matched. Apply manually to LightGBMMetaLearner.fit()."
    )

# ── Fix 3: pass current_close into meta.fit from BarometerSystem.fit ────────
F3_OLD = (
    "        all_preds = {nm: b.predict(X_seq) for nm, b in self.barometers.items()}\n"
    '        self.meta.fit(all_preds, regime, {"t1": y1, "t5": y5, "t21": y21})'
)
F3_NEW = (
    "        all_preds = {nm: b.predict(X_seq) for nm, b in self.barometers.items()}\n"
    "        # FIX: supply current_close so Fix 2 direction labels are correct\n"
    '        current_close_arr = df["close"].values[-n:]\n'
    "        self.meta.fit(all_preds, regime, {\n"
    '            "t1": y1, "t5": y5, "t21": y21,\n'
    '            "current_close": current_close_arr\n'
    "        })"
)
if F3_OLD in src:
    src = src.replace(F3_OLD, F3_NEW)
    applied.append("Fix 3: current_close_arr passed to meta.fit()")
elif "current_close_arr" in src:
    applied.append("Fix 3: already applied (skipped)")
else:
    warnings.append(
        "Fix 3: pattern not matched. Apply manually to BarometerSystem.fit()."
    )

# ── Fix 4: WalkForwardEvaluator uses correct target per horizon ──────────────
F4_OLD = (
    '                n    = len(preds[h]["price"])\n'
    '                ytrue = df["target_1d"].values[te][-n:]\n'
    '                ypred = preds[h]["price"]\n'
    "                rmse  = float(np.sqrt(mean_squared_error(ytrue, ypred)))\n"
    "                mae   = float(mean_absolute_error(ytrue, ypred))\n"
    "                mape  = float(np.mean(np.abs((ytrue - ypred) / (np.abs(ytrue) + 1e-8))) * 100)\n"
    '                da    = float(((np.diff(ytrue) > 0) == preds[h]["direction"][:-1]).mean())'
)
F4_NEW = (
    "                # FIX: use the correct target column for each horizon\n"
    '                _HT = {"t1":"target_1d","t5":"target_5d","t21":"target_21d","t63":"target_63d"}\n'
    '                target_col = _HT.get(h, "target_1d")\n'
    '                n     = len(preds[h]["price"])\n'
    "                ytrue = df[target_col].values[te][-n:]\n"
    '                ypred = preds[h]["price"]\n'
    "                rmse  = float(np.sqrt(mean_squared_error(ytrue, ypred)))\n"
    "                mae   = float(mean_absolute_error(ytrue, ypred))\n"
    "                mape  = float(np.mean(np.abs((ytrue - ypred) / (np.abs(ytrue) + 1e-8))) * 100)\n"
    "                actual_dir = (np.diff(ytrue) > 0)\n"
    '                pred_dir   = preds[h]["direction"][:-1].astype(bool)\n'
    '                da = float((actual_dir == pred_dir).mean()) if len(actual_dir) > 0 else float("nan")'
)
if F4_OLD in src:
    src = src.replace(F4_OLD, F4_NEW)
    applied.append("Fix 4: WalkForwardEvaluator uses correct target per horizon")
elif '_HT = {"t1":"target_1d"' in src:
    applied.append("Fix 4: already applied (skipped)")
else:
    warnings.append(
        "Fix 4: pattern not matched. Apply manually to WalkForwardEvaluator.evaluate()."
    )

# ── Write ────────────────────────────────────────────────────────────────────
if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print("  [!]", w)

if src == original:
    print("\nNo changes made (all already applied or patterns unmatched).")
    sys.exit(1 if warnings else 0)

shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
print("Backup: barometer_core.py.bak")
TARGET.write_text(src, encoding="utf-8")
print("\nApplied:")
for a in applied:
    print("  [+]", a)
print("\nDone. Now run:  pytest tests/ -v")
