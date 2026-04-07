"""
Diagnostic script: Print the number of features the saved LightGBM
booster expects vs what the current pipeline assembles.
"""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

path = "barometer_saved/MSFT"

# Load saved feature columns (input to base models)
features = joblib.load(f"{path}/features.pkl")
print(f"Saved feature_cols count: {len(features)}")
print("First 5:", features[:5])
print("Last 5:", features[-5:])
print()

# Load LightGBM booster and inspect
bst_reg = lgb.Booster(model_file=f"{path}/meta_reg_t1.lgb")
bst_clf = lgb.Booster(model_file=f"{path}/meta_clf_t1.lgb")
print(f"meta_reg_t1.lgb num_feature: {bst_reg.num_feature()}")
print(f"meta_clf_t1.lgb num_feature: {bst_clf.num_feature()}")
print()

# Show feature names saved inside the booster (if any)
fn = bst_reg.feature_name()
print(f"Booster feature names ({len(fn)}):")
for i, n in enumerate(fn):
    print(f"  [{i}] {n}")
