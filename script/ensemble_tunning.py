# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 03:31:50 2025

@author: LAdedo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import average_precision_score

# Import shared modular components
from pipeline_helpers import (
    load_and_preprocess_data,
    get_preprocessor,
    get_base_learners,
    get_oof_predictions,
    search_best_weights
)

# Load data using relative path helper
X, y = load_and_preprocess_data()
print(f"Final data shape: {X.shape}, labels: {y.shape}")
print("Label distribution:\n", y.value_counts())

# --- Split data ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# --- Preprocessor & Base Learners ---
pre = get_preprocessor(X_train)

# Imbalance ratio for XGB
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / max(pos, 1))

lr, rf, xgb = get_base_learners(pre, scale_pos_weight=scale_pos_weight)

# --- Fit base learners (optional, for diagnostics) ---
for name, model in [("LR", lr), ("RF", rf), ("XGB", xgb)]:
    model.fit(X_train, y_train)
    print(f"{name} fitted.")

# --- Soft Voting Ensemble with Optimized OOF Weight Search ---
weight_grid = [
    [1,1,1], [1,1,2], [1,2,1], [2,1,1],
    [1,2,2], [2,1,2], [2,2,1], [1,1,3], [1,3,1], [3,1,1],
    [1,2,3], [2,3,1], [3,1,2], [2,2,3], [3,2,2]
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nPrecomputing out-of-fold predictions for base estimators...")
estimators = [("lr", lr), ("rf", rf), ("xgb", xgb)]
oof_probs = get_oof_predictions(estimators, X_train, y_train, cv)

print("Optimizing voting weights on cached predictions...")
best_weights, best_cv_ap = search_best_weights(oof_probs, y_train, weight_grid)
print("Best voting weights:", best_weights, "CV AP:", best_cv_ap)

# Fit final voter on full training set using optimized weights
voter_best = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    voting="soft",
    weights=best_weights
)
voter_best.fit(X_train, y_train)

# --- Evaluate on validation set ---
val_pred_proba = voter_best.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, val_pred_proba)
print("Validation PR-AUC for best ensemble:", val_pr_auc)
