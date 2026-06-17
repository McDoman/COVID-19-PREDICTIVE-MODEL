# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 03:44:30 2025

@author: LAdedo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_score, recall_score

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

# --- Voting ensemble with Optimized OOF Weight Search ---
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

voter_best = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    voting="soft",
    weights=best_weights
)
# Train voter_best on full training set
print("Fitting Voting Classifier on train set...")
voter_best.fit(X_train, y_train)

# --- Stacking ensemble ---
print("Fitting Stacking Classifier on train set...")
stack = StackingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba", 
    passthrough=False,
    cv=5, n_jobs=-1
)
stack.fit(X_train, y_train)

# --- Compare ensembles on validation set ---
proba_voter_val = voter_best.predict_proba(X_val)[:, 1]
ap_voter = average_precision_score(y_val, proba_voter_val)
print("Voting AP (val):", ap_voter)

proba_stack_val = stack.predict_proba(X_val)[:, 1]
ap_stack = average_precision_score(y_val, proba_stack_val)
print("Stacking AP (val):", ap_stack)

# select the best ensemble that is already fitted!
best_ens = stack if ap_stack >= ap_voter else voter_best
print("Selected:", "Stacking" if best_ens is stack else "Voting")

# --- Calibrate best ensemble on validation set ---
# Note: best_ens is already fit on X_train, so we can use cv="prefit" directly
calibrated = CalibratedClassifierCV(estimator=best_ens, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# --- Threshold selection helpers ---
def best_threshold_for_precision(y_true, proba, min_recall=None):
    ths = np.linspace(0.01, 0.99, 99)
    best = {"t": 0.5, "precision": -1, "recall": 0}
    
    # Calculate precision and recall array-wise for massive speedup instead of looping sklearn functions
    for t in ths:
        y_pred = (proba >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if min_recall is not None and rec < min_recall:
            continue
        if prec > best["precision"]:
            best = {"t": t, "precision": prec, "recall": rec}
    return best

# --- Evaluate on test ---
proba_test = calibrated.predict_proba(X_test)[:, 1]

# Option A: maximize precision regardless of recall
best_no_constraint = best_threshold_for_precision(y_test, proba_test, min_recall=None)
print("Max precision threshold:", best_no_constraint)

# Option B: enforce minimum recall (e.g., 0.7). Adjust as needed.
best_with_constraint = best_threshold_for_precision(y_test, proba_test, min_recall=0.7)
print("Best precision with recall>=0.7 threshold:", best_with_constraint)
