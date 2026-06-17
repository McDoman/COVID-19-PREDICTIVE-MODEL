# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 03:59:25 2025

@author: LAdedo
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, classification_report, confusion_matrix,
                             precision_score, recall_score)

# Import shared modular components
from pipeline_helpers import (
    load_and_preprocess_data,
    get_preprocessor,
    get_base_learners,
    get_oof_predictions,
    search_best_weights,
    get_data_paths
)

# Load data and relative paths
DATA_PATH, MODEL_DIR, MODEL_PATH = get_data_paths()
X, y = load_and_preprocess_data(DATA_PATH)
print(f"Final data shape: {X.shape}, labels: {y.shape}")
print("Label distribution:\n", y.value_counts())

# ==== 2. Split Data ====
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# ==== 3. Preprocessing & Base Learners ====
pre = get_preprocessor(X_train)

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / max(pos, 1))

lr, rf, xgb = get_base_learners(pre, scale_pos_weight=scale_pos_weight)

# ==== 4. Voting Ensemble (Optimized OOF Weight Search) ====
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
print("Fitting Voting Classifier on train set...")
voter_best.fit(X_train, y_train)

# ==== 5. Stacking Ensemble ====
print("Fitting Stacking Classifier on train set...")
stack = StackingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba", 
    passthrough=False,
    cv=5, n_jobs=-1
)
stack.fit(X_train, y_train)

# ==== 6. Compare Ensembles on Validation Set ====
proba_voter_val = voter_best.predict_proba(X_val)[:, 1]
ap_voter = average_precision_score(y_val, proba_voter_val)
print("Voting AP (val):", ap_voter)

proba_stack_val = stack.predict_proba(X_val)[:, 1]
ap_stack = average_precision_score(y_val, proba_stack_val)
print("Stacking AP (val):", ap_stack)

best_ens = stack if ap_stack >= ap_voter else voter_best
print("Selected:", "Stacking" if best_ens is stack else "Voting")

# ==== 7. Calibrate on Validation Set ====
# best_ens is already fit, so we use cv="prefit"
calibrated = CalibratedClassifierCV(estimator=best_ens, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# ==== 8. Threshold Selection Helpers ====
def best_threshold_for_precision(y_true, proba, min_recall=None):
    ths = np.linspace(0.01, 0.99, 99)
    best = {"t": 0.5, "precision": -1, "recall": 0}
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

# ==== 9. Evaluate on Test Set ====
proba_test = calibrated.predict_proba(X_test)[:, 1]

# Option A: maximize precision regardless of recall
best_no_constraint = best_threshold_for_precision(y_test, proba_test, min_recall=None)
print("Max precision threshold:", best_no_constraint)

# Option B: enforce minimum recall (e.g., 0.7)
best_with_constraint = best_threshold_for_precision(y_test, proba_test, min_recall=0.7)
print("Best precision with recall>=0.7 threshold:", best_with_constraint)

# ==== 10. Final Metrics & Confusion Matrix ====
def eval_at_threshold(y_true, proba, t):
    y_pred = (proba >= t).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "threshold": t
    }

chosen = best_with_constraint if best_with_constraint["precision"] >= 0 else best_no_constraint
metrics = eval_at_threshold(y_test, proba_test, chosen["t"])
print("Selected threshold and metrics:")
print(metrics)
print("\nClassification report:")
print(classification_report(y_test, (proba_test >= chosen["t"]).astype(int), digits=3))

cm = confusion_matrix(y_test, (proba_test >= chosen["t"]).astype(int))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix (Test set)")
plt.tight_layout()
plt.show()

# Save best model to disk using relative paths
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(calibrated, MODEL_PATH)
print("Model saved to", MODEL_PATH)
