# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 03:39:44 2025

@author: LAdedo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import average_precision_score

# Import shared modular components
from pipeline_helpers import (
    load_and_preprocess_data,
    get_preprocessor,
    get_base_learners
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

# --- Stacking ensemble ---
stack = StackingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    passthrough=False,            
    cv=5, n_jobs=-1
)
stack.fit(X_train, y_train)

# --- PR-AUC on validation set ---
proba_stack_val = stack.predict_proba(X_val)[:, 1]
ap_stack = average_precision_score(y_val, proba_stack_val)
print("Stacking AP (val):", ap_stack)
