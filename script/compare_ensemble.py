# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 03:44:29 2025

@author: LAdedo
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

df = pd.read_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv")
TARGET = "result"

label_map = {
    "NEGATIVE": 0, "0": 0, "FALSE": 0,
    "POSITIVE": 1, "1": 1, "TRUE": 1,
    "PENDING": -1, "INDETERMINATE": -1, "UNKNOWN": -1
}
raw_labels = df[TARGET].astype(str).str.strip().str.upper()
y = raw_labels.map(label_map)
y = y.fillna(-1).astype(int)
unmapped = raw_labels[y == -1].unique()
if len(unmapped) > 0:
    print("⚠️ Unmapped/missing labels treated as -1:", unmapped)

mask = y != -1
X = df.loc[mask].drop(columns=[TARGET])
y = y[mask]
print(f"Final data shape: {X.shape}, labels: {y.shape}")
print("Label distribution:\n", y.value_counts())

# --- Split data ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# --- Preprocessor ---
num_cols = X_train.select_dtypes(include=[float, int]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# --- Base learners ---
lr = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
rf = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2,
        n_jobs=-1, class_weight="balanced_subsample", random_state=42
    ))
])
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / max(pos, 1))
xgb = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="aucpr",
        n_jobs=-1, random_state=42,
        scale_pos_weight=scale_pos_weight, tree_method="hist"
        # Remove use_label_encoder if you get a warning
    ))
])

# --- Voting ensemble with PR-AUC grid search ---
voter = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
    voting="soft",
    weights=[1, 1, 1]
)
weight_grid = [
    [1,1,1], [1,1,2], [1,2,1], [2,1,1],
    [1,2,2], [2,1,2], [2,2,1], [1,1,3], [1,3,1], [3,1,1],
    [1,2,3], [2,3,1], [3,1,2], [2,2,3], [3,2,2]
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=voter,
    param_grid={"weights": weight_grid},
    scoring="average_precision",
    cv=cv, n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
voter_best = grid.best_estimator_
print("Best voting weights:", grid.best_params_, "CV AP:", grid.best_score_)

# --- Stacking ensemble ---
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

best_ens = stack if ap_stack >= ap_voter else voter_best
print("Selected:", "Stacking" if best_ens is stack else "Voting")