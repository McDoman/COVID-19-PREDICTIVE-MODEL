# -*- coding: utf-8 -*-
"""
Shared Pipeline Helpers for COVID-19 Predictive Model
Contains data loading, preprocessing pipeline creation, base learner definitions,
and optimized voting ensemble weight search using precomputed OOF probabilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# Centralized Relative Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/processed/clean.csv"))
MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../models"))
MODEL_PATH = os.path.join(MODEL_DIR, "best_cv.joblib")

def get_data_paths():
    return DATA_PATH, MODEL_DIR, MODEL_PATH

def load_and_preprocess_data(data_path=DATA_PATH):
    """
    Loads clean.csv, maps targets, removes non-binary labels, and returns X and y.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Please run clean_data.py first.")
        
    df = pd.read_csv(data_path)
    TARGET = "result"
    
    label_map = {
        "NEGATIVE": 0, "0": 0, "FALSE": 0,
        "POSITIVE": 1, "1": 1, "TRUE": 1,
        "PENDING": -1, "INDETERMINATE": -1, "UNKNOWN": -1
    }
    
    raw_labels = df[TARGET].astype(str).str.strip().str.upper()
    y = raw_labels.map(label_map)
    y = y.fillna(-1).astype(int)
    
    mask = y != -1
    X = df.loc[mask].drop(columns=[TARGET])
    y = y[mask]
    
    return X, y

def get_preprocessor(X_train):
    """
    Builds the ColumnTransformer for numerical and categorical imputation and scaling.
    """
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
    return pre

def get_base_learners(preprocessor, scale_pos_weight=1.0):
    """
    Returns the three base classifier pipelines: Logistic Regression, Random Forest, and XGBoost.
    """
    lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ))
    ])
    xgb = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="aucpr",
            n_jobs=-1, random_state=42,
            scale_pos_weight=scale_pos_weight, tree_method="hist"
        ))
    ])
    return lr, rf, xgb

def get_oof_predictions(estimators, X, y, cv):
    """
    Generates out-of-fold probability predictions for the positive class
    for each base estimator using cross-validation.
    """
    oof_preds = []
    for name, est in estimators:
        # Generate cross-validated probability predictions
        proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        oof_preds.append(proba)
    return np.column_stack(oof_preds)

def search_best_weights(oof_probs, y, weight_grid):
    """
    Finds the weights that maximize Average Precision (PR-AUC) using OOF probabilities.
    Bypasses re-fitting estimators.
    """
    best_score = -1
    best_weights = None
    
    for w in weight_grid:
        w_arr = np.array(w)
        # Calculate soft-voted probabilities
        voted_probs = np.average(oof_probs, axis=1, weights=w_arr)
        score = average_precision_score(y, voted_probs)
        if score > best_score:
            best_score = score
            best_weights = w
            
    return best_weights, best_score
