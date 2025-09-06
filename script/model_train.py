# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:22:27 2025

@author: LAdedo
"""

#%% Train and tune models
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv")
TARGET = "result"

label_map = {
    "NEGATIVE": 0, "0": 0, "FALSE": 0,
    "POSITIVE": 1, "1": 1, "TRUE": 1,
    "PENDING": -1, "INDETERMINATE": -1, "UNKNOWN": -1
}
raw_labels = df[TARGET].astype(str).str.strip().str.upper()
y = raw_labels.map(label_map)

y = y.fillna(-1)
unmapped = raw_labels[y.isna()].unique()
if len(unmapped) > 0:
    print("⚠️ Unmapped labels found:", unmapped)

# Keep only 0/1, drop -1
mask = y.isin([0, 1])
dropped = len(y) - mask.sum()
if dropped > 0:
    print(f"Dropping {dropped} rows with non-binary labels")
df, y = df.loc[mask].copy(), y.loc[mask]
X = df.drop(columns=[TARGET])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# Class imbalance ratio (for XGB)
pos = y.sum(); neg = len(y) - pos
scale_pos_weight = neg / max(pos, 1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_grids = [
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"),
     {"clf__C": np.logspace(-3, 2, 10), "clf__penalty": ["l2"]}),
    ("rf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=42),
     {"clf__n_estimators": [200, 400, 800],
      "clf__max_depth": [None, 5, 10, 20],
      "clf__min_samples_split": [2, 5, 10]}),
    ("xgb", XGBClassifier(
        n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", eval_metric="auc", random_state=42, n_jobs=-1,
        scale_pos_weight=scale_pos_weight
     ),
     {"clf__max_depth": [3,4,5,6],
      "clf__min_child_weight": [1,2,5],
      "clf__gamma": [0, 0.5, 1.0]})
]

best = None
results = []
for name, base_clf, grid in models_and_grids:
    pipe = Pipeline([("pre", pre), ("clf", base_clf)])
    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=grid, n_iter=min(20, np.prod([len(v) for v in grid.values()])),
        scoring="roc_auc", cv=cv, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X, y)
    mean_auc = search.best_score_
    results.append((name, mean_auc, search.best_params_))
    print(f"{name}: CV ROC-AUC={mean_auc:.3f} | best params={search.best_params_}")
    if best is None or mean_auc > best["auc"]:
        best = {"name": name, "model": search.best_estimator_, "auc": mean_auc, "params": search.best_params_}

print("Best model:", best["name"], best["auc"])
joblib.dump(best, "C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/best_cv.joblib")