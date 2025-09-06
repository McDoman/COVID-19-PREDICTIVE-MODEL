# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 02:57:48 2025

@author: LAdedo
"""

#%% Setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv")  # or read_excel("data/raw/your.xlsx")
TARGET = "result"  

label_map = {
    "NEGATIVE": 0, "0": 0, "FALSE": 0,
    "POSITIVE": 1, "1": 1, "TRUE": 1,
    "PENDING": -1, "INDETERMINATE": -1, "UNKNOWN": -1
}
raw_labels = df[TARGET].astype(str).str.strip().str.upper()
y = raw_labels.map(label_map)

y = y.fillna(-1).astype(int)
print("Label distribution:\n", y.value_counts())
unmapped = raw_labels[y == -1].unique()
if len(unmapped) > 0:
    print("⚠️ Unmapped/missing labels treated as -1:", unmapped)

X = df.drop(columns=[TARGET])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

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

# Stratified splits for multiclass/multibinary classification
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

print("Splits:", X_train.shape, X_val.shape, X_test.shape)