# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 16:58:11 2025

@author: LAdedo
"""

#%% Imports and config
import pandas as pd, numpy as np, os, re
from pathlib import Path

DATA_PATH = "C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/raw/COVID19.xlsx"  # update
SHEET_NAME = "COVID19"
TARGET = "result"
POS_LABEL = "POSITIVE"
ID_COLS = []  

# Remove columns to avoid model cheat
LEAKAGE_COLS = ["Symptoms_Abnormal lung X-Ray findings", 
                "Symptoms_Pneumonia (clinical or radiologic)",
                "Symptoms_Acute respiratory distress syndrome",
                "Symptoms_Fluid in cavity through X-Ray"]

#%% Load
df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
df = df_raw.copy()

# Normalize
normalize = lambda c: re.sub(r"\s+", "_", str(c).strip().lower())

# Standardize column names
df.columns = [normalize(c) for c in df.columns]
assert TARGET in df.columns, f"'{TARGET}' not found. Columns: {df.columns.tolist()}"

# Normalize ID/LEAKAGE 
ID_COLS = [normalize(c) for c in ID_COLS]
LEAKAGE_COLS = [normalize(c) for c in LEAKAGE_COLS]
drop_cols = [c for c in (ID_COLS + LEAKAGE_COLS) if c in df.columns]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Encode target robustly
s = df.pop(TARGET).astype(str).str.strip().str.upper()
valid = {'NEGATIVE', 'POSITIVE', '0', '1', 'FALSE', 'TRUE'}
mask = s.isin(valid)

bad = s[~mask].value_counts(dropna=False)
if len(bad):
    print("Dropping rows with non-final labels:", bad.to_dict())
df = df.loc[mask].copy()
y = s.loc[mask].map({
    'NEGATIVE': 0, 'POSITIVE': 1,
    '0': 0, '1': 1,
    'FALSE': 0, 'TRUE': 1
}).astype('int8')

X = df
assert y.notna().all()
print("y dtype:", y.dtype, "| y counts:", y.value_counts().to_dict())
print("X shape:", X.shape)
