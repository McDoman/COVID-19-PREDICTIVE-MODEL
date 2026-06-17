# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:46:29 2025

@author: LAdedo
"""

#%% Cleaning template
import os
import re
import numpy as np
import pandas as pd

# Centralized Relative Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/raw/COVID19.xlsx"))
OUT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/processed/clean.csv"))

# Make sure directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Load data with fallback engine
try:
    df = pd.read_excel(DATA_PATH, engine="calamine")
except Exception:
    df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Standardize column names
df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

# Detect yes/no columns and clean them (comma bug fixed: "unknown", "pending" separate strings)
yn_like = []
for c in df.columns:
    if df[c].dtype == "O" and df[c].str.lower().isin(["yes","no","y","unknown","pending","indeterminate","n","true","false","positive","negative"]).mean() > 0.5:
        yn_like.append(c)

for c in yn_like:
    df[c] = df[c].astype(str).str.lower().map({
        "yes": 1, "y": 1, "true": 1, "positive": 1,
        "pending": -1, "indeterminate": -1, "no": 0,
        "n": 0, "false": 0, "negative": 0, "unknown": -1
    })

# Example: clip implausible ages
if "age" in df.columns:
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan

df = df.drop_duplicates()

df.to_csv(OUT_PATH, index=False)
print("Saved to", OUT_PATH, df.shape)
