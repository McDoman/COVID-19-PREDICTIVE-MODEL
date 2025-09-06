# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:36:50 2025

@author: LAdedo
"""

#%% Basic info
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_excel("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/raw/COVID19.xlsx", engine="openpyxl")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
TARGET = "result"

#%% Overview
print(df.info())
print(df.describe(include="all").transpose().head(20))
print(df[TARGET].value_counts(dropna=False))

#%% Missingness
missing = df.isna().mean().sort_values(ascending=False)
print("Missingness (top 20):\n", missing.head(20))

#%% Numeric distributions
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET]
for c in num_cols[:12]:
    sns.histplot(df[c], kde=True)
    plt.title(f"Distribution: {c}")
    plt.show()

#%% Categorical vs target
cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]
for c in cat_cols[:12]:
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x=c, hue=TARGET)
    plt.title(f"{c} by {TARGET}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

#%% Correlation heatmap (numeric)
if num_cols:
    corr = df[num_cols + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (numeric)")
    plt.show()