# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:46:29 2025

@author: LAdedo
"""

#%% Cleaning template
import pandas as pd, numpy as np, re

DATA_PATH = "C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/raw/COVID19.xlsx"
TARGET = "result"

df = pd.read_excel(DATA_PATH, engine="openpyxl")
df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

yn_like = []
for c in df.columns:
    if df[c].dtype == "O" and df[c].str.lower().isin(["yes","no","y","unknown""pending","indeterminate","n","true","false","positive","negative"]).mean() > 0.5:
        yn_like.append(c)
for c in yn_like:
    df[c] = df[c].astype(str).str.lower().map({"yes":1,"y":1,"true":1,"positive":1,"pending":-1,"indeterminate":-1,"no":0,"n":0,"false":0,"negative":0,"unknown":-1})

# Example: clip implausible ages
if "age" in df.columns:
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan

df = df.drop_duplicates()

df.to_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv", index=False)
print("Saved to C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv", df.shape)