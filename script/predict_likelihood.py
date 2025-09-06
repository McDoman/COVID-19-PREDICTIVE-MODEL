# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:31:40 2025

@author: LAdedo
"""

import pandas as pd, joblib

art = joblib.load("models/model.joblib")
model = art["calibrated_model"]; thr = art["threshold"]

# Single JSON-like record
sample = {
    # "age": 45, "sex": "male", "fever": 1, "cough": 1, ...
    # Make sure keys match your training columns
}
df_in = pd.DataFrame([sample])
p = model.predict_proba(df_in)[:,1][0]
label = int(p >= thr)
print({"probability": float(p), "label": label})

# Batch from Excel
new_df = pd.read_excel("data/raw/new_cases.xlsx", engine="openpyxl")
probas = model.predict_proba(new_df)[:,1]
labels = (probas >= thr).astype(int)
new_df.assign(covid_probability=probas, covid_pred=labels).to_excel("data/predictions.xlsx", index=False)