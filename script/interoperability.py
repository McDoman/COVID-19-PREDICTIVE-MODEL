# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:43:00 2025

@author: LAdedo
"""

import shap, joblib, pandas as pd
art = joblib.load("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/model.joblib")
model = art["calibrated_model"].base_estimator if hasattr(art["calibrated_model"], "base_estimator") else art["calibrated_model"]
# If base_estimator is a Pipeline with XGB/RF at step "clf":
try:
    final_clf = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    X_sample = pd.read_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv").drop(columns=["result"]).sample(500, random_state=0)
    X_trans = pre.fit_transform(X_sample)  # fit only for demo; better reuse training fit
    explainer = shap.TreeExplainer(final_clf)
    shap_values = explainer.shap_values(X_trans)
    shap.summary_plot(shap_values, X_trans, show=True)
except Exception as e:
    print("SHAP optional step:", e)