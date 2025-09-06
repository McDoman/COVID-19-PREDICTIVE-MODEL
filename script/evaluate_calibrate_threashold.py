# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:25:45 2025

@author: LAdedo
"""

#%% Evaluate, calibrate, pick threshold
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve,
                             average_precision_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix, classification_report)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Data and splits
df = pd.read_csv("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/data/processed/clean.csv")
TARGET = "result"

label_map = {
    "NEGATIVE": 0, "0": 0, "FALSE": 0,
    "POSITIVE": 1, "1": 1, "TRUE": 1,
    "PENDING": -1, "INDETERMINATE": -1, "UNKNOWN": -1
}
raw_labels = df[TARGET].astype(str).str.strip().str.upper()
y = raw_labels.map(label_map)
# Classify any NaN
y = y.fillna(-1)
mask = y.isin([0, 1])
df, y = df.loc[mask].copy(), y.loc[mask]
X = df.drop(columns=[TARGET])

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

best = joblib.load("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/best_cv.joblib")
base = best["model"]

# Fit on train
base.fit(X_train, y_train)

# Calibrate on val
cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv="prefit")
cal.fit(X_val, y_val)

# Evaluate on test
proba = cal.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)

# Choose threshold
ths = np.linspace(0.05, 0.95, 91)
f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in ths]
best_t = ths[int(np.argmax(f1s))]
print(f"ROC-AUC={roc_auc:.3f} | PR-AUC={pr_auc:.3f} | Best threshold={best_t:.2f}")

y_pred = (proba >= best_t).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Curves
fpr, tpr, _ = roc_curve(y_test, proba)
precision, recall, _ = precision_recall_curve(y_test, proba)

plt.plot(fpr, tpr); plt.plot([0,1], [0,1], "--"); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.show()
plt.plot(recall, precision); plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.show()

# Calibration plot
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],"--"); plt.title("Calibration Curve"); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues"); plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# Save final artifacts
import datetime as dt
artifacts = {"calibrated_model": cal, "threshold": float(best_t), "cv_summary": best}
joblib.dump(artifacts, "C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/model.joblib")
with open("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/metadata.txt", "w") as f:
    f.write(f"Saved: {dt.datetime.now()}\nROC-AUC: {roc_auc:.3f}\nPR-AUC: {pr_auc:.3f}\nThreshold: {best_t:.3f}\nModel: {best['name']}\n")