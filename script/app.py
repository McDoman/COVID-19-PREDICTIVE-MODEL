# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:35:45 2025

@author: LAdedo
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
best = joblib.load("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/models/best_cv.joblib")
model = best["model"]

@app.route("/")
def home():
    return render_template("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/template/index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = request.form.get("age")
        sex = request.form.get("sex")
        fever = request.form.get("symptom_fever")
        cough = request.form.get("symptom_cough")
        input_data = pd.DataFrame([{
            "age": int(age),
            "sex": sex,
            "symptom_fever": int(fever),
            "symptom_cough": int(cough)
        }])

        # Make prediction
        pred_prob = model.predict_proba(input_data)[0,1]
        pred_class = model.predict(input_data)[0]

        return render_template("C:/Users/ladedo/Desktop/ArcGIS.file/ADEDO/EHA2/covid_ml_project/template/index.html", 
                               prediction=pred_class, 
                               probability=round(pred_prob, 3))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
