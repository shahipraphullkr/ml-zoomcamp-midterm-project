"""Flask web service that serves the diabetes model.

Run locally:  python predict.py
"""
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = "model.bin"
FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask("diabetes-prediction")


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    # Missing measurements are allowed (the model imputes them)
    row = {col: data.get(col, np.nan) for col in FEATURES}
    try:
        X = pd.DataFrame([row], columns=FEATURES).astype(float)
    except (TypeError, ValueError):
        return jsonify({"error": "All values must be numbers"}), 400

    # Same rule as training: 0 means "not measured" for these columns
    zero_cols = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    X[zero_cols] = X[zero_cols].replace(0, np.nan)

    proba = float(model.predict_proba(X)[0, 1])
    return jsonify({
        "diabetes_probability": round(proba, 4),
        "diabetes": proba >= 0.5,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
