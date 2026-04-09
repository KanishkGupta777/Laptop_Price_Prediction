"""
app.py — Flask REST API for Laptop Price Prediction
====================================================
Dataset: 1303 real laptops | Prices in Euros

Endpoints:
  GET  /           → health check + model info
  POST /predict    → predict price from specs
  GET  /encodings  → valid values for each category

Run:
    python project_root/api/app.py

Sample curl:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "Company": "Dell", "TypeName": "Notebook",
               "Inches": 15.6, "Ram_GB": 8, "Storage_GB": 256,
               "SSD": 1, "Weight_KG": 2.0, "OpSys": "Windows",
               "CPU_Brand": "Intel", "GPU_Brand": "Nvidia",
               "Resolution_MP": 2073600, "IPS": 1, "Touchscreen": 0
             }'
"""

import os, pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "laptop_price_model.pkl")

with open(MODEL_PATH, "rb") as f:
    ARTIFACT = pickle.load(f)

MODEL    = ARTIFACT["model"]
ENCODERS = ARTIFACT["label_encoders"]
FEATURES = ARTIFACT["feature_names"]

REQUIRED = ["Company","TypeName","Inches","Ram_GB","Storage_GB","SSD",
            "Weight_KG","OpSys","CPU_Brand","GPU_Brand",
            "Resolution_MP","IPS","Touchscreen"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status"      : "✅ Running",
        "model"       : ARTIFACT["model_name"],
        "r2_score"    : ARTIFACT["r2"],
        "mae_euros"   : ARTIFACT["mae"],
        "dataset"     : "1303 real laptops (Kaggle)",
        "predict_at"  : "POST /predict",
        "encodings_at": "GET  /encodings"
    })


@app.route("/encodings", methods=["GET"])
def encodings():
    return jsonify({
        "valid_values": {col: list(le.classes_) for col, le in ENCODERS.items()},
        "note": "Resolution_MP examples: 1920x1080=2073600, 2560x1440=3686400"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        missing = [f for f in REQUIRED if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        encoded = dict(data)
        for col, le in ENCODERS.items():
            val = encoded.get(col)
            if val not in le.classes_:
                return jsonify({
                    "error"       : f"Invalid value '{val}' for '{col}'",
                    "valid_values": list(le.classes_)
                }), 400
            encoded[col] = int(le.transform([val])[0])

        input_df = pd.DataFrame([encoded])[FEATURES]
        price    = round(float(MODEL.predict(input_df)[0]), 2)

        return jsonify({
            "predicted_price_euros": price,
            "predicted_price_usd"  : round(price * 1.08, 2),
            "model_used"           : ARTIFACT["model_name"],
            "input_received"       : data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"🚀 Laptop Price Prediction API")
    print(f"   Model   : {ARTIFACT['model_name']}")
    print(f"   R²      : {ARTIFACT['r2']}")
    print(f"   MAE     : €{ARTIFACT['mae']}")
    print(f"   URL     : http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
