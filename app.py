"""
app.py
Flask REST API – Fishery Order Anomaly Detection

Endpoints
─────────
GET  /health               – liveness check
GET  /products             – list all products
GET  /outlets              – list all outlets
POST /predict              – check ONE order line for anomaly
POST /predict/batch        – check MANY order lines at once

Oracle APEX (or any client) can call POST /predict with:
{
    "outlet_id":   1,
    "product_id":  2,
    "quantity_in_kg": 150.0,
    "order_day":  "2026-04-18"
}

Response:
{
    "is_anomaly":     true,
    "anomaly_score":  -0.1234,   // more negative = more anomalous
    "message":        "...",
    "day_of_week":    4,
    "is_weekend":     false
}
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── constants ────────────────────────────────────────────────────────────────
MODEL_DIR = "model"
DATA_DIR  = "data"
FEATURES  = ["quantity_in_kg", "day_of_week", "is_weekend", "month",
             "product_id", "outlet_id"]

# ── app init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow Oracle APEX (cross-origin) to call this API


# ── load artefacts once at startup ───────────────────────────────────────────
def _load_artefacts():
    model_path  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model not found.")
    return joblib.load(model_path), joblib.load(scaler_path)


def _ensure_artefacts():
    """Load model; if missing or incompatible (e.g. pickle version mismatch), retrain."""
    try:
        return _load_artefacts()
    except Exception as exc:
        print(f"[app] Could not load model ({exc}). Retraining ...")
        from train import train
        train()
        return _load_artefacts()


model, scaler = _ensure_artefacts()

products_df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
outlets_df  = pd.read_csv(os.path.join(DATA_DIR, "outlets.csv"))

print("[app] Model loaded. API ready.")


# ── helpers ───────────────────────────────────────────────────────────────────
def _extract_features(item: dict) -> tuple[np.ndarray, int, bool]:
    """
    Parse one prediction request dict and return:
        feature_row  – shape (1, 6)
        day_of_week  – int 0-6
        is_weekend   – bool
    Raises ValueError with a descriptive message on bad input.
    """
    required = ["outlet_id", "product_id", "quantity_in_kg", "order_day"]
    missing  = [f for f in required if f not in item]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    try:
        order_date = datetime.strptime(str(item["order_day"]), "%Y-%m-%d")
    except ValueError:
        raise ValueError("order_day must be YYYY-MM-DD")

    qty = float(item["quantity_in_kg"])
    if qty <= 0:
        raise ValueError("quantity_in_kg must be > 0")

    day_of_week = order_date.weekday()
    is_weekend  = day_of_week >= 5
    month       = order_date.month

    row = np.array([[
        qty,
        day_of_week,
        int(is_weekend),
        month,
        int(item["product_id"]),
        int(item["outlet_id"]),
    ]], dtype=float)

    return row, day_of_week, is_weekend


def _run_prediction(row: np.ndarray, day_of_week: int,
                    is_weekend: bool, item: dict) -> dict:
    row_scaled  = scaler.transform(row)
    prediction  = int(model.predict(row_scaled)[0])       # 1 = normal, -1 = anomaly
    score       = float(model.decision_function(row_scaled)[0])
    is_anomaly  = prediction == -1

    qty     = item["quantity_in_kg"]
    prod_id = item["product_id"]
    out_id  = item["outlet_id"]

    if is_anomaly:
        message = (
            f"ANOMALY DETECTED: {qty} kg for product {prod_id} "
            f"from outlet {out_id} is unusually high."
        )
    else:
        message = f"Normal order: {qty} kg for product {prod_id}."

    return {
        "is_anomaly":    is_anomaly,
        "anomaly_score": round(score, 4),
        "message":       message,
        "day_of_week":   day_of_week,
        "is_weekend":    is_weekend,
    }


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":      "Fishery Order Anomaly Detection API",
        "endpoints": {
            "GET  /health":         "Liveness check",
            "GET  /products":       "List all products",
            "GET  /outlets":        "List all outlets",
            "POST /predict":        "Check one order line for anomaly",
            "POST /predict/batch":  "Check many order lines at once",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok",
                    "message": "Fishery Anomaly Detection API is running"})


@app.route("/products", methods=["GET"])
def get_products():
    return jsonify(products_df.to_dict(orient="records"))


@app.route("/outlets", methods=["GET"])
def get_outlets():
    return jsonify(outlets_df.to_dict(orient="records"))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Check ONE order line for anomaly.

    Request body (JSON):
    {
        "outlet_id":      1,
        "product_id":     2,
        "quantity_in_kg": 150.0,
        "order_day":      "2026-04-18"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        row, dow, iw = _extract_features(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = _run_prediction(row, dow, iw, data)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Check MANY order lines at once.

    Request body (JSON array):
    [
    {"outlet_id": 1, "product_id": 2, "quantity_in_kg": 150.0, "order_day": "2026-04-18"},
        {"outlet_id": 3, "product_id": 5, "quantity_in_kg":  30.0, "order_day": "2026-04-18"}
    ]
    """
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array"}), 400

    results = []
    for i, item in enumerate(data):
        try:
            row, dow, iw = _extract_features(item)
            result = _run_prediction(row, dow, iw, item)
        except ValueError as exc:
            result = {"error": str(exc), "index": i}
        results.append(result)

    return jsonify(results)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
