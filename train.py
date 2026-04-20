"""
train.py
Trains an Isolation Forest anomaly-detection model on the fake order data.

Features used:
  quantity_in_kg – the raw order quantity
  day_of_week   – 0 (Monday) … 6 (Sunday)
  is_weekend    – 1 if Saturday/Sunday
  month         – 1–12 (seasonal pattern)
  product_id    – encoded as integer
  outlet_id     – encoded as integer

The trained model + scaler are saved to the 'model/' folder so the
Flask API can load them at startup.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from generate_data import generate_fake_data, save_data

# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
FEATURES  = ["quantity_in_kg", "day_of_week", "is_weekend", "month",
             "product_id", "outlet_id"]


def build_feature_matrix(orders_df: pd.DataFrame,
                         order_products_df: pd.DataFrame) -> pd.DataFrame:
    """Merge tables and engineer time features."""
    df = order_products_df.merge(orders_df, on="order_id")
    df["order_day"]   = pd.to_datetime(df["order_day"])
    df["day_of_week"] = df["order_day"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["order_day"].dt.month
    return df


def train(contamination: float = 0.02, n_estimators: int = 100) -> None:
    """
    1. Generate (or load) fake data.
    2. Build feature matrix.
    3. Fit IsolationForest + StandardScaler.
    4. Save artefacts to MODEL_DIR.
    """
    # ── data ─────────────────────────────────────────────────────────────────
    orders_csv         = os.path.join(DATA_DIR, "orders.csv")
    order_products_csv = os.path.join(DATA_DIR, "order_products.csv")

    if os.path.exists(orders_csv) and os.path.exists(order_products_csv):
        print("[train] Loading existing CSVs from data/ ...")
        orders_df         = pd.read_csv(orders_csv)
        order_products_df = pd.read_csv(order_products_csv)
    else:
        print("[train] No CSVs found – generating fake data ...")
        save_data(DATA_DIR)
        orders_df         = pd.read_csv(orders_csv)
        order_products_df = pd.read_csv(order_products_csv)

    # ── features ─────────────────────────────────────────────────────────────
    df = build_feature_matrix(orders_df, order_products_df)
    X  = df[FEATURES].values

    # ── scale ─────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"[train] Training IsolationForest on {len(X):,} rows "
          f"(contamination={contamination}, n_estimators={n_estimators}) ...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # ── quick report ──────────────────────────────────────────────────────────
    preds         = model.predict(X_scaled)
    n_flagged     = int((preds == -1).sum())
    flagged_pct   = n_flagged / len(preds) * 100
    print(f"[train] Flagged {n_flagged:,} anomalies ({flagged_pct:.2f}% of training set)")

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"[train] Model + scaler saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    train()
