"""
main.py  –  Entry point

Usage
─────
  python main.py --train     # generate data + train model, then start API
  python main.py             # start API only (model must already exist)

The Flask API runs on  http://localhost:5000
Oracle APEX can call it via REST (make sure port 5000 is reachable).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Fishery Anomaly Detection")
    parser.add_argument(
        "--train", action="store_true",
        help="Generate fake data + train model before starting the API"
    )
    args = parser.parse_args()

    # ── optionally train ──────────────────────────────────────────────────────
    if args.train:
        from train import train
        train()

    # ── guard: make sure model exists ─────────────────────────────────────────
    model_path = os.path.join("model", "isolation_forest.pkl")
    if not os.path.exists(model_path):
        print("[main] Model not found. Running training first ...")
        from train import train
        train()

    # ── start Flask API ───────────────────────────────────────────────────────
    print("[main] Starting Flask API on http://0.0.0.0:5000 ...")
    from app import app
    app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
