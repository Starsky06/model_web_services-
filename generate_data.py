"""
generate_data.py
Generates fake historical order data based on the ERD:
  - Product       (product_id, product_name)
  - Outlet        (outlet_id, outlet_name)
  - Order         (order_id, order_day, outlet_id)
  - Order_Product (order_id, product_id, quantity_in_kg)

About 2% of quantity values are intentionally injected as anomalies
(3-5x the normal quantity) so the ML model has something to learn.
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ── seed for reproducibility ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ── static lookup tables ─────────────────────────────────────────────────────
PRODUCTS = [
    (1,  "Salmon"),
    (2,  "Tuna"),
    (3,  "Cod"),
    (4,  "Mackerel"),
    (5,  "Seabass"),
    (6,  "Red Snapper"),
    (7,  "Grouper"),
    (8,  "Prawns"),
    (9,  "Squid"),
    (10, "Tilapia"),
]

OUTLETS = [
    (1, "Outlet KL Central"),
    (2, "Outlet Petaling Jaya"),
    (3, "Outlet Subang Jaya"),
    (4, "Outlet Shah Alam"),
    (5, "Outlet Klang"),
]


def generate_fake_data(
    start: str = "2024-01-01",
    end: str = "2026-04-01",
    anomaly_ratio: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns four DataFrames:
        products_df, outlets_df, orders_df, order_products_df
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    orders: list[dict]         = []
    order_products: list[dict] = []
    order_id = 1

    # ── generate day-by-day ──────────────────────────────────────────────────
    for day_offset in range((end_dt - start_dt).days):
        order_date   = start_dt + timedelta(days=day_offset)
        day_of_week  = order_date.weekday()   # 0 = Monday �?6 = Sunday
        is_weekend   = day_of_week >= 5

        for outlet in OUTLETS:
            # Outlets order more on weekdays; less on weekends
            avg_orders = 0.5 if is_weekend else 1.5
            num_orders = int(np.random.poisson(avg_orders))

            for _ in range(num_orders):
                num_products      = random.randint(1, 4)
                selected_products = random.sample(PRODUCTS, num_products)

                orders.append({
                    "order_id":  order_id,
                    "order_day": order_date.strftime("%Y-%m-%d"),
                    "outlet_id": outlet[0],
                })

                for product in selected_products:
                    # Normal distribution �?weekends slightly smaller batches
                    mean = 25.0 if is_weekend else 40.0
                    qty  = max(5.0, round(float(np.random.normal(mean, 12)), 2))

                    order_products.append({
                        "order_id":      order_id,
                        "product_id":    product[0],
                        "quantity_in_kg": qty,
                    })

                order_id += 1

    # ── inject anomalies ─────────────────────────────────────────────────────
    n_anomalies = max(1, int(len(order_products) * anomaly_ratio))
    anomaly_idx = random.sample(range(len(order_products)), n_anomalies)
    for idx in anomaly_idx:
        multiplier = random.uniform(3.0, 5.0)
        order_products[idx]["quantity_in_kg"] = round(
            order_products[idx]["quantity_in_kg"] * multiplier, 2
        )

    # ── build DataFrames ─────────────────────────────────────────────────────
    products_df       = pd.DataFrame(PRODUCTS,       columns=["product_id", "product_name"])
    outlets_df        = pd.DataFrame(OUTLETS,        columns=["outlet_id",  "outlet_name"])
    orders_df         = pd.DataFrame(orders)
    order_products_df = pd.DataFrame(order_products)

    print(f"[generate_data] {len(orders_df):,} orders | "
          f"{len(order_products_df):,} order-product rows | "
          f"{n_anomalies} anomalies injected")

    return products_df, outlets_df, orders_df, order_products_df


def save_data(output_dir: str = "data") -> None:
    """Generate fake data and save all four CSVs to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    products_df, outlets_df, orders_df, order_products_df = generate_fake_data()

    products_df.to_csv(       os.path.join(output_dir, "products.csv"),       index=False)
    outlets_df.to_csv(        os.path.join(output_dir, "outlets.csv"),        index=False)
    orders_df.to_csv(         os.path.join(output_dir, "orders.csv"),         index=False)
    order_products_df.to_csv( os.path.join(output_dir, "order_products.csv"), index=False)

    print(f"[generate_data] CSVs saved to '{output_dir}/'")


if __name__ == "__main__":
    save_data()
