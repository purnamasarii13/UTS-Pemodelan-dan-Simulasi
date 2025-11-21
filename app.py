# app.py
# Simulasi sistem dinamik backlog & keterlambatan pengiriman
# Grafik interaktif dengan Plotly (klik titik -> tampil angka)

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# 1. Load dataset saat aplikasi start
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset_ecommerce (8).csv")

df = pd.read_csv(DATA_PATH)
df["order_date"] = pd.to_datetime(df["order_date"])

# agregasi jumlah order per hari
orders_per_day = df.groupby("order_date").size().sort_index()
avg_orders = orders_per_day.mean()

# ringkasan seperlunya
dataset_summary = {
    "date_max": df["order_date"].max().date(),
}

# data untuk grafik order historis
ORDERS_DATES = [d.strftime("%Y-%m-%d") for d in orders_per_day.index]
ORDERS_VALUES = orders_per_day.values.astype(float).tolist()


# ============================================================
# 2. Fungsi simulasi sistem dinamik backlog & delay
# ============================================================

def simulate_backlog_and_delay(
    order_series,
    initial_backlog=0,
    base_capacity=None,
    policy="constant",
    backlog_threshold=150,
    capacity_step=10,
    max_capacity_multiplier=2.0,
):
    orders = order_series.values
    T = len(orders)

    if base_capacity is None:
        base_capacity = orders.mean()
    if base_capacity <= 0:
        base_capacity = 1.0  # hindari nol

    backlog = np.zeros(T + 1)
    delivery_rate = np.zeros(T)
    capacity = np.zeros(T)
    delay = np.zeros(T)

    backlog[0] = initial_backlog
    current_capacity = float(base_capacity)
    max_capacity = base_capacity * max_capacity_multiplier

    for t in range(T):
        # kebijakan kapasitas
        if policy == "adaptive":
            if backlog[t] > backlog_threshold and current_capacity < max_capacity:
                current_capacity = min(max_capacity, current_capacity + capacity_step)

        capacity[t] = current_capacity

        # delivery rate dibatasi kapasitas dan jumlah yang tersedia (backlog + order baru)
        available = backlog[t] + orders[t]
        delivery_rate[t] = min(available, current_capacity)

        # update backlog
        backlog[t + 1] = max(0, backlog[t] + orders[t] - delivery_rate[t])

        # estimasi delay (hari)
        if current_capacity > 0:
            delay[t] = backlog[t] / current_capacity
        else:
            delay[t] = np.nan

    result = {
        "orders": orders,
        "backlog": backlog,
        "delivery_rate": delivery_rate,
        "capacity": capacity,
        "delay": delay,
    }
    return result


# ============================================================
# 3. Route utama (form + hasil simulasi + data grafik)
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    # nilai default form
    form_data = {
        "base_capacity": round(float(avg_orders), 2),
        "policy": "constant",
        "backlog_threshold": 150,
        "capacity_step": 15,
        "max_capacity_multiplier": 2.0,
        "initial_backlog": 0,
        "num_days_show": min(30, len(orders_per_day)),
    }

    results = None
    sim_chart = None  # data untuk grafik backlog & delay

    if request.method == "POST":

        def get_float(name, default, min_val=None, max_val=None):
            """Ambil float dari form, dukung koma, + batas min/max."""
            raw = request.form.get(name, default)
            if isinstance(raw, str):
                raw = raw.replace(",", ".")  # dukung input 27,32
            try:
                value = float(raw)
            except (ValueError, TypeError):
                return default

            if min_val is not None and value < min_val:
                value = min_val
            if max_val is not None and value > max_val:
                value = max_val
            return value

        def get_int(name, default, min_val=None, max_val=None):
            """Ambil int dari form dengan batas min/max."""
            raw = request.form.get(name, default)
            try:
                # kalau misalnya "7,0" -> 7
                value = int(float(str(raw).replace(",", ".")))
            except (ValueError, TypeError):
                return default

            if min_val is not None and value < min_val:
                value = min_val
            if max_val is not None and value > max_val:
                value = max_val
            return value

        # --- ambil & validasi input dari user ---
        form_data["base_capacity"] = get_float(
            "base_capacity",
            form_data["base_capacity"],
            min_val=1.0,        # kapasitas minimal
        )

        form_data["policy"] = request.form.get("policy", form_data["policy"])

        form_data["backlog_threshold"] = get_float(
            "backlog_threshold",
            form_data["backlog_threshold"],
            min_val=0.0,        # backlog tidak boleh negatif
        )

        form_data["capacity_step"] = get_float(
            "capacity_step",
            form_data["capacity_step"],
            min_val=0.0,        # boleh 0 (tidak naik)
        )

        form_data["max_capacity_multiplier"] = get_float(
            "max_capacity_multiplier",
            form_data["max_capacity_multiplier"],
            min_val=0.1,        # jangan 0 atau negatif
        )

        form_data["initial_backlog"] = get_float(
            "initial_backlog",
            form_data["initial_backlog"],
            min_val=0.0,        # backlog awal minimal 0
        )

        form_data["num_days_show"] = get_int(
            "num_days_show",
            form_data["num_days_show"],
            min_val=1,
            max_val=len(orders_per_day),  # tidak boleh lebih banyak dari data
        )

        # jalankan simulasi
        sim = simulate_backlog_and_delay(
            orders_per_day,
            initial_backlog=form_data["initial_backlog"],
            base_capacity=form_data["base_capacity"],
            policy=form_data["policy"],
            backlog_threshold=form_data["backlog_threshold"],
            capacity_step=form_data["capacity_step"],
            max_capacity_multiplier=form_data["max_capacity_multiplier"],
        )

        # ambil tanggal asli dari index orders_per_day
        dates = orders_per_day.index

        n = min(form_data["num_days_show"], len(orders_per_day))
        rows = []
        for t in range(n):
            rows.append(
                {
                    "day": dates[t].strftime("%Y-%m-%d"),  # tanggal dari dataset
                    "order": sim["orders"][t],
                    "capacity": sim["capacity"][t],
                    "delivery_rate": sim["delivery_rate"][t],
                    "backlog": sim["backlog"][t],
                    "delay": sim["delay"][t],
                }
            )

        results = {
            "rows": rows,
            "num_days_show": n,
        }

        # data untuk grafik simulasi
        sim_dates = [d.strftime("%Y-%m-%d") for d in dates[:n]]
        backlog_slice = sim["backlog"][:n].astype(float).tolist()
        delay_slice = sim["delay"][:n].astype(float).tolist()

        sim_chart = {
            "dates": sim_dates,
            "backlog": backlog_slice,
            "delay": delay_slice,
        }

    # data grafik order historis (selalu ada)
    orders_chart = {
        "dates": ORDERS_DATES,
        "orders": ORDERS_VALUES,
    }

    return render_template(
        "index.html",
        summary=dataset_summary,
        form=form_data,
        results=results,
        orders_chart=orders_chart,
        sim_chart=sim_chart,
    )


# ============================================================
# 4. Entry point
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
