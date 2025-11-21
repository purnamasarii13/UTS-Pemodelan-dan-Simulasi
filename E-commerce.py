# ============================================================
# 0. Import library
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Load & cek data
#    (ganti path kalau file-mu ada di folder lain)
# ============================================================

df = pd.read_csv("dataset_ecommerce (8).csv")

# pastikan kolom tanggal jadi datetime
df['order_date'] = pd.to_datetime(df['order_date'])

print("===== 5 baris pertama =====")
print(df.head())

print("\n===== Info data =====")
print(df.info())

print("\n===== Deskripsi numerik =====")
print(df.describe())

# ============================================================
# 2. Analisis deskriptif
# ============================================================

# jumlah transaksi per kurir
courier_counts = df['courier_delivery'].value_counts()
print("\n===== Jumlah transaksi per kurir =====")
print(courier_counts)

# distribusi jenis layanan
delivery_type_counts = df['type_of_delivery'].value_counts()
print("\n===== Jumlah transaksi per jenis layanan =====")
print(delivery_type_counts)

# statistik estimasi lama pengiriman per jenis layanan
delivery_time_stats = (
    df.groupby('type_of_delivery')['estimated_delivery_time_days']
      .agg(['mean', 'min', 'max', 'count'])
)
print("\n===== Statistik estimasi lama pengiriman per jenis layanan =====")
print(delivery_time_stats)

# rating rata-rata per kurir
courier_rating_stats = (
    df.groupby('courier_delivery')['product_rating']
      .agg(['mean', 'count'])
)
print("\n===== Rating rata-rata per kurir =====")
print(courier_rating_stats)

# rating rata-rata per jenis layanan
delivery_type_rating_stats = (
    df.groupby('type_of_delivery')['product_rating']
      .agg(['mean', 'count'])
)
print("\n===== Rating rata-rata per jenis layanan =====")
print(delivery_type_rating_stats)

# distribusi rating
rating_counts = df['product_rating'].value_counts().sort_index()
print("\n===== Distribusi rating produk =====")
print(rating_counts)

# ============================================================
# 3. Visualisasi dasar
# ============================================================

# (a) jumlah transaksi per kurir
plt.figure()
courier_counts.plot(kind='bar')
plt.xlabel("Kurir")
plt.ylabel("Jumlah transaksi")
plt.title("Distribusi transaksi per kurir")
plt.tight_layout()
plt.show()

# (b) jumlah transaksi per jenis layanan
plt.figure()
delivery_type_counts.plot(kind='bar')
plt.xlabel("Jenis layanan")
plt.ylabel("Jumlah transaksi")
plt.title("Distribusi transaksi per jenis layanan")
plt.tight_layout()
plt.show()

# (c) distribusi rating produk
plt.figure()
rating_counts.plot(kind='bar')
plt.xlabel("Rating produk")
plt.ylabel("Frekuensi")
plt.title("Distribusi rating produk")
plt.tight_layout()
plt.show()

# ============================================================
# 4. Time series: jumlah order per hari
# ============================================================

orders_per_day = df.groupby('order_date').size().sort_index()
print("\n===== Statistik jumlah order per hari =====")
print(orders_per_day.describe())

plt.figure()
orders_per_day.plot()
plt.xlabel("Tanggal")
plt.ylabel("Jumlah order")
plt.title("Jumlah order per hari")
plt.tight_layout()
plt.show()

# ============================================================
# 5. Fungsi simulasi Sistem Dinamik backlog & keterlambatan
# ============================================================

def simulate_backlog_and_delay(order_series,
                               initial_backlog=0,
                               base_capacity=None,
                               policy="constant",
                               backlog_threshold=150,
                               capacity_step=10,
                               max_capacity_multiplier=2.0):
    """
    order_series : pandas Series (index = tanggal, values = jumlah order)
    initial_backlog : backlog awal
    base_capacity : kapasitas dasar (jika None → pakai rata-rata order)
    policy : "constant" atau "adaptive"
    backlog_threshold : jika policy == "adaptive",
                        kapasitas naik kalau backlog > threshold
    capacity_step : kenaikan kapasitas per hari ketika backlog tinggi
    max_capacity_multiplier : batas atas kapasitas (x rata-rata)
    """

    orders = order_series.values
    T = len(orders)

    if base_capacity is None:
        base_capacity = orders.mean()

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

        # delivery rate dibatasi kapasitas & jumlah order+backlog
        available = backlog[t] + orders[t]
        delivery_rate[t] = min(available, current_capacity)

        # update backlog
        backlog[t+1] = max(0, backlog[t] + orders[t] - delivery_rate[t])

        # hitung delay perkiraan (hari)
        if current_capacity > 0:
            delay[t] = backlog[t] / current_capacity
        else:
            delay[t] = np.nan

    result = {
        "orders": orders,
        "backlog": backlog,
        "delivery_rate": delivery_rate,
        "capacity": capacity,
        "delay": delay
    }
    return result

# ============================================================
# 6. Jalankan simulasi skenario
# ============================================================

# gunakan data historis sebagai order rate
orders_series = orders_per_day

# kapasitas dasar = rata-rata jumlah order per hari
avg_orders = orders_series.mean()
print("\nRata-rata order per hari (sebagai basis kapasitas):", avg_orders)

# skenario 1: kapasitas konstan
res_constant = simulate_backlog_and_delay(
    orders_series,
    initial_backlog=0,
    base_capacity=avg_orders,
    policy="constant"
)

# skenario 2: kapasitas adaptif ketika backlog melewati ambang
res_adaptive = simulate_backlog_and_delay(
    orders_series,
    initial_backlog=0,
    base_capacity=avg_orders * 0.9,   # misal awalnya sedikit di bawah rata-rata
    policy="adaptive",
    backlog_threshold=150,
    capacity_step=15,
    max_capacity_multiplier=2.0
)

days = np.arange(len(orders_series))

# ============================================================
# 7. Plot perbandingan backlog antar skenario
# ============================================================

plt.figure()
plt.plot(days, res_constant["backlog"][:-1], label="Kapasitas konstan")
plt.plot(days, res_adaptive["backlog"][:-1], label="Kapasitas adaptif")
plt.xlabel("Hari ke-")
plt.ylabel("Backlog (pesanan)")
plt.title("Perbandingan backlog: kapasitas konstan vs adaptif")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 8. Plot perbandingan perkiraan keterlambatan
# ============================================================

plt.figure()
plt.plot(days, res_constant["delay"], label="Kapasitas konstan")
plt.plot(days, res_adaptive["delay"], label="Kapasitas adaptif")
plt.xlabel("Hari ke-")
plt.ylabel("Perkiraan keterlambatan (hari)")
plt.title("Perbandingan keterlambatan rata-rata")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 9. Plot evolusi kapasitas pada skenario adaptif
# ============================================================

plt.figure()
plt.plot(days, res_adaptive["capacity"])
plt.xlabel("Hari ke-")
plt.ylabel("Kapasitas kirim per hari (paket)")
plt.title("Perubahan kapasitas kurir (skenario adaptif)")
plt.tight_layout()
plt.show()
