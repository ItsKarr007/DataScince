from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Data ────────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(BASE_DIR, "index.csv"))
df = df_raw.copy()

# ── Cleaning ────────────────────────────────────────────────
df = df.dropna(subset=['coffee_name', 'cash_type'])

df['money'] = df['money'].fillna(df['money'].mean())

df['coffee_name'] = df['coffee_name'].str.strip().str.lower()
df['cash_type']   = df['cash_type'].str.strip().str.lower()

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])

# ── Remove Outlier (IQR) ─────────────────────────────────────
Q1 = df['money'].quantile(0.25)
Q3 = df['money'].quantile(0.75)
IQR = Q3 - Q1

df_final = df[
    (df['money'] >= Q1 - 1.5 * IQR) &
    (df['money'] <= Q3 + 1.5 * IQR)
].copy()

# ── Feature Engineering ─────────────────────────────────────
df_final['date']  = df_final['datetime'].dt.date
df_final['hour']  = df_final['datetime'].dt.hour
df_final['month'] = df_final['datetime'].dt.strftime('%Y-%m')

# Urutan hari biar tidak acak
df_final['day'] = pd.Categorical(
    df_final['datetime'].dt.day_name(),
    categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    ordered=True
)

# ── Helper IQR ──────────────────────────────────────────────
def iqr_stats(s):
    s = s.dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = s[(s < lower) | (s > upper)]
    return Q1, Q3, IQR, lower, upper, outliers


# ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── KPI ─────────────────────────────────────────────────────
@app.route("/api/kpi")
def api_kpi():
    return jsonify({
        "total_transaksi": int(len(df_final)),
        "total_pendapatan": round(float(df_final['money'].sum()), 2),
        "rata_money": round(float(df_final['money'].mean()), 2),
        "jenis_kopi": int(df_final['coffee_name'].nunique()),
        "pct_card": round(float((df_final['cash_type'] == 'card').mean() * 100), 1),
        "hari_aktif": int(df_final['date'].nunique()),
    })


# ── Statistik ───────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    result = {}

    s = df_final['money']
    Q1, Q3, IQR, lower, upper, outliers = iqr_stats(s)

    result['money'] = {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "q1": round(float(Q1), 2),
        "q3": round(float(Q3), 2),
        "iqr": round(float(IQR), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "outlier_count": int(len(outliers)),
        "skewness": round(float(s.skew()), 2),
        "kurtosis": round(float(s.kurt()), 2),
        "missing": int(df_raw['money'].isna().sum()),
        "variance": round(float(s.var()), 2),
        "range": round(float(s.max() - s.min()), 2),
    }

    for col in ['coffee_name', 'cash_type']:
        vc = df_final[col].value_counts()

        result[col] = {
            "count": int(df_final[col].count()),
            "unique": int(df_final[col].nunique()),
            "top": str(vc.index[0]) if len(vc) > 0 else "-",
            "freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "missing": int(df_raw[col].isna().sum()),
            "value_counts": {k: int(v) for k, v in vc.items()},
        }

    return jsonify(result)


# ── Histogram ───────────────────────────────────────────────
@app.route("/api/histogram")
def api_histogram():
    s = df_final['money']
    counts, edges = np.histogram(s, bins=20)

    labels = [f"{edges[i]:.2f}" for i in range(len(edges)-1)]

    return jsonify({
        "labels": labels,
        "values": counts.tolist()
    })


# ── Boxplot ─────────────────────────────────────────────────
@app.route("/api/boxplot")
def api_boxplot():
    s = df_final['money']
    Q1, Q3, IQR, lower, upper, outliers = iqr_stats(s)

    return jsonify({
        "min": round(float(s[s >= lower].min()), 2),
        "q1": round(float(Q1), 2),
        "median": round(float(s.median()), 2),
        "q3": round(float(Q3), 2),
        "max": round(float(s[s <= upper].max()), 2),
        "outliers": outliers.round(2).tolist(),
    })


# ── Monthly ────────────────────────────────────────────────
@app.route("/api/monthly")
def api_monthly():
    monthly = df_final.groupby('month')['money'].sum().sort_index()

    return jsonify({
        "labels": monthly.index.tolist(),
        "values": monthly.round(2).tolist()
    })


# ── Hourly ─────────────────────────────────────────────────
@app.route("/api/hourly")
def api_hourly():
    hourly = df_final.groupby('hour')['money'].sum().sort_index()
    count  = df_final.groupby('hour')['money'].count().sort_index()

    return jsonify({
        "labels": hourly.index.tolist(),
        "values": hourly.round(2).tolist(),
        "counts": count.tolist(),
    })


# ── Top Produk ─────────────────────────────────────────────
@app.route("/api/top_products")
def api_top_products():
    cnt = df_final['coffee_name'].value_counts().head(10)
    rev = df_final.groupby('coffee_name')['money'].sum().reindex(cnt.index)

    return jsonify({
        "labels": cnt.index.str.title().tolist(),
        "counts": cnt.values.tolist(),
        "revenue": rev.round(2).tolist(),
    })


# ── Payment ────────────────────────────────────────────────
@app.route("/api/payment")
def api_payment():
    vc  = df_final['cash_type'].value_counts()
    rev = df_final.groupby('cash_type')['money'].sum().reindex(vc.index)

    return jsonify({
        "labels": vc.index.tolist(),
        "counts": vc.values.tolist(),
        "revenue": rev.round(2).tolist(),
    })


# ── Heatmap ────────────────────────────────────────────────
@app.route("/api/heatmap")
def api_heatmap():
    pivot = df_final.pivot_table(
        values='money',
        index='day',
        columns='hour',
        aggfunc='sum',
        fill_value=0
    ).sort_index()

    return jsonify({
        "days": pivot.index.tolist(),
        "hours": [int(c) for c in pivot.columns],
        "matrix": pivot.values.round(2).tolist(),
    })


# ── Scatter ────────────────────────────────────────────────
@app.route("/api/scatter")
def api_scatter():
    if len(df_final) == 0:
        return jsonify({"x": [], "y": [], "coffee": []})

    sample = df_final[['hour','money','coffee_name']].sample(
        min(400, len(df_final)), random_state=42
    )

    return jsonify({
        "x": sample['hour'].tolist(),
        "y": sample['money'].round(2).tolist(),
        "coffee": sample['coffee_name'].str.title().tolist(),
    })


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)