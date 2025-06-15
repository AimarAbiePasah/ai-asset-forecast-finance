import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import ta
from PIL import Image


st.set_page_config(layout="wide")


st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #eef2f7, #dbe6f3);
    font-family: 'Segoe UI', sans-serif;
}
.big-font {
    font-size: 36px !important;
    font-weight: 800;
    background: -webkit-linear-gradient(#1d2b64, #f8cdda);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-font {
    font-size: 16px !important;
    color: #555;
    margin-top: -8px;
}
.card {
    background-color: #ffffffcc;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-top: 10px;
}
.metric-container {
    background: #ffffff;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    transition: transform 0.2s ease-in-out;
}
.metric-container:hover {
    transform: scale(1.02);
}
.footer {
    font-size: 13px;
    color: #888;
    text-align: center;
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)


icon = Image.open("logo.jpg")  
col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.image(icon, width=1000)

with col_title:
    st.markdown("<div class='big-font'>AI-Based Asset Forecast</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-font'>Prediksi harga saham berdasarkan indikator teknikal dan model Random Forest</div>", unsafe_allow_html=True)

st.markdown("---")

# === UI Input 2 Kolom ===
col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("📌 Kode Saham (misal: BBCA.JK, TLKM.JK)", "BBCA.JK").upper()
    tanggal_awal = st.date_input("📅 Tanggal Awal Prediksi", pd.to_datetime("2025-06-16"))
    tanggal_akhir = st.date_input("📅 Tanggal Akhir Prediksi", pd.to_datetime("2025-06-20"))
    run = st.button("🔍 Proses Prediksi")

with col2:
    st.markdown("#### 📘 Penjelasan")
    st.markdown("""
    <div class='card'>
        <ul>
            <li>📊 Fitur: RSI, MACD, MA5, Volatilitas, Volume</li>
            <li>🔎 Data historis 90 hari</li>
            <li>📅 Hari libur dilewati otomatis</li>
            <li>🤖 Algoritma: Random Forest Regressor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === LOGIKA UTAMA ===
if run:
    try:
        start_pred = pd.to_datetime(tanggal_awal)
        end_pred = pd.to_datetime(tanggal_akhir)
        today = pd.to_datetime("today").normalize()

        if start_pred >= end_pred:
            st.error("❌ Tanggal akhir harus setelah tanggal awal.")
            st.stop()

        data = yf.download(ticker, start=start_pred - timedelta(days=90), end=start_pred, interval="1d", auto_adjust=True)
        data.dropna(inplace=True)

        if len(data) < 30:
            st.warning("⚠️ Data historis kurang dari 30 hari. Akurasi model mungkin terbatas.")
        data = data.tail(30).copy()

        # Fitur teknikal
        data["Price"] = data["Close"]
        data["Day_Index"] = np.arange(len(data))
        data["Return"] = data["Price"].pct_change()
        data["MA5"] = data["Price"].rolling(window=5).mean()
        data["Volatility"] = data["Price"].rolling(window=5).std()
        data["RSI"] = ta.momentum.RSIIndicator(data["Price"], window=14).rsi()
        data["MACD"] = ta.trend.MACD(data["Price"]).macd()
        data["Target"] = data["Price"].shift(-1)
        data.dropna(inplace=True)

        features = ["Price", "Day_Index", "Return", "MA5", "Volatility", "RSI", "MACD", "Volume"]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(data[features], data["Target"])

        tanggal_target = pd.date_range(start=start_pred, end=end_pred, freq="B")
        last_row = data.iloc[-1].copy()
        results = []

        for tanggal in tanggal_target:
            check = yf.download(ticker, start=tanggal, end=tanggal + timedelta(days=1), interval='1d', auto_adjust=True)

            if not check.empty and tanggal <= today:
                harga = check["Close"].iloc[0]
                tipe = "data aktual"
                last_row["Price"] = harga
                last_row["Return"] = data["Return"].mean()
                last_row["MA5"] = data["MA5"].mean()
                last_row["Volatility"] = data["Volatility"].mean()
                last_row["RSI"] = data["RSI"].mean()
                last_row["MACD"] = data["MACD"].mean()
                last_row["Volume"] = check["Volume"].iloc[0]
            elif check.empty and tanggal <= today:
                continue
            else:
                pred_input = pd.DataFrame([{
                    "Price": last_row["Price"],
                    "Day_Index": last_row["Day_Index"] + 1,
                    "Return": last_row["Return"],
                    "MA5": last_row["MA5"],
                    "Volatility": last_row["Volatility"],
                    "RSI": last_row["RSI"],
                    "MACD": last_row["MACD"],
                    "Volume": last_row["Volume"]
                }])
                harga = model.predict(pred_input)[0]
                tipe = "prediksi"
                last_row["Price"] = harga
                last_row["MA5"] = (last_row["MA5"] * 4 + harga) / 5
                last_row["Volatility"] = data["Volatility"].mean()
                last_row["RSI"] = data["RSI"].mean()
                last_row["MACD"] = data["MACD"].mean()
                last_row["Volume"] = data["Volume"].mean()

            last_row["Day_Index"] += 1
            results.append({"Tanggal": tanggal.date(), "Harga": round(float(harga), 2), "Tipe": tipe})

        hasil_df = pd.DataFrame(results)

        if hasil_df.empty:
            st.warning("⚠️ Tidak ada hari kerja aktif (dengan data atau bisa diprediksi).")
            st.stop()

        # Tabel
        st.markdown("### 📋 Hasil Prediksi")
        st.dataframe(hasil_df.set_index("Tanggal"))

        # Ambil harga acuan Jumat terakhir
        hasil_df["Tanggal"] = pd.to_datetime(hasil_df["Tanggal"])
        jumat_df = hasil_df[(hasil_df["Tanggal"].dt.weekday == 4) & (hasil_df["Tipe"] == "data aktual")]
        harga_awal = jumat_df.iloc[-1]["Harga"] if not jumat_df.empty else hasil_df.iloc[0]["Harga"]

        pred_only = hasil_df[hasil_df["Tipe"] == "prediksi"]
        harga_akhir = pred_only.iloc[-1]["Harga"] if not pred_only.empty else hasil_df.iloc[-1]["Harga"]

        selisih = harga_akhir - harga_awal
        persen = (selisih / harga_awal) * 100
        rekomendasi = "BUY" if persen > 0.5 else "SELL" if persen < -0.5 else "HOLD"

        col3, col4, col5 = st.columns(3)
        col3.markdown(f"<div class='metric-container'><h5>📈 Harga Jumat</h5><h4>Rp{harga_awal:,.2f}</h4></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-container'><h5>📉 Akhir Prediksi</h5><h4>Rp{harga_akhir:,.2f}<br>({persen:.2f}%)</h4></div>", unsafe_allow_html=True)
        col5.markdown(f"<div class='metric-container'><h5>📌 Rekomendasi</h5><h4>{rekomendasi}</h4></div>", unsafe_allow_html=True)

        # Interpretasi
        st.markdown("---")
        st.subheader("💬 Interpretasi")
        st.markdown(f"""
        - Acuan: **Penutupan Jumat terakhir**
        - Harga akhir: **Hasil akhir prediksi**
        - Kenaikan: **Rp{selisih:,.2f}** ({persen:.2f}%)
        - Rekomendasi AI: **`{rekomendasi}`**
        """)

        st.markdown("<div class='footer'>© 2025 AsetCerdas.ai — Aimar Abie Pasah</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
