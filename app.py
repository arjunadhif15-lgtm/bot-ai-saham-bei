import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================================
# KONFIGURASI
# =========================================

st.set_page_config(
    page_title="BOT AI SAHAM BEI 2.0",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BOT AI SAHAM BEI 2.0")
st.caption(
    "Scanner saham BEI: tren, momentum, volume, "
    "indikasi akumulasi, area beli, target dan risiko."
)

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN",
    "ANTM", "MDKA", "BRMS", "PTBA", "ADRO",
    "ITMG", "INCO", "UNTR", "ASII", "TLKM",
    "GOTO", "BRPT", "BREN", "DEWA", "BUKA",
    "WOOD", "KPIG", "BBYB", "COCO", "FORE",
    "NCKL", "EMTK", "AVIA", "VKTR", "IOTF"
]


# =========================================
# AMBIL DATA
# =========================================

@st.cache_data(ttl=300, show_spinner=False)
def ambil_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        wajib = ["Open", "High", "Low", "Close", "Volume"]

        if not all(k in df.columns for k in wajib):
            return None

        df = df[wajib].copy()

        for kolom in wajib:
            df[kolom] = pd.to_numeric(
                df[kolom], errors="coerce"
            )

        df = df.dropna()

        if len(df) < 30:
            return None

        return df

    except Exception:
        return None


# =========================================
# INDIKATOR TEKNIKAL
# =========================================

def hitung_indikator(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    # RSI 14
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0), "RSI"
    ] = 0

    # Volume rata-rata 20 hari
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    df["VOL_RATIO"] = (
        df["Volume"] /
        df["AVG_VOL20"].replace(0, np.nan)
    )

    # Breakout terhadap 20 hari sebelumnya
    df["HIGH20_PREV"] = (
        high.shift(1).rolling(20).max()
    )

    # ATR 14: ukuran volatilitas
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    # Return harian
    df["CHANGE_PCT"] = close.pct_change() * 100

    return df


# =========================================
# ANALISIS SAHAM
# =========================================

def analisis_saham(ticker):
    data = ambil_data(ticker)

    if data is None:
        return None

    df = hitung_indikator(data)

    if len(df) < 30:
        return None

    x = df.iloc[-1]

    harga = float(x["Close"])
    ema9 = float(x["EMA9"])
    ema21 = float(x["EMA21"])
    ema50 = float(x["EMA50"])

    rsi = float(x["RSI"]) if pd.notna(x["RSI"]) else 50
    atr = float(x["ATR14"]) if pd.notna(x["ATR14"]) else 0

    vol_ratio = (
        float(x["VOL_RATIO"])
        if pd.notna(x["VOL_RATIO"])
        else 0
    )

    high20_prev = (
        float(x["HIGH20_PREV"])
        if pd.notna(x["HIGH20_PREV"])
        else 0
    )

    perubahan = (
        float(x["CHANGE_PCT"])
        if pd.notna(x["CHANGE_PCT"])
        else 0
    )

    # =====================================
    # SKOR 0-100
    # =====================================

    skor = 0
    alasan = []

    # Tren harga: maksimum 30 poin
    if harga > ema9:
        skor += 10
        alasan.append("Harga di atas EMA9")

    if ema9 > ema21:
        skor += 10
        alasan.append("EMA9 di atas EMA21")

    if harga > ema50:
        skor += 10
        alasan.append("Harga di atas EMA50")

    # RSI: maksimum 20 poin
    if 50 <= rsi <= 65:
        skor += 20
        alasan.append("Momentum RSI sehat")
    elif 45 <= rsi < 50:
        skor += 10
        alasan.append("RSI mulai menguat")
    elif 65 < rsi <= 70:
        skor += 12
        alasan.append("Momentum kuat")
    elif rsi > 70:
        skor += 5
        alasan.append("RSI tinggi, waspadai koreksi")
    else:
        alasan.append("Momentum RSI belum kuat")

    # Volume: maksimum 20 poin
    if vol_ratio >= 2:
        skor += 20
        alasan.append("Volume lebih dari 2x rata-rata")
    elif vol_ratio >= 1.5:
        skor += 15
        alasan.append("Volume meningkat")
    elif vol_ratio >= 1.2:
        skor += 10
        alasan.append("Volume di atas rata-rata")
    elif vol_ratio >= 1:
        skor += 5

    # Breakout: maksimum 20 poin
    breakout = (
        high20_prev > 0 and harga > high20_prev
    )

    if breakout:
        skor += 20
        alasan.append("Breakout high 20 hari")

    # Pergerakan harga: maksimum 10 poin
    if perubahan > 0:
        skor += 10
        alasan.append("Harga naik pada sesi terakhir")

    skor = min(100, max(0, skor))

    # =====================================
    # KLASIFIKASI TREN
    # =====================================

    if harga > ema21 and ema21 > ema50:
        tren = "NAIK"
    elif harga < ema21 and ema21 < ema50:
        tren = "TURUN"
    else:
        tren = "SIDEWAYS / TRANSISI"

    # =====================================
    # INDIKASI AKUMULASI
    # =====================================

    # Ini hanya indikator proksi dari harga dan volume,
    # bukan data transaksi broker atau bukti bandar.

    indikasi_akumulasi = (
        vol_ratio >= 1.5
        and perubahan >= 0
        and harga >= ema21
    )

    if indikasi_akumulasi:
        akumulasi = "INDIKASI POSITIF"
    elif vol_ratio >= 1.5 and perubahan < 0:
        akumulasi = "WASPADA TEKANAN JUAL"
    else:
        akumulasi = "BELUM TERKONFIRMASI"

    # =====================================
    # SINYAL
    # =====================================

    if breakout and ema9 > ema21 and 45 <= rsi <= 70:
        sinyal = "BREAKOUT POTENSIAL"
    elif skor >= 60 and harga > ema21:
        sinyal = "PANTAU"
    else:
        sinyal = "BELUM ADA SINYAL"

    # =====================================
    # AREA BELI DAN RISIKO
    # =====================================

    # Area entry merupakan estimasi berbasis EMA,
    # bukan rekomendasi beli otomatis.

    if harga > ema21:
        entry_bawah = min(ema9, ema21)
        entry_atas = max(ema9, ema21)
    else:
        entry_bawah = harga - (0.5 * atr)
        entry_atas = harga

    # Harga acuan untuk simulasi risiko
    acuan_entry = max(
        harga,
        entry_atas
    )

    stop_loss = acuan_entry - (1.5 * atr)

    risiko_per_saham = max(
        acuan_entry - stop_loss, 0
    )

    target1 = acuan_entry + risiko_per_saham
    target2 = acuan_entry + (2 * risiko_per_saham)

    # Pastikan nilai tetap positif
    stop_loss = max(0, stop_loss)

    return {
        "Kode": ticker.replace(".JK", ""),
        "Harga": round(harga, 2),
        "Perubahan %": round(perubahan, 2),
        "Skor": skor,
        "Tren": tren,
        "RSI": round(rsi, 2),
        "Rasio Volume": round(vol_ratio, 2),
        "Breakout": "YA" if breakout else "TIDAK",
        "Indikasi Akumulasi": akumulasi,
        "Sinyal": sinyal,
        "Area Beli Bawah": round(entry_bawah, 2),
        "Area Beli Atas": round(entry_atas, 2),
        "Stop Loss": round(stop_loss, 2),
        "Target 1": round(target1, 2),
        "Target 2": round(target2, 2),
        "Alasan": ", ".join(alasan)
    }


# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("⚙️ Pengaturan")

input_ticker = st.sidebar.text_area(
    "Daftar kode saham",
    value=",".join(DEFAULT_TICKERS),
    height=180
)

modal = st.sidebar.number_input(
    "Modal simulasi (Rp)",
    min_value=100000,
    value=1000000,
    step=100000
)

if st.sidebar.button("Hapus cache data"):
    ambil_data.clear()
    st.sidebar.success("Cache berhasil dibersihkan.")


# =========================================
# MULAI SCAN
# =========================================

if st.button(
    "🔍 MULAI SCAN SAHAM",
    type="primary",
    use_container_width=True
):

    tickers = []

    for item in input_ticker.split(","):
        kode = item.strip().upper()

        if not kode:
            continue

        if not kode.endswith(".JK"):
            kode += ".JK"

        if kode not in tickers:
            tickers.append(kode)

    if not tickers:
        st.error("Masukkan minimal satu kode saham.")
        st.stop()

    hasil = []
    gagal = []

    bar = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(tickers):
        status.write(
            f"Memindai {ticker}: "
            f"{i + 1} dari {len(tickers)}"
        )

        try:
            item = analisis_saham(ticker)

            if item is not None:
                hasil.append(item)
            else:
                gagal.append(ticker)

        except Exception:
            gagal.append(ticker)

        bar.progress((i + 1) / len(tickers))

    bar.empty()
    status.empty()

    if hasil:
        hasil_df = pd.DataFrame(hasil)

        hasil_df = hasil_df.sort_values(
            ["Skor", "Rasio Volume"],
            ascending=[False, False]
        ).reset_index(drop=True)

        st.session_state["hasil_df"] = hasil_df
        st.session_state["gagal"] = gagal
        st.session_state["modal"] = modal

    else:
        st.error(
            "Data tidak berhasil diperoleh. "
            "Coba lagi nanti."
        )


# =========================================
# HASIL
# =========================================

if "hasil_df" in st.session_state:

    df = st.session_state["hasil_df"]
    gagal = st.session_state.get("gagal", [])
    modal_hasil = st.session_state.get("modal", modal)

    st.success(
        f"Pemindaian selesai: {len(df)} saham berhasil dianalisis."
    )

    st.caption(
        "Waktu tampilan: "
        + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

    # Ringkasan
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Saham Dipindai", len(df))

    c2.metric(
        "Breakout Potensial",
        int((df["Sinyal"] == "BREAKOUT POTENSIAL").sum())
    )

    c3.metric(
        "Tren Naik",
        int((df["Tren"] == "NAIK").sum())
    )

    c4.metric(
        "Indikasi Akumulasi",
        int((df["Indikasi Akumulasi"] == "INDIKASI POSITIF").sum())
    )

    st.divider()

    # Ranking
    st.subheader("🏆 RANKING SAHAM TERBAIK")

    st.dataframe(
        df[
            [
                "Kode", "Harga", "Perubahan %",
                "Skor", "Tren", "RSI",
                "Rasio Volume", "Sinyal"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Filter
    st.subheader("🔎 FILTER SAHAM")

    pilihan_sinyal = st.selectbox(
        "Pilih sinyal",
        [
            "SEMUA",
            "BREAKOUT POTENSIAL",
            "PANTAU",
            "BELUM ADA SINYAL"
        ]
    )

    if pilihan_sinyal == "SEMUA":
        tampilan = df.copy()
    else:
        tampilan = df[
            df["Sinyal"] == pilihan_sinyal
        ].copy()

    st.dataframe(
        tampilan,
        use_container_width=True,
        hide_index=True
    )

    # Indikasi akumulasi
    st.subheader("🐋 INDIKASI AKUMULASI / TEKANAN JUAL")

    st.dataframe(
        df[
            [
                "Kode", "Harga", "Perubahan %",
                "Rasio Volume", "Tren",
                "Indikasi Akumulasi"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Rincian
    st.subheader("🎯 RINCIAN STRATEGI SAHAM")

    kode_pilihan = st.selectbox(
        "Pilih saham",
        df["Kode"].tolist()
    )

    r = df[df["Kode"] == kode_pilihan].iloc[0]

    a, b, c = st.columns(3)

    a.metric("Harga", f"Rp{r['Harga']:,.2f}")
    b.metric("Skor", f"{r['Skor']}/100")
    c.metric("Tren", r["Tren"])

    st.write("**Sinyal:**", r["Sinyal"])
    st.write("**RSI:**", r["RSI"])
    st.write("**Rasio volume:**", r["Rasio Volume"], "kali")
    st.write("**Breakout:**", r["Breakout"])

    st.write(
        "**Indikasi akumulasi:**",
        r["Indikasi Akumulasi"]
    )

    st.write(
        "**Area beli potensial:**",
        f"Rp{r['Area Beli Bawah']:,.2f} - "
        f"Rp{r['Area Beli Atas']:,.2f}"
    )

    st.write(
        "**Stop loss simulasi:**",
        f"Rp{r['Stop Loss']:,.2f}"
    )

    st.write(
        "**Target 1 simulasi:**",
        f"Rp{r['Target 1']:,.2f}"
    )

    st.write(
        "**Target 2 simulasi:**",
        f"Rp{r['Target 2']:,.2f}"
    )

    st.write("**Alasan skor:**", r["Alasan"])

    # Simulasi modal
    st.subheader("💰 SIMULASI LOT")

    harga_lot = float(r["Harga"]) * 100

    jumlah_lot = (
        int(modal_hasil // harga_lot)
        if harga_lot > 0 else 0
    )

    total_biaya = jumlah_lot * harga_lot

    sisa = modal_hasil - total_biaya

    c1, c2, c3 = st.columns(3)

    c1.metric("Modal", f"Rp{modal_hasil:,.0f}")
    c2.metric("Perkiraan Lot", f"{jumlah_lot} lot")
    c3.metric("Sisa Modal", f"Rp{sisa:,.0f}")

    # Download
    st.subheader("📥 DOWNLOAD HASIL")

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Download hasil analisis CSV",
        data=csv,
        file_name="hasil_bot_ai_saham.csv",
        mime="text/csv",
        use_container_width=True
    )

    if gagal:
        with st.expander("Saham yang gagal dipindai"):
            st.write(", ".join(gagal))

    st.warning(
        "Skor adalah penilaian teknikal berbasis aturan, "
        "bukan probabilitas pasti harga naik. Indikasi "
        "akumulasi menggunakan proksi harga dan volume, "
        "bukan broker summary. Area beli, target, dan stop "
        "loss adalah simulasi. Data Yahoo Finance dapat "
        "terlambat atau tidak tersedia. Periksa likuiditas, "
        "kondisi emiten, dan risiko sebelum bertransaksi."
    )

else:
    st.info(
        "Tekan tombol MULAI SCAN SAHAM untuk memulai."
    )

st.divider()

st.caption(
    "BOT AI SAHAM BEI 2.0 | Data harian Yahoo Finance. "
    "Bukan data real-time dan belum mengeksekusi order otomatis."
)

# ==============================
# AI ANALIS SAHAM GEMINI
# ==============================

st.divider()
st.header("🤖 AI Analyst Saham BEI")

st.write(
    "Gunakan AI untuk membantu menganalisis "
    "saham Indonesia berdasarkan informasi yang tersedia."
)

ticker_ai = st.text_input(
    "Kode saham",
    value="BBRI",
    key="ticker_ai"
).upper().strip()

data_saham = st.text_area(
    "Data saham (opsional)",
    placeholder=(
        "Contoh: harga sekarang 3500, "
        "EMA9 di atas EMA21, RSI 60, "
        "volume meningkat, laba bersih naik."
    ),
    key="data_saham_ai"
)

pertanyaan_ai = st.text_area(
    "Apa yang ingin dianalisis?",
    value=(
        "Analisis tren, momentum, risiko, "
        "area beli, target harga, dan stop loss. "
        "Jelaskan skenario bullish dan bearish."
    ),
    key="pertanyaan_ai"
)

if st.button("🔍 ANALISIS DENGAN AI", key="tombol_ai"):

    try:
        from google import genai

        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
        Kamu adalah asisten analisis saham Bursa Efek Indonesia.

        Kode saham: {ticker_ai}
        Data yang diberikan pengguna: {data_saham}
        Permintaan analisis: {pertanyaan_ai}

        Buat analisis dalam bahasa Indonesia dengan bagian:
        1. Ringkasan kondisi saham.
        2. Analisis teknikal jika datanya tersedia.
        3. Analisis fundamental jika datanya tersedia.
        4. Skenario bullish dan bearish.
        5. Area beli dan target harga hanya jika data cukup.
        6. Stop loss dan manajemen risiko.
        7. Kesimpulan dan risiko utama.

        Jangan mengarang harga terkini, laporan keuangan,
        transaksi broker, atau data bandar.
        Jika data tidak tersedia, katakan dengan jujur.
        Bedakan fakta, asumsi, dan perkiraan.
        Analisis bukan jaminan keuntungan.
        """

        with st.spinner("AI sedang menganalisis saham..."):

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        st.subheader("📊 Hasil Analisis AI")
        st.markdown(response.text)

    except Exception as e:
        st.error(
            "Analisis AI gagal. Periksa API key, "
            "kuota, koneksi, dan nama model Gemini."
        )
        st.caption(str(e))
