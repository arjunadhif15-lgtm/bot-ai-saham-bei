import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="BOT AI SAHAM BEI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BOT AI SAHAM BEI 3.0")

st.caption(
    "Scanner saham BEI dengan indikator teknikal, "
    "analisis harga dan volume, serta bantuan Gemini AI."
)

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN",
    "ANTM", "MDKA", "BRMS", "PTBA", "ADRO",
    "ITMG", "INCO", "UNTR", "ASII", "TLKM",
    "GOTO", "BRPT", "BREN", "DEWA", "BUKA",
    "WOOD", "KPIG", "BBYB", "COCO", "FORE",
    "NCKL", "EMTK", "AVIA", "VKTR", "IOTF"
]


# ==========================================
# FUNGSI MENGAMBIL DATA SAHAM
# ==========================================

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

        kolom_wajib = [
            "Open", "High", "Low", "Close", "Volume"
        ]

        if not all(k in df.columns for k in kolom_wajib):
            return None

        df = df[kolom_wajib].copy()

        for kolom in kolom_wajib:
            df[kolom] = pd.to_numeric(
                df[kolom],
                errors="coerce"
            )

        df = df.dropna()

        if len(df) < 30:
            return None

        return df

    except Exception:
        return None


# ==========================================
# FUNGSI INDIKATOR TEKNIKAL
# ==========================================

def hitung_indikator(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # EMA
    df["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
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
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "RSI"
    ] = 0

    # Volume rata-rata 20 hari
    df["AVG_VOL20"] = df["Volume"].rolling(20).mean()

    df["VOL_RATIO"] = (
        df["Volume"]
        / df["AVG_VOL20"].replace(0, np.nan)
    )

    # Harga tertinggi 20 hari sebelumnya
    df["HIGH20_PREV"] = (
        high.shift(1).rolling(20).max()
    )

    # ATR 14 untuk mengukur volatilitas
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    # Perubahan harian
    df["CHANGE_PCT"] = close.pct_change() * 100

    return df


# ==========================================
# FUNGSI ANALISIS SAHAM
# ==========================================

def analisis_saham(ticker):
    data = ambil_data(ticker)

    if data is None:
        return None

    df = hitung_indikator(data)

    x = df.iloc[-1]

    harga = float(x["Close"])
    ema9 = float(x["EMA9"])
    ema21 = float(x["EMA21"])
    ema50 = float(x["EMA50"])

    rsi = (
        float(x["RSI"])
        if pd.notna(x["RSI"])
        else 50
    )

    atr = (
        float(x["ATR14"])
        if pd.notna(x["ATR14"])
        else 0
    )

    vol_ratio = (
        float(x["VOL_RATIO"])
        if pd.notna(x["VOL_RATIO"])
        else 0
    )

    high20 = (
        float(x["HIGH20_PREV"])
        if pd.notna(x["HIGH20_PREV"])
        else 0
    )

    perubahan = (
        float(x["CHANGE_PCT"])
        if pd.notna(x["CHANGE_PCT"])
        else 0
    )

    # --------------------------------------
    # SKOR TEKNIKAL
    # --------------------------------------

    skor = 0
    alasan = []

    if harga > ema9:
        skor += 10
        alasan.append("Harga di atas EMA9")

    if ema9 > ema21:
        skor += 10
        alasan.append("EMA9 di atas EMA21")

    if harga > ema50:
        skor += 10
        alasan.append("Harga di atas EMA50")

    if 50 <= rsi <= 65:
        skor += 20
        alasan.append("Momentum RSI relatif sehat")

    elif 45 <= rsi < 50:
        skor += 10
        alasan.append("RSI mulai menguat")

    elif 65 < rsi <= 70:
        skor += 12
        alasan.append("Momentum relatif kuat")

    elif rsi > 70:
        skor += 5
        alasan.append("RSI tinggi, waspadai koreksi")

    else:
        alasan.append("Momentum RSI belum kuat")

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

    breakout = (
        high20 > 0 and harga > high20
    )

    if breakout:
        skor += 20
        alasan.append("Breakout tertinggi 20 hari")

    if perubahan > 0:
        skor += 10
        alasan.append("Harga naik pada sesi terakhir")

    skor = min(100, skor)

    # --------------------------------------
    # KLASIFIKASI TREN
    # --------------------------------------

    if harga > ema21 and ema21 > ema50:
        tren = "NAIK"

    elif harga < ema21 and ema21 < ema50:
        tren = "TURUN"

    else:
        tren = "SIDEWAYS / TRANSISI"

    # --------------------------------------
    # PROKSI AKUMULASI / TEKANAN JUAL
    # --------------------------------------

    if (
        vol_ratio >= 1.5
        and perubahan >= 0
        and harga >= ema21
    ):
        akumulasi = "INDIKASI POSITIF"

    elif vol_ratio >= 1.5 and perubahan < 0:
        akumulasi = "WASPADA TEKANAN JUAL"

    else:
        akumulasi = "BELUM TERKONFIRMASI"

    # --------------------------------------
    # SINYAL TEKNIKAL
    # --------------------------------------

    if breakout and ema9 > ema21 and 45 <= rsi <= 70:
        sinyal = "BREAKOUT POTENSIAL"

    elif skor >= 60 and harga > ema21:
        sinyal = "PANTAU"

    else:
        sinyal = "BELUM ADA SINYAL"

    # --------------------------------------
    # AREA BELI DAN TARGET SIMULASI
    # --------------------------------------

    if harga > ema21:
        entry_bawah = min(ema9, ema21)
        entry_atas = max(ema9, ema21)

    else:
        entry_bawah = max(0, harga - 0.5 * atr)
        entry_atas = harga

    acuan_entry = max(harga, entry_atas)

    risiko = 1.5 * atr

    stop_loss = max(0, acuan_entry - risiko)

    target1 = acuan_entry + risiko
    target2 = acuan_entry + (2 * risiko)

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


# ==========================================
# SIDEBAR PENGATURAN
# ==========================================

st.sidebar.header("⚙️ Pengaturan")

input_ticker = st.sidebar.text_area(
    "Daftar kode saham BEI",
    value=",".join(DEFAULT_TICKERS),
    height=180
)

modal = st.sidebar.number_input(
    "Modal simulasi (Rp)",
    min_value=100000,
    value=1000000,
    step=100000
)

if st.sidebar.button("🧹 Hapus cache data"):
    ambil_data.clear()
    st.sidebar.success("Cache berhasil dibersihkan.")


# ==========================================
# SCANNER SAHAM
# ==========================================

st.header("🔍 Scanner Saham BEI")

if st.button(
    "MULAI SCAN SAHAM",
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

    else:
        hasil = []
        gagal = []

        progress = st.progress(0)
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

            progress.progress((i + 1) / len(tickers))

        progress.empty()
        status.empty()

        if hasil:
            hasil_df = pd.DataFrame(hasil)

            hasil_df = hasil_df.sort_values(
                ["Skor", "Rasio Volume"],
                ascending=[False, False]
            ).reset_index(drop=True)

            st.session_state["hasil_df"] = hasil_df
            st.session_state["gagal"] = gagal

        else:
            st.error(
                "Data saham tidak berhasil diperoleh. "
                "Coba lagi nanti."
            )


# ==========================================
# TAMPILKAN HASIL SCANNER
# ==========================================

if "hasil_df" in st.session_state:

    df = st.session_state["hasil_df"]
    gagal = st.session_state.get("gagal", [])

    st.success(
        f"{len(df)} saham berhasil dianalisis."
    )

    st.caption(
        "Waktu tampilan: "
        + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

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
        "Indikasi Positif",
        int(
            (
                df["Indikasi Akumulasi"]
                == "INDIKASI POSITIF"
            ).sum()
        )
    )

    st.divider()

    st.subheader("🏆 Ranking Saham")

    kolom_ranking = [
        "Kode", "Harga", "Perubahan %",
        "Skor", "Tren", "RSI",
        "Rasio Volume", "Sinyal"
    ]

    st.dataframe(
        df[kolom_ranking],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔎 Filter Sinyal")

    pilihan_sinyal = st.selectbox(
        "Pilih jenis sinyal",
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

    st.subheader("🐋 Indikasi Akumulasi / Tekanan Jual")

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

    # --------------------------------------
    # RINCIAN SAHAM
    # --------------------------------------

    st.subheader("🎯 Rincian Strategi Saham")

    kode_pilihan = st.selectbox(
        "Pilih saham untuk melihat rincian",
        df["Kode"].tolist(),
        key="pilihan_rincian"
    )

    r = df[df["Kode"] == kode_pilihan].iloc[0]

    a, b, c = st.columns(3)

    a.metric("Harga", f"Rp{r['Harga']:,.2f}")
    b.metric("Skor Teknikal", f"{r['Skor']}/100")
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

    # --------------------------------------
    # GRAFIK HARGA
    # --------------------------------------

    st.subheader("📈 Grafik Harga dan EMA")

    data_grafik = ambil_data(
        kode_pilihan + ".JK"
    )

    if data_grafik is not None:

        grafik = hitung_indikator(data_grafik)

        st.line_chart(
            grafik[["Close", "EMA9", "EMA21", "EMA50"]].tail(90)
        )

        st.caption(
            "Grafik menampilkan harga penutupan dan EMA "
            "selama sekitar 90 sesi terakhir."
        )

    # --------------------------------------
    # SIMULASI LOT
    # --------------------------------------

    st.subheader("💰 Simulasi Pembelian Saham")

    modal_simulasi = st.number_input(
        "Modal yang akan disimulasikan (Rp)",
        min_value=0,
        value=1000000,
        step=100000,
        key="modal_simulasi"
    )

    harga_lot = float(r["Harga"]) * 100

    jumlah_lot = (
        int(modal_simulasi // harga_lot)
        if harga_lot > 0
        else 0
    )

    total_biaya = jumlah_lot * harga_lot

    sisa_modal = modal_simulasi - total_biaya

    c1, c2, c3 = st.columns(3)

    c1.metric("Modal", f"Rp{modal_simulasi:,.0f}")
    c2.metric("Perkiraan Lot", f"{jumlah_lot} lot")
    c3.metric("Sisa Modal", f"Rp{sisa_modal:,.0f}")

    st.caption(
        "Simulasi belum memperhitungkan biaya transaksi "
        "broker dan perubahan harga."
    )

    # --------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------

    st.subheader("📥 Unduh Hasil Scanner")

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

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
        "Skor teknikal bukan probabilitas pasti harga naik. "
        "Indikasi akumulasi hanya menggunakan proksi harga "
        "dan volume, bukan transaksi broker. Target dan "
        "stop loss merupakan simulasi, bukan jaminan."
    )

else:
    st.info(
        "Tekan tombol MULAI SCAN SAHAM untuk memulai "
        "pemindaian."
    )


# ==========================================
# ANALISIS AI GEMINI DENGAN DATA OTOMATIS
# ==========================================

st.divider()

st.header("🤖 AI Analisis Data Saham")

st.write(
    "Masukkan kode saham. Bot mengambil data harga "
    "dan volume, kemudian meminta Gemini AI "
    "menganalisis data tersebut."
)

kode_ai = st.text_input(
    "Kode saham untuk dianalisis AI",
    value="BBRI",
    key="kode_ai_otomatis"
).upper().strip()

if st.button(
    "ANALISIS DATA DENGAN AI",
    key="analisis_data_ai",
    type="primary"
):

    if not kode_ai:
        st.warning("Masukkan kode saham terlebih dahulu.")

    else:

        try:
            from google import genai

            api_key = st.secrets["GEMINI_API_KEY"]

            client = genai.Client(
                api_key=api_key
            )

            ticker_ai = kode_ai

            if not ticker_ai.endswith(".JK"):
                ticker_ai += ".JK"

            with st.spinner(
                "Mengambil data saham dan meminta analisis AI..."
            ):

                data_ai = ambil_data(ticker_ai)

                if data_ai is None:

                    st.error(
                        "Data saham tidak ditemukan atau "
                        "jumlah data belum mencukupi."
                    )

                else:

                    df_ai = hitung_indikator(data_ai)

                    terakhir = df_ai.iloc[-1]

                    harga = float(terakhir["Close"])

                    ema9 = float(terakhir["EMA9"])
                    ema21 = float(terakhir["EMA21"])
                    ema50 = float(terakhir["EMA50"])

                    rsi = (
                        float(terakhir["RSI"])
                        if pd.notna(terakhir["RSI"])
                        else None
                    )

                    volume = float(terakhir["Volume"])

                    rasio_volume = (
                        float(terakhir["VOL_RATIO"])
                        if pd.notna(terakhir["VOL_RATIO"])
                        else None
                    )

                    perubahan_harian = (
                        float(terakhir["CHANGE_PCT"])
                        if pd.notna(terakhir["CHANGE_PCT"])
                        else 0
                    )

                    if len(df_ai) >= 22:

                        perubahan_bulanan = (
                            harga / float(df_ai["Close"].iloc[-22]) - 1
                        ) * 100

                    else:
                        perubahan_bulanan = None

                    ringkasan_ai = {
                        "Kode saham": kode_ai,
                        "Tanggal data terakhir": str(
                            df_ai.index[-1].date()
                        ),
                        "Harga penutupan": round(harga, 2),
                        "Perubahan harian (%)": round(
                            perubahan_harian, 2
                        ),
                        "Perubahan sekitar 1 bulan (%)": (
                            round(perubahan_bulanan, 2)
                            if perubahan_bulanan is not None
                            else "Data belum cukup"
                        ),
                        "EMA 9": round(ema9, 2),
                        "EMA 21": round(ema21, 2),
                        "EMA 50": round(ema50, 2),
                        "RSI 14": (
                            round(rsi, 2)
                            if rsi is not None
                            else "Tidak tersedia"
                        ),
                        "Volume terakhir": int(volume),
                        "Rasio volume terhadap rata-rata 20 hari": (
                            round(rasio_volume, 2)
                            if rasio_volume is not None
                            else "Tidak tersedia"
                        )
                    }

                    prompt_ai = f"""
                    Kamu adalah asisten analis saham
                    Bursa Efek Indonesia.

                    Analisis saham berdasarkan data berikut:

                    {ringkasan_ai}

                    Buat analisis berbahasa Indonesia
                    dengan struktur berikut:

                    1. Ringkasan kondisi saham.
                    2. Tren berdasarkan EMA 9, EMA 21,
                       dan EMA 50.
                    3. Momentum berdasarkan RSI dan
                       perubahan harga.
                    4. Interpretasi volume serta indikasi
                       akumulasi atau tekanan jual.
                    5. Skenario bullish dan bearish.
                    6. Area beli, target harga, dan stop loss
                       jika data cukup.
                    7. Risiko utama dan kesimpulan.

                    Untuk target dan stop loss, jelaskan
                    dasar perhitungannya. Jika data tidak
                    cukup, jangan mengarang angka.

                    Jangan mengarang harga real-time,
                    transaksi broker, aktivitas bandar,
                    berita, atau laporan fundamental.

                    Data harga dan volume tidak membuktikan
                    adanya akumulasi oleh bandar.

                    Bedakan fakta dan perkiraan.
                    Analisis bukan jaminan keuntungan.
                    """

                    respons_ai = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt_ai
                    )

                    st.session_state["ringkasan_ai"] = ringkasan_ai
                    st.session_state["hasil_ai"] = respons_ai.text or ""

            if "ringkasan_ai" in st.session_state:

                st.subheader("📊 Data yang Dianalisis")

                st.json(
                    st.session_state["ringkasan_ai"]
                )

                st.subheader("🤖 Hasil Analisis Gemini AI")

                hasil_teks = st.session_state.get(
                    "hasil_ai", ""
                )

                if hasil_teks:
                    st.markdown(hasil_teks)

                else:
                    st.warning(
                        "Gemini tidak mengembalikan teks. "
                        "Coba lagi."
                    )

        except Exception as e:

            st.error(
                "Analisis AI gagal. Periksa koneksi, "
                "API key, kuota, dan nama model Gemini."
            )

            st.caption(str(e))


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "BOT AI SAHAM BEI 3.0 | Data historis Yahoo Finance "
    "| Analisis Gemini AI | Bukan penasihat keuangan."
)

st.caption(
    "Aplikasi ini tidak mengeksekusi transaksi otomatis. "
    "Data bisa terlambat atau tidak lengkap. "
    "Periksa kondisi emiten dan risiko sebelum bertransaksi."
)
