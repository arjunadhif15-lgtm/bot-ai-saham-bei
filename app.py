import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Bot AI Saham BEI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BOT AI SAHAM BEI")
st.caption(
    "Pemindai saham Indonesia berdasarkan indikator teknikal. "
    "Gunakan hasil sebagai bahan analisis, bukan jaminan keuntungan."
)

# ==========================================
# DAFTAR SAHAM
# ==========================================

DEFAULT_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
    "BBTN.JK", "ANTM.JK", "MDKA.JK", "BRMS.JK",
    "PTBA.JK", "ADRO.JK", "ITMG.JK", "INCO.JK",
    "UNTR.JK", "ASII.JK", "TLKM.JK", "GOTO.JK",
    "BRPT.JK", "BREN.JK", "DEWA.JK", "BUKA.JK",
    "WOOD.JK", "KPIG.JK", "BBYB.JK", "COCO.JK",
    "FORE.JK", "NCKL.JK", "EMTK.JK", "AVIA.JK",
    "VKTR.JK", "IOTF.JK"
]

# ==========================================
# FUNGSI MENGAMBIL DATA SAHAM
# ==========================================

@st.cache_data(ttl=300, show_spinner=False)
def ambil_data(ticker):
    try:
        data = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        # Menangani format kolom yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        kolom_wajib = ["Close", "High", "Low", "Volume"]

        for kolom in kolom_wajib:
            if kolom not in data.columns:
                return None

        data = data[kolom_wajib].copy()

        for kolom in kolom_wajib:
            data[kolom] = pd.to_numeric(
                data[kolom], errors="coerce"
            )

        data = data.dropna()

        if len(data) < 30:
            return None

        return data

    except Exception:
        return None


# ==========================================
# FUNGSI MENGHITUNG INDIKATOR
# ==========================================

def hitung_indikator(data):
    df = data.copy()

    # Exponential Moving Average
    df["EMA9"] = df["Close"].ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = df["Close"].ewm(
        span=21, adjust=False
    ).mean()

    # RSI 14
    perubahan = df["Close"].diff()

    kenaikan = perubahan.clip(lower=0)
    penurunan = -perubahan.clip(upper=0)

    rata_naik = kenaikan.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rata_turun = penurunan.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = rata_naik / rata_turun.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))
    df["RSI14"] = df["RSI14"].fillna(50)

    # Rata-rata volume 20 hari
    df["AVG_VOLUME20"] = df["Volume"].rolling(
        window=20
    ).mean()

    df["VOLUME_RATIO"] = (
        df["Volume"] /
        df["AVG_VOLUME20"].replace(0, np.nan)
    )

    # Harga tertinggi 20 hari sebelumnya
    df["HIGH20_SEBELUMNYA"] = (
        df["High"].shift(1).rolling(window=20).max()
    )

    # Perubahan harga harian
    df["PERUBAHAN_PERSEN"] = (
        df["Close"].pct_change() * 100
    )

    return df


# ==========================================
# FUNGSI ANALISIS SAHAM
# ==========================================

def analisis_saham(ticker):
    data = ambil_data(ticker)

    if data is None:
        return None

    df = hitung_indikator(data)

    if len(df) < 25:
        return None

    terakhir = df.iloc[-1]

    harga = float(terakhir["Close"])
    ema9 = float(terakhir["EMA9"])
    ema21 = float(terakhir["EMA21"])
    rsi = float(terakhir["RSI14"])

    volume = float(terakhir["Volume"])

    rasio_volume = terakhir["VOLUME_RATIO"]

    if pd.isna(rasio_volume):
        rasio_volume = 0
    else:
        rasio_volume = float(rasio_volume)

    high20 = terakhir["HIGH20_SEBELUMNYA"]

    if pd.isna(high20):
        high20 = 0
    else:
        high20 = float(high20)

    perubahan = terakhir["PERUBAHAN_PERSEN"]

    if pd.isna(perubahan):
        perubahan = 0
    else:
        perubahan = float(perubahan)

    # Harga tertinggi dan terendah 20 hari
    high20_aktual = float(df["High"].tail(20).max())
    low20_aktual = float(df["Low"].tail(20).min())

    # ======================================
    # PENILAIAN SKOR
    # ======================================

    skor = 0
    alasan = []

    # Tren EMA
    if harga > ema9:
        skor += 1
        alasan.append("Harga di atas EMA9")

    if ema9 > ema21:
        skor += 2
        alasan.append("EMA9 di atas EMA21")

    # RSI
    if 50 <= rsi <= 65:
        skor += 2
        alasan.append("RSI mendukung momentum")

    elif 40 <= rsi < 50:
        skor += 1
        alasan.append("RSI mulai pulih")

    elif rsi > 70:
        skor -= 1
        alasan.append("RSI tinggi, waspadai koreksi")

    elif rsi < 30:
        alasan.append("RSI rendah, tekanan jual tinggi")

    # Volume
    if rasio_volume >= 1.5:
        skor += 2
        alasan.append("Volume meningkat")

    elif rasio_volume >= 1.2:
        skor += 1
        alasan.append("Volume di atas rata-rata")

    # Breakout
    breakout = (
        high20 > 0 and harga > high20
    )

    if breakout:
        skor += 3
        alasan.append("Breakout harga tertinggi 20 hari")

    # ======================================
    # MENENTUKAN SINYAL
    # ======================================

    if breakout and ema9 > ema21 and rsi < 75:
        sinyal = "BREAKOUT POTENSIAL"

    elif skor >= 4 and harga > ema21:
        sinyal = "PANTAU"

    else:
        sinyal = "BELUM ADA SINYAL"

    # ======================================
    # SIMULASI TARGET DAN STOP LOSS
    # ======================================

    stop_loss = harga * 0.97
    target_1 = harga * 1.05
    target_2 = harga * 1.08

    # ======================================
    # PERKIRAAN LOT DENGAN MODAL
    # ======================================

    modal_lot = harga * 100

    return {
        "Kode Saham": ticker.replace(".JK", ""),
        "Harga": round(harga, 2),
        "Perubahan (%)": round(perubahan, 2),
        "RSI": round(rsi, 2),
        "EMA9": round(ema9, 2),
        "EMA21": round(ema21, 2),
        "Volume": int(volume),
        "Rasio Volume": round(rasio_volume, 2),
        "High 20 Hari": round(high20_aktual, 2),
        "Low 20 Hari": round(low20_aktual, 2),
        "Skor": int(skor),
        "Sinyal": sinyal,
        "Stop Loss (-3%)": round(stop_loss, 2),
        "Target (+5%)": round(target_1, 2),
        "Target (+8%)": round(target_2, 2),
        "Modal 1 Lot": round(modal_lot, 0),
        "Alasan": ", ".join(alasan)
    }


# ==========================================
# PENGATURAN PEMINDAIAN
# ==========================================

st.sidebar.header("⚙️ Pengaturan")

daftar_saham = st.sidebar.text_area(
    "Daftar kode saham (pisahkan dengan koma)",
    value=",".join(
        ticker.replace(".JK", "")
        for ticker in DEFAULT_TICKERS
    ),
    height=180
)

modal = st.sidebar.number_input(
    "Modal simulasi (Rp)",
    min_value=100000,
    value=1000000,
    step=100000
)

jumlah_top = st.sidebar.slider(
    "Jumlah saham terbaik",
    min_value=5,
    max_value=30,
    value=10
)

st.sidebar.caption(
    "Data berasal dari Yahoo Finance. "
    "Ketersediaan dan waktu pembaruan data dapat berbeda."
)

# ==========================================
# TOMBOL PEMINDAIAN
# ==========================================

tombol_scan = st.button(
    "🔍 MULAI SCAN SAHAM",
    type="primary",
    use_container_width=True
)

if tombol_scan:

    # Membersihkan daftar saham
    tickers_input = [
        kode.strip().upper()
        for kode in daftar_saham.split(",")
        if kode.strip()
    ]

    tickers = []

    for kode in tickers_input:
        if not kode.endswith(".JK"):
            kode = kode + ".JK"

        if kode not in tickers:
            tickers.append(kode)

    if not tickers:
        st.error("Masukkan minimal satu kode saham.")
        st.stop()

    hasil = []
    gagal = []

    progress = st.progress(0)
    status = st.empty()

    # ======================================
    # PROSES SCAN
    # ======================================

    for i, ticker in enumerate(tickers):

        status.text(
            f"Memindai {ticker} "
            f"({i + 1}/{len(tickers)})..."
        )

        try:
            hasil_saham = analisis_saham(ticker)

            if hasil_saham is not None:
                hasil.append(hasil_saham)
            else:
                gagal.append(ticker)

        except Exception:
            gagal.append(ticker)

        progress.progress((i + 1) / len(tickers))

    progress.empty()
    status.empty()

    # ======================================
    # HASIL SCAN
    # ======================================

    if not hasil:
        st.error(
            "Data saham tidak berhasil diperoleh. "
            "Coba lagi nanti atau periksa koneksi internet."
        )
        st.stop()

    df_hasil = pd.DataFrame(hasil)

    df_hasil = df_hasil.sort_values(
        by=["Skor", "Rasio Volume"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Menyimpan hasil dalam session
    st.session_state["hasil_scan"] = df_hasil
    st.session_state["gagal_scan"] = gagal
    st.session_state["modal_scan"] = modal
    st.session_state["jumlah_top"] = jumlah_top


# ==========================================
# MENAMPILKAN HASIL YANG TERSIMPAN
# ==========================================

if "hasil_scan" in st.session_state:

    df_hasil = st.session_state["hasil_scan"]
    gagal = st.session_state.get("gagal_scan", [])
    modal = st.session_state.get("modal_scan", 1000000)
    jumlah_top = st.session_state.get("jumlah_top", 10)

    st.success(
        f"Pemindaian selesai. "
        f"Ditemukan {len(df_hasil)} saham."
    )

    st.caption(
        "Waktu hasil ditampilkan: "
        + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

    # ======================================
    # RINGKASAN
    # ======================================

    total_saham = len(df_hasil)

    total_breakout = len(
        df_hasil[
            df_hasil["Sinyal"] == "BREAKOUT POTENSIAL"
        ]
    )

    total_pantau = len(
        df_hasil[
            df_hasil["Sinyal"] == "PANTAU"
        ]
    )

    total_volume = len(
        df_hasil[
            df_hasil["Rasio Volume"] >= 1.5
        ]
    )

    kolom1, kolom2, kolom3, kolom4 = st.columns(4)

    kolom1.metric(
        "Saham Terpindai",
        total_saham
    )

    kolom2.metric(
        "Breakout Potensial",
        total_breakout
    )

    kolom3.metric(
        "Saham Dipantau",
        total_pantau
    )

    kolom4.metric(
        "Volume Meningkat",
        total_volume
    )

    st.divider()

    # ======================================
    # TOP SAHAM
    # ======================================

    st.subheader("🏆 TOP SAHAM BERDASARKAN SKOR")

    top_saham = df_hasil.head(jumlah_top).copy()

    st.dataframe(
        top_saham[
            [
                "Kode Saham",
                "Harga",
                "Perubahan (%)",
                "RSI",
                "Rasio Volume",
                "Skor",
                "Sinyal"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # BREAKOUT POTENSIAL
    # ======================================

    st.subheader("🚀 DAFTAR BREAKOUT POTENSIAL")

    breakout_df = df_hasil[
        df_hasil["Sinyal"] == "BREAKOUT POTENSIAL"
    ]

    if not breakout_df.empty:

        st.dataframe(
            breakout_df[
                [
                    "Kode Saham",
                    "Harga",
                    "High 20 Hari",
                    "RSI",
                    "Rasio Volume",
                    "Stop Loss (-3%)",
                    "Target (+5%)",
                    "Target (+8%)",
                    "Skor"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Belum ditemukan saham dengan kriteria "
            "breakout potensial."
        )

    # ======================================
    # VOLUME MENINGKAT
    # ======================================

    st.subheader("📊 SAHAM DENGAN LONJAKAN VOLUME")

    volume_df = df_hasil[
        df_hasil["Rasio Volume"] >= 1.5
    ].sort_values(
        by="Rasio Volume",
        ascending=False
    )

    if not volume_df.empty:

        st.dataframe(
            volume_df[
                [
                    "Kode Saham",
                    "Harga",
                    "Perubahan (%)",
                    "Volume",
                    "Rasio Volume",
                    "Skor",
                    "Sinyal"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Belum ada saham dengan volume "
            "minimal 1,5 kali rata-rata."
        )

    # ======================================
    # SIMULASI MODAL
    # ======================================

    st.subheader("💰 SIMULASI PEMBELIAN SAHAM")

    pilihan_saham = st.selectbox(
        "Pilih saham untuk simulasi",
        options=df_hasil["Kode Saham"].tolist()
    )

    data_pilihan = df_hasil[
        df_hasil["Kode Saham"] == pilihan_saham
    ].iloc[0]

    harga_pilihan = float(data_pilihan["Harga"])

    biaya_per_lot = harga_pilihan * 100

    lot_maksimal = int(modal // biaya_per_lot)

    total_biaya = lot_maksimal * biaya_per_lot

    sisa_modal = modal - total_biaya

    kolom1, kolom2, kolom3 = st.columns(3)

    kolom1.metric(
        "Modal",
        f"Rp{modal:,.0f}"
    )

    kolom2.metric(
        "Perkiraan Lot",
        f"{lot_maksimal} lot"
    )

    kolom3.metric(
        "Sisa Modal",
        f"Rp{sisa_modal:,.0f}"
    )

    st.caption(
        "Simulasi belum memperhitungkan biaya broker, "
        "pajak, dan perubahan harga. Pembelian 1 lot "
        "saham BEI umumnya setara dengan 100 lembar."
    )

    # ======================================
    # RINCIAN SAHAM
    # ======================================

    st.subheader("🔎 RINCIAN ANALISIS SAHAM")

    rincian = st.selectbox(
        "Pilih saham untuk melihat rincian",
        options=df_hasil["Kode Saham"].tolist(),
        key="rincian_saham"
    )

    baris = df_hasil[
        df_hasil["Kode Saham"] == rincian
    ].iloc[0]

    kolom1, kolom2, kolom3 = st.columns(3)

    kolom1.metric(
        "Harga Terakhir",
        f"Rp{baris['Harga']:,.2f}"
    )

    kolom2.metric(
        "RSI 14",
        f"{baris['RSI']:.2f}"
    )

    kolom3.metric(
        "Skor Analisis",
        int(baris["Skor"])
    )

    st.write(
        "**Sinyal:**",
        baris["Sinyal"]
    )

    st.write(
        "**EMA 9:**",
        f"Rp{baris['EMA9']:,.2f}"
    )

    st.write(
        "**EMA 21:**",
        f"Rp{baris['EMA21']:,.2f}"
    )

    st.write(
        "**Rasio volume:**",
        f"{baris['Rasio Volume']:.2f} kali"
    )

    st.write(
        "**Simulasi stop loss (-3%):**",
        f"Rp{baris['Stop Loss (-3%)']:,.2f}"
    )

    st.write(
        "**Simulasi target (+5%):**",
        f"Rp{baris['Target (+5%)']:,.2f}"
    )

    st.write(
        "**Simulasi target (+8%):**",
        f"Rp{baris['Target (+8%)']:,.2f}"
    )

    st.write(
        "**Alasan penilaian:**",
        baris["Alasan"] if baris["Alasan"] else "-"
    )

    # ======================================
    # SEMUA HASIL
    # ======================================

    st.subheader("📋 SEMUA HASIL PEMINDAIAN")

    filter_sinyal = st.selectbox(
        "Filter berdasarkan sinyal",
        [
            "SEMUA",
            "BREAKOUT POTENSIAL",
            "PANTAU",
            "BELUM ADA SINYAL"
        ]
    )

    if filter_sinyal == "SEMUA":
        hasil_filter = df_hasil.copy()
    else:
        hasil_filter = df_hasil[
            df_hasil["Sinyal"] == filter_sinyal
        ].copy()

    st.dataframe(
        hasil_filter,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # DOWNLOAD CSV
    # ======================================

    csv_data = df_hasil.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Semua Hasil (CSV)",
        data=csv_data,
        file_name=(
            "hasil_scan_saham_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    # ======================================
    # SAHAM YANG GAGAL DIPINDAI
    # ======================================

    if gagal:

        with st.expander(
            "Lihat saham yang gagal memperoleh data"
        ):
            st.write(", ".join(gagal))

    # ======================================
    # PERINGATAN
    # ======================================

    st.warning(
        "PERINGATAN RISIKO: Sinyal ini dihitung dari "
        "indikator teknikal sederhana dan bukan prediksi "
        "pasti. Data Yahoo Finance dapat terlambat, "
        "tidak lengkap, atau gagal diperbarui. "
        "Target harga dan stop loss hanya simulasi, "
        "bukan jaminan hasil transaksi. "
        "Periksa kembali harga dan kondisi saham "
        "sebelum mengambil keputusan."
    )

else:

    st.info(
        "Tekan tombol **MULAI SCAN SAHAM** untuk "
        "memulai pemindaian saham."
    )

    st.write(
        "Bot ini memindai daftar saham BEI menggunakan "
        "EMA9, EMA21, RSI14, rasio volume, dan breakout "
        "harga tertinggi 20 hari."
    )

st.divider()

st.caption(
    "Sumber data: Yahoo Finance melalui yfinance. "
    "Pemindaian menggunakan data harian, bukan data "
    "real-time intraday. Data dapat terlambat atau "
    "tidak tersedia. Target dan stop hanya simulasi, "
    "bukan jaminan hasil transaksi."
)
