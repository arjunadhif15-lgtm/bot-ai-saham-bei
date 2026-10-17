import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================================================
# BOT AI SAHAM BEI 4.0
# =========================================================

st.set_page_config(
    page_title="BOT AI SAHAM BEI 4.0",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BOT AI SAHAM BEI 4.0")
st.caption(
    "Scanner saham BEI dengan analisis teknikal, "
    "penyaringan likuiditas, dan bantuan Gemini AI."
)

# =========================================================
# DAFTAR SAHAM CADANGAN
# Bukan daftar lengkap seluruh saham BEI.
# =========================================================

DEFAULT_TICKERS = [
    "AADI", "ACES", "ADHI", "ADMR", "ADRO",
    "AGRO", "AKRA", "AMMN", "AMRT", "ANTM",
    "ARTO", "ASII", "ASSA", "AUTO", "AVIA",
    "BABP", "BANK", "BBCA", "BBNI", "BBRI",
    "BBTN", "BBYB", "BFIN", "BJBR", "BJTM",
    "BKSL", "BMRI", "BMTR", "BNGA", "BRIS",
    "BRMS", "BRPT", "BSDE", "BUKA", "BUMI",
    "BULL", "BWPT", "CDIA", "CLEO", "CMRY",
    "CNMA", "COCO", "CPIN", "CTRA", "CUAN",
    "DEWA", "DOID", "DSNG", "DSSA", "EMTK",
    "ENRG", "ERAA", "ESSA", "EXCL", "GGRM",
    "GJTL", "GOTO", "HEAL", "HMSP", "HRUM",
    "ICBP", "INCO", "INDF", "INDY", "INKP",
    "INTP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KIJA", "KLBF", "KPIG", "LPPF", "LSIP",
    "MBMA", "MDIY", "MDKA", "MEDC", "MIKA",
    "MIDI", "MNCN", "MPMX", "MTEL", "MYOR",
    "NCKL", "NICL", "NISP", "PANI", "PGAS",
    "PGEO", "PNBN", "PNLF", "PTBA", "PTPP",
    "PWON", "RAJA", "SCMA", "SIDO", "SMGR",
    "SMRA", "SRTG", "SSIA", "TAPG", "TBIG",
    "TINS", "TKIM", "TLKM", "TOWR", "TPIA",
    "TUGU", "ULTJ", "UNTR", "UNVR", "VKTR",
    "WIFI", "WOOD", "WSBP", "WSKT", "WTON",
    "FORE", "IOTF", "BRRC", "VIVA", "MDRN",
    "NTBK", "WEGE", "OKAS", "TAYS", "AGRS",
    "IBOS", "BVIC", "TRIN", "MINE", "NANO"
]


# =========================================================
# FUNGSI NORMALISASI KODE SAHAM
# =========================================================

def normalisasi_kode(value):
    kode = str(value).strip().upper()

    if kode.endswith(".JK"):
        kode = kode[:-3]

    if kode in {
        "",
        "NAN",
        "NONE",
        "KODE",
        "TICKER",
        "SYMBOL",
        "KODE SAHAM"
    }:
        return None

    if 1 <= len(kode) <= 6 and kode.isalnum():
        return kode

    return None


# =========================================================
# MEMBACA CSV DAFTAR SAHAM
# =========================================================

def baca_csv_kode(file):
    try:
        tabel = pd.read_csv(file)

        if tabel.empty:
            return []

        kolom_kode = None

        pilihan_kolom = [
            "kode",
            "kode saham",
            "ticker",
            "symbol",
            "code",
            "stock"
        ]

        for kolom in tabel.columns:
            if str(kolom).strip().lower() in pilihan_kolom:
                kolom_kode = kolom
                break

        if kolom_kode is None:
            kolom_kode = tabel.columns[0]

        hasil = []

        for nilai in tabel[kolom_kode]:
            kode = normalisasi_kode(nilai)

            if kode:
                hasil.append(kode)

        return hasil

    except Exception:
        return []


# =========================================================
# MENGAMBIL DATA HARGA DARI YAHOO FINANCE
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def ambil_data(kode):

    simbol = (
        kode if kode.endswith(".JK")
        else f"{kode}.JK"
    )

    try:
        df = yf.download(
            simbol,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        kolom_wajib = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            kolom in df.columns
            for kolom in kolom_wajib
        ):
            return pd.DataFrame()

        df = df[kolom_wajib].copy()

        for kolom in kolom_wajib:
            df[kolom] = pd.to_numeric(
                df[kolom],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        df = df[
            (df["Close"] > 0)
            & (df["Volume"] >= 0)
        ]

        if len(df) < 30:
            return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


# =========================================================
# MENGHITUNG INDIKATOR TEKNIKAL
# =========================================================

def hitung_indikator(df):

    d = df.copy()

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    # EMA
    d["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    d["EMA21"] = close.ewm(
        span=21,
        adjust=False
    ).mean()

    d["EMA50"] = close.ewm(
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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    # Penanganan kasus khusus RSI
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    d["RSI14"] = rsi.fillna(50)

    # Volume
    d["AVG_VOL20"] = volume.rolling(
        20
    ).mean()

    d["VOL_RATIO"] = (
        volume
        / d["AVG_VOL20"].replace(0, np.nan)
    )

    # Harga tertinggi 20 hari sebelumnya
    d["HIGH20_PREV"] = (
        high.shift(1)
        .rolling(20)
        .max()
    )

    # ATR 14
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR14"] = tr.rolling(14).mean()

    # Perubahan harga harian
    d["CHANGE_PCT"] = (
        close.pct_change() * 100
    )

    # Perkiraan nilai transaksi
    d["NILAI_TRANSAKSI"] = (
        close * volume
    )

    d["AVG_NILAI20"] = (
        d["NILAI_TRANSAKSI"]
        .rolling(20)
        .mean()
    )

    return d


# =========================================================
# ANALISIS DAN PENILAIAN SAHAM
# =========================================================

def analisis_saham(
    kode,
    df,
    batas_likuiditas
):

    d = hitung_indikator(df)

    r = d.iloc[-1]

    harga = float(r["Close"])

    ema9 = float(r["EMA9"])
    ema21 = float(r["EMA21"])
    ema50 = float(r["EMA50"])

    rsi = float(r["RSI14"])

    volume_ratio = (
        float(r["VOL_RATIO"])
        if pd.notna(r["VOL_RATIO"])
        else 0.0
    )

    atr = (
        float(r["ATR14"])
        if pd.notna(r["ATR14"])
        and r["ATR14"] > 0
        else harga * 0.03
    )

    high20 = (
        float(r["HIGH20_PREV"])
        if pd.notna(r["HIGH20_PREV"])
        else np.nan
    )

    avg_nilai = (
        float(r["AVG_NILAI20"])
        if pd.notna(r["AVG_NILAI20"])
        else 0.0
    )

    # Tren
    tren_naik = (
        harga > ema21
        and ema9 > ema21
        and ema21 > ema50
    )

    tren_menengah = (
        harga > ema21
        and ema9 > ema21
    )

    # Breakout
    breakout = (
        pd.notna(high20)
        and harga > high20
    )

    # Likuiditas
    likuid = (
        avg_nilai >= batas_likuiditas
    )

    # Skor teknikal, maksimum 100
    skor = 0

    if tren_naik:
        skor += 25
    elif tren_menengah:
        skor += 15

    if 50 <= rsi <= 68:
        skor += 15
    elif 45 <= rsi < 50:
        skor += 7

    if volume_ratio >= 1.5:
        skor += 20
    elif volume_ratio >= 1.0:
        skor += 10

    if breakout:
        skor += 20

    if likuid:
        skor += 20

    # Kategori
    if tren_naik:
        tren = "Naik"
    elif harga > ema50:
        tren = "Campuran"
    else:
        tren = "Turun"

    if skor >= 70 and likuid:
        kategori = "Kandidat potensial"
    elif skor >= 45:
        kategori = "Pantauan"
    else:
        kategori = "Belum memenuhi"

    # Level indikatif berbasis ATR
    area_beli_bawah = max(
        0,
        min(harga, ema21) - 0.25 * atr
    )

    area_beli_atas = (
        max(harga, ema9) + 0.15 * atr
    )

    stop_loss = max(
        0,
        harga - 1.5 * atr
    )

    target1 = harga + 2 * atr
    target2 = harga + 3 * atr

    perubahan = (
        float(r["CHANGE_PCT"])
        if pd.notna(r["CHANGE_PCT"])
        else 0.0
    )

    return {
        "Kode": kode,
        "Harga": round(harga, 2),
        "Perubahan %": round(perubahan, 2),
        "Skor": int(skor),
        "Kategori": kategori,
        "Tren": tren,
        "RSI": round(rsi, 1),
        "Rasio Volume": round(volume_ratio, 2),
        "Rata-rata Nilai Transaksi 20H (Rp)": int(
            avg_nilai
        ),
        "Likuid": "Ya" if likuid else "Tidak",
        "Breakout 20H": "Ya" if breakout else "Tidak",
        "Area Beli Bawah": round(
            area_beli_bawah, 2
        ),
        "Area Beli Atas": round(
            area_beli_atas, 2
        ),
        "Stop Loss Indikatif": round(
            stop_loss, 2
        ),
        "Target 1 Indikatif": round(
            target1, 2
        ),
        "Target 2 Indikatif": round(
            target2, 2
        ),
        "EMA9": round(ema9, 2),
        "EMA21": round(ema21, 2),
        "EMA50": round(ema50, 2)
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    st.subheader("Daftar saham")

    uploaded_csv = st.file_uploader(
        "Unggah CSV daftar saham BEI",
        type=["csv"],
        help=(
            "CSV dapat berisi kolom kode, kode saham, "
            "ticker, atau symbol."
        )
    )

    gabungkan_csv = st.checkbox(
        "Gabungkan daftar CSV dengan daftar cadangan",
        value=False
    )

    kode_manual = st.text_area(
        "Tambahkan kode saham manual",
        placeholder="Contoh: BBCA, BBRI, KPIG"
    )

    st.subheader("Modal")

    modal = st.number_input(
        "Modal simulasi (Rp)",
        min_value=100000,
        value=3000000,
        step=100000
    )

    st.subheader("Likuiditas")

    minimum_miliar = st.number_input(
        "Minimum rata-rata nilai transaksi harian (Rp miliar)",
        min_value=0.0,
        max_value=1000.0,
        value=1.0,
        step=0.5
    )

    batas_likuiditas = (
        minimum_miliar * 1_000_000_000
    )

    st.subheader("Batas pemindaian")

    jumlah_maks = st.number_input(
        "Maksimum kode saham",
        min_value=10,
        max_value=1000,
        value=150,
        step=10
    )

    mulai_scan = st.button(
        "🚀 MULAI SCAN",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Simpan GEMINI_API_KEY di Streamlit Cloud Secrets. "
        "Jangan menaruh API key langsung di kode."
    )


# =========================================================
# MENYIAPKAN DAFTAR PEMINDAIAN
# =========================================================

tickers = DEFAULT_TICKERS.copy()

if gabungkan_csv and uploaded_csv is not None:
    tickers.extend(
        baca_csv_kode(uploaded_csv)
    )

if kode_manual.strip():
    tickers.extend(
        [
            kode
            for kode in (
                normalisasi_kode(x)
                for x in kode_manual.split(",")
            )
            if kode
        ]
    )

tickers = list(dict.fromkeys(tickers))

jumlah_kode_asli = len(tickers)

tickers = tickers[:int(jumlah_maks)]

st.info(
    f"Jumlah kode yang akan dipindai: **{len(tickers)}**. "
    "Daftar bawaan bukan daftar lengkap seluruh emiten BEI. "
    "Gunakan CSV terbaru untuk memperluas cakupan."
)

if jumlah_kode_asli > len(tickers):
    st.warning(
        f"Ada {jumlah_kode_asli - len(tickers)} kode "
        "yang belum dipindai karena batas maksimum."
    )


# =========================================================
# SESSION STATE
# =========================================================

if "hasil_df" not in st.session_state:
    st.session_state["hasil_df"] = pd.DataFrame()

if "gagal" not in st.session_state:
    st.session_state["gagal"] = []

if "data_detail" not in st.session_state:
    st.session_state["data_detail"] = {}


# =========================================================
# PROSES PEMINDAIAN
# =========================================================

if mulai_scan:

    hasil = []
    gagal = []
    detail_data = {}

    progress = st.progress(0)
    status = st.empty()

    for i, kode in enumerate(
        tickers,
        start=1
    ):

        status.write(
            f"Memindai {kode} ({i}/{len(tickers)})..."
        )

        try:
            data = ambil_data(kode)

            if data.empty:

                gagal.append({
                    "Kode": kode,
                    "Alasan": (
                        "Data kosong atau tidak tersedia "
                        "di Yahoo Finance"
                    )
                })

            else:

                detail_data[kode] = data

                hasil.append(
                    analisis_saham(
                        kode,
                        data,
                        batas_likuiditas
                    )
                )

        except Exception as e:

            gagal.append({
                "Kode": kode,
                "Alasan": str(e)[:160]
            })

        progress.progress(
            i / max(1, len(tickers))
        )

        # Mengurangi permintaan beruntun
        time.sleep(0.1)

    hasil_df = pd.DataFrame(hasil)

    if not hasil_df.empty:

        hasil_df = hasil_df.sort_values(
            [
                "Skor",
                "Rata-rata Nilai Transaksi 20H (Rp)"
            ],
            ascending=[False, False]
        ).reset_index(drop=True)

    st.session_state["hasil_df"] = hasil_df
    st.session_state["gagal"] = gagal
    st.session_state["data_detail"] = detail_data

    status.success(
        f"Pemindaian selesai. "
        f"Berhasil: {len(hasil)} | "
        f"Gagal/tidak tersedia: {len(gagal)}"
    )


# =========================================================
# MENAMPILKAN HASIL
# =========================================================

hasil_df = st.session_state["hasil_df"]

if not hasil_df.empty:

    st.divider()
    st.subheader("🏆 Hasil Pemindaian")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Berhasil dipindai",
        len(hasil_df)
    )

    col2.metric(
        "Kandidat potensial",
        int(
            (
                hasil_df["Kategori"]
                == "Kandidat potensial"
            ).sum()
        )
    )

    col3.metric(
        "Saham memenuhi likuiditas",
        int(
            (
                hasil_df["Likuid"]
                == "Ya"
            ).sum()
        )
    )

    kategori_tersedia = (
        hasil_df["Kategori"]
        .dropna()
        .unique()
        .tolist()
    )

    filter_kategori = st.multiselect(
        "Filter kategori",
        kategori_tersedia,
        default=kategori_tersedia
    )

    hasil_tampil = hasil_df[
        hasil_df["Kategori"].isin(
            filter_kategori
        )
    ]

    st.dataframe(
        hasil_tampil,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download hasil pemindaian CSV",
        data=hasil_tampil.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="hasil_scan_saham_bei.csv",
        mime="text/csv"
    )

    # =====================================================
    # DETAIL SAHAM
    # =====================================================

    st.divider()
    st.subheader("🔎 Detail Saham")

    pilihan_saham = st.selectbox(
        "Pilih kode saham",
        hasil_df["Kode"].tolist()
    )

    row = hasil_df[
        hasil_df["Kode"] == pilihan_saham
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Harga terakhir",
        f"Rp {row['Harga']:,.0f}"
    )

    c2.metric(
        "Skor teknikal",
        f"{row['Skor']}/100"
    )

    c3.metric(
        "RSI",
        row["RSI"]
    )

    c4.metric(
        "Rasio volume",
        row["Rasio Volume"]
    )

    st.write(
        f"**Kategori:** {row['Kategori']}  \n"
        f"**Tren:** {row['Tren']}  \n"
        f"**Breakout 20 hari:** {row['Breakout 20H']}  \n"
        f"**Likuid menurut batas pilihan:** {row['Likuid']}"
    )

    st.write(
        f"**Area beli indikatif:** "
        f"Rp {row['Area Beli Bawah']:,.0f} "
        f"– Rp {row['Area Beli Atas']:,.0f}"
    )

    st.write(
        f"**Stop loss indikatif:** "
        f"Rp {row['Stop Loss Indikatif']:,.0f}"
    )

    st.write(
        f"**Target 1 indikatif:** "
        f"Rp {row['Target 1 Indikatif']:,.0f}"
    )

    st.write(
        f"**Target 2 indikatif:** "
        f"Rp {row['Target 2 Indikatif']:,.0f}"
    )

    # =====================================================
    # SIMULASI MODAL
    # =====================================================

    st.divider()
    st.subheader("💰 Simulasi Pembelian")

    harga_saham = float(row["Harga"])

    nilai_per_lot = harga_saham * 100

    if nilai_per_lot > 0:

        jumlah_lot = int(
            modal // nilai_per_lot
        )

        total_beli = (
            jumlah_lot * nilai_per_lot
        )

        sisa_modal = modal - total_beli

        sc1, sc2, sc3 = st.columns(3)

        sc1.metric(
            "Perkiraan lot",
            f"{jumlah_lot:,}"
        )

        sc2.metric(
            "Nilai pembelian",
            f"Rp {total_beli:,.0f}"
        )

        sc3.metric(
            "Sisa modal",
            f"Rp {sisa_modal:,.0f}"
        )

        st.caption(
            "Simulasi belum memperhitungkan biaya transaksi, "
            "slippage, atau perubahan harga."
        )

    # =====================================================
    # GRAFIK HARGA DAN EMA
    # =====================================================

    st.divider()
    st.subheader("📊 Grafik Harga dan EMA")

    data_detail = (
        st.session_state["data_detail"]
        .get(pilihan_saham)
    )

    if data_detail is not None and not data_detail.empty:

        grafik = hitung_indikator(
            data_detail
        )[
            [
                "Close",
                "EMA9",
                "EMA21",
                "EMA50"
            ]
        ].tail(90)

        st.line_chart(grafik)

        st.caption(
            "Grafik menunjukkan harga penutupan dan EMA "
            "hingga 90 sesi terakhir yang tersedia."
        )


# =========================================================
# DAFTAR SAHAM GAGAL
# =========================================================

gagal = st.session_state["gagal"]

if gagal:

    st.divider()

    with st.expander(
        f"⚠️ Saham gagal/tidak tersedia ({len(gagal)})"
    ):

        st.dataframe(
            pd.DataFrame(gagal),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download daftar saham gagal",
            data=pd.DataFrame(gagal).to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="saham_gagal_scan.csv",
            mime="text/csv"
        )


# =========================================================
# GEMINI AI
# =========================================================

st.divider()
st.subheader("🤖 Analisis Saham dengan Gemini AI")

st.caption(
    "Gemini membantu menafsirkan indikator. "
    "Hasil AI bukan jaminan keuntungan."
)

if not hasil_df.empty:

    kode_ai = st.selectbox(
        "Pilih saham untuk analisis AI",
        hasil_df["Kode"].tolist(),
        key="pilihan_ai"
    )

    if st.button("✨ Analisis dengan Gemini AI"):

        try:

            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

            from google import genai

            client = genai.Client(
                api_key=api_key
            )

            baris = hasil_df[
                hasil_df["Kode"] == kode_ai
            ].iloc[0].to_dict()

            prompt = f"""
Anda adalah asisten analisis saham Indonesia.

Analisis data teknikal berikut dalam bahasa Indonesia.

Kode saham: {kode_ai}

DATA:
{baris}

Bahas:
1. Kondisi tren berdasarkan EMA.
2. RSI dan momentum.
3. Rasio volume.
4. Likuiditas berdasarkan perkiraan nilai transaksi.
5. Apakah ada indikasi breakout 20 hari.
6. Risiko penurunan harga.
7. Skenario bullish, netral, dan bearish.
8. Hal yang perlu dikonfirmasi sebelum membeli.

Aturan:
- Jangan mengarang data fundamental atau berita.
- Jangan mengklaim mengetahui aktivitas bandar.
- Jangan menjanjikan keuntungan.
- Jelaskan bahwa indikator teknikal memiliki keterbatasan.
- Jika data tidak cukup, katakan secara terbuka.
- Berikan analisis yang seimbang, bukan perintah membeli atau menjual.
- Tegaskan bahwa ini bukan nasihat keuangan.
"""

            with st.spinner(
                "Gemini sedang menganalisis..."
            ):

                response = (
                    client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )
                )

            if response.text:

                st.markdown(
                    response.text
                )

            else:

                st.warning(
                    "Gemini tidak mengembalikan teks."
                )

        except KeyError:

            st.error(
                "GEMINI_API_KEY belum ditemukan. "
                "Buka Streamlit Cloud, masuk ke Settings, "
                "lalu Secrets."
            )

        except Exception as e:

            st.error(
                f"Analisis Gemini gagal: {e}"
            )

else:

    st.info(
        "Jalankan pemindaian terlebih dahulu "
        "untuk menggunakan Gemini AI."
    )


# =========================================================
# CATATAN
# =========================================================

st.divider()

st.caption(
    "BOT AI SAHAM BEI 4.0 | "
    "Sumber harga: Yahoo Finance. "
    "Nilai transaksi merupakan perkiraan harga × volume, "
    "bukan data order book atau bukti aktivitas bandar. "
    "Level beli, stop loss, dan target merupakan perkiraan "
    "teknikal, bukan kepastian. "
    "Verifikasi data sebelum mengambil keputusan investasi."
)
