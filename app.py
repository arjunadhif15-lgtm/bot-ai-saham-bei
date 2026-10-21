import os
import time
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BOT AI SAHAM BEI 6.0",
    page_icon="📈",
    layout="wide"
)

VERSION = "6.0"

# ==========================================================
# DAFTAR TICKER BAWAAN
# Daftar ini bukan daftar resmi BEI yang dijamin lengkap.
# Tambahkan ticker melalui CSV atau input manual.
# ==========================================================

DEFAULT_TICKERS = """
AADI ABMM ACES ACST ADES ADHI ADMF ADMR ADRO AGAR AGII AGRO AGRS
AKRA AMAG AMAR AMMN AMRT ANJT ANTM ARCI ARII ARNA ARTO ASGR ASII
ASLC ASRI ASSA AUTO AVIA BABP BACA BAJA BANK BAPA BATA BAYU BBCA
BBHI BBKP BBMD BBNI BBRI BBSI BBTN BBYB BCAP BCIC BDMN BEBS BEEF
BEKS BELI BFIN BHAT BHIT BIKE BINA BIRD BISI BJBR BJTM BKSL BMAS
BMRI BNBA BNGA BNII BNLI BOBA BOGA BOLT BRAM BRIS BRMS BRNA BRPT
BRRC BSDE BSIM BSSR BTON BTPS BUDI BUKA BULL BUMI BUVA BWPT
BYAN CAMP CARS CASA CASS CEKA CENT CFIN CGAS CHEM CINT CITA CITY
CMNP CMNT CMPP CMRY CNKO COAL COCO CPIN CPRO CSAP CSIS CTRA CYBR
DAJK DATA DAYA DCII DEWA DGIK DGNS DGWG DILD DIVA DKFT DLTA DMAS
DMND DNAR DOID DPNS DPUM DRMA DSNG DSSA DUTI DVLA DWGL EAST ECII
EDGE EKAD ELIT ELPI ELSA EMTK ENAK ENRG ESSA ESTI EURO EXCL FAST
FASW FILM FIRE FISH FOLK FORU FPNI GDST GDYR GEMA GEMS GGRM GGRP
GIAA GJTL GLVA GOLD GOOD GPRA GTRA GUNA GWSA GZCO HAIS HALO HDFA
HEAL HERO HEXA HGII HMSP HOME HRTA HRUM IATA IBOS IDEA IFII IGAR
IKAI IKAN IMAS IMPC INAF INAI INCI INCO INDF INDR INDS INDY INET
INKP INOV INPC INPP INRU INTA INTD INTG IPCC IPOL IPPE IPTV IRRA
ISAT ISSP ITIC ITMG JAST JATI JECC JGLE JKON JMAS JPFA JRPT JSMR
JTPE KAEF KARW KBLI KBLM KDSI KEEN KEJU KETR KIAS KIJA KINO KIOS
KKGI KLBF KMDS KMTR KOBX KPAL KPAS KPIG KREN KRYA LABA LAJU LAND
LAPD LCKM LEAD LINK LMAS LMPI LPPF LPCK LPKR LRNA LSIP LTLS LUCY
MABA MAIN MAPA MAPB MAPI MARK MASA MAYA MBAP MBSS MBTO MCAS MCOL
MDIA MDKA MDLN MDRN MEDC MEGA MERK META MFMI MGNA MGRO MICE MIDI
MIKA MINA MINE MIRA MITI MKNT MLBI MLIA MLPL MMLP MNCN MORA MPPA
MPOW MPRO MRAT MSIN MSKY MTDL MTEL MTLA MTMH MTPS MTRA MTWI MYOH
MYOR MYTX NANO NASA NATO NCKL NELY NETV NFCX NICE NICL NIKL NINE
NIPS NIRO NISP NOBU NRCA NTBK NUSA OBMD OILS OKAS OMRE OPMS PADA
PADI PALM PAMG PANS PBID PBRX PBSA PCAR PDES PEGE PEHA PGAS PGEO
PGLI PJAA PKPK PLAN PLIN PMJS PMMP PMUI PNBN PNBS PNIN PNLF PNSE
POLA POLI POLL PORT POWR PPRE PPRO PRAY PSAB PSDN PSKT PSSI PTBA
PTIS PTMP PTPP PTRO PTSN PUDP PURA PURE PURI PWON PYFA RAJA RALS
RANC RBMS RDTX REAL RELI RICY RIGS RISE RMBA RMKE ROCK RODA ROTI
RUCI RUNS SAFE SAME SAMF SAPX SATU SBAT SBMA SCCO SCMA SCNP SDMU
SDPC SDRA SGRO SHID SHIP SIDO SIMP SINI SIPD SKBM SKLT SKRN SLIS
SMAR SMBR SMCB SMDM SMDR SMGR SMIL SMKL SMKM SMMT SMRA SMSM SOCI
SOFA SONA SOSS SPMA SPTO SRAJ SRTG SSIA SSMS SSTM STAR STRK SUGI
SULI SUPA SUPR SURE SWAT SWID TAMA TARA TBIG TBLA TBMS TCID TCPI
TDPM TFAS TGKA TGRA TIFA TINS TKIM TLKM TMAS TMPO TOBA TOPS TOTL
TOTO TOWR TPIA TPMA TRGU TRIM TRIN TRIS TRJA TRST TSPC TUGU TURI
ULTJ UNIQ UNTR UNVR VAST VICI VINS VIVA VKTR WAPO WEGE WEHA WIFI
WIIM WIKA WINS WIRG WMPP WOMF WOOD WTON ZATA ZBRA ZINC ZONE
""".split()


# ==========================================================
# FUNGSI DASAR
# ==========================================================

def clean_ticker(value):
    value = str(value).strip().upper()

    if value.endswith(".JK"):
        value = value[:-3]

    return "".join(
        char for char in value
        if char.isalnum() or char == "."
    )


def fmt_rp(value):
    try:
        if pd.isna(value):
            return "-"
        return "Rp{:,.0f}".format(
            float(value)
        ).replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def calc_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    return rsi


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker, period="1y"):
    symbol = (
        ticker if ticker.endswith(".JK")
        else ticker + ".JK"
    )

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(col).title()
        for col in data.columns
    ]

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()
    data = data.dropna(subset=["Close"])

    return data


def csv_bytes(df):
    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


def get_tickers(uploaded_file, manual_text, use_default, max_tickers):
    tickers = DEFAULT_TICKERS.copy() if use_default else []

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)

            csv = pd.read_csv(uploaded_file)

            if csv.empty:
                uploaded_file.seek(0)
                csv = pd.read_csv(
                    uploaded_file,
                    header=None
                )

            column = None

            for col in csv.columns:
                if any(
                    word in str(col).lower()
                    for word in [
                        "ticker", "symbol",
                        "kode", "saham"
                    ]
                ):
                    column = col
                    break

            if column is None:
                column = csv.columns[0]

            tickers.extend(
                clean_ticker(value)
                for value in csv[column].dropna().tolist()
            )

        except Exception as exc:
            st.warning(f"CSV gagal dibaca: {exc}")

    if manual_text.strip():
        text = (
            manual_text.replace(",", " ")
            .replace(";", " ")
            .replace("\n", " ")
        )

        tickers.extend(
            clean_ticker(value)
            for value in text.split()
        )

    result = []
    seen = set()

    for ticker in tickers:
        if ticker and ticker not in seen:
            seen.add(ticker)
            result.append(ticker)

    return result[:max_tickers]


# ==========================================================
# ANALISIS TEKNIKAL
# ==========================================================

def analyze_stock(ticker, mode):
    # Catatan:
    # Kedua mode menggunakan candle harian.
    period = "6mo" if mode == "Trading Harian" else "1y"

    df = fetch_history(ticker, period)

    if df.empty or len(df) < 55:
        return None, "Data kosong atau kurang dari 55 candle"

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["RSI14"] = calc_rsi(close, 14)

    df["VolAvg20"] = volume.rolling(20).mean()

    df["VolRatio"] = (
        volume /
        df["VolAvg20"].replace(0, np.nan)
    )

    # High sebelumnya, tidak termasuk candle terakhir.
    df["PrevHigh10"] = high.shift(1).rolling(10).max()
    df["PrevHigh20"] = high.shift(1).rolling(20).max()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    last = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(last["Close"])

    if not np.isfinite(price) or price <= 0:
        return None, "Harga tidak valid"

    previous_price = float(previous["Close"])

    change_pct = (
        (price / previous_price - 1) * 100
        if previous_price > 0 else 0
    )

    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    ema50 = float(last["EMA50"])

    rsi = (
        float(last["RSI14"])
        if pd.notna(last["RSI14"])
        else 50
    )

    vol_ratio = (
        float(last["VolRatio"])
        if pd.notna(last["VolRatio"])
        else 0
    )

    atr = (
        float(last["ATR14"])
        if pd.notna(last["ATR14"])
        else price * 0.03
    )

    atr = max(atr, price * 0.005)

    avg_turnover = float(
        (close * volume).tail(20).mean()
    )

    if avg_turnover < 1_000_000_000:
        return None, "Rata-rata nilai transaksi 20 hari di bawah Rp1 miliar"
        
    today_turnover = price * float(last["Volume"])

    high10 = last["PrevHigh10"]
    high20 = last["PrevHigh20"]

    breakout10 = (
        pd.notna(high10) and price > float(high10)
    )

    breakout20 = (
        pd.notna(high20) and price > float(high20)
    )

    breakout = (
        breakout10
        if mode == "Trading Harian"
        else breakout20
    )

    # ------------------------------------------------------
    # A. SKOR TEKNIKAL
    # ------------------------------------------------------

    score = 0
    notes = []

    if price > ema9 > ema21:
        score += 20
        notes.append("Tren pendek positif")

    elif price > ema21:
        score += 12
        notes.append("Harga di atas EMA21")

    if price > ema50:
        score += 10
        notes.append("Harga di atas EMA50")

    if 50 <= rsi <= 68:
        score += 15
        notes.append("Momentum RSI mendukung")

    elif 42 <= rsi < 50:
        score += 8

    elif rsi > 75:
        notes.append("RSI tinggi, waspadai koreksi")

    if vol_ratio >= 1.8:
        score += 20
        notes.append("Volume melonjak")

    elif vol_ratio >= 1.2:
        score += 12
        notes.append("Volume di atas rata-rata")

    if breakout20:
        score += 20
        notes.append("Breakout high 20 hari")

    elif breakout10:
        score += 12
        notes.append("Breakout high 10 hari")

    if avg_turnover >= 5_000_000_000:
        score += 15
        notes.append("Nilai transaksi rata-rata tinggi")

    elif avg_turnover >= 1_000_000_000:
        score += 10
        notes.append("Nilai transaksi rata-rata cukup")

    elif avg_turnover >= 250_000_000:
        score += 5

    score = min(score, 100)

    # ------------------------------------------------------
    # B. PROXY AKUMULASI HARGA-VOLUME
    # Bukan bukti aktivitas bandar.
    # ------------------------------------------------------

    change5 = (
        (price / float(close.iloc[-6]) - 1) * 100
        if float(close.iloc[-6]) > 0
        else 0
    )

    accumulation_score = 0
    accumulation_notes = []

    if vol_ratio >= 1.3 and -2 <= change_pct <= 2:
        accumulation_score += 35
        accumulation_notes.append(
            "Volume meningkat saat harga relatif datar"
        )

    if price > ema21 and change5 >= 0:
        accumulation_score += 25
        accumulation_notes.append(
            "Harga bertahan di atas EMA21"
        )

    if change5 > 0 and vol_ratio >= 1.2:
        accumulation_score += 25
        accumulation_notes.append(
            "Kenaikan harga disertai volume"
        )

    if change_pct < -3 and vol_ratio >= 1.5:
        accumulation_score -= 20
        accumulation_notes.append(
            "Tekanan jual besar perlu diwaspadai"
        )

    accumulation_score = max(
        0, min(100, accumulation_score)
    )

    # ------------------------------------------------------
    # C DAN D. ENTRY, STOP LOSS, TARGET
    # Estimasi berdasarkan ATR.
    # ------------------------------------------------------

    if mode == "Trading Harian":
        entry_low = max(0, price - 0.25 * atr)
        entry_high = price + 0.15 * atr

        stop = price - 1.0 * atr

        target1 = price + 1.0 * atr
        target2 = price + 1.5 * atr
        target3 = price + 2.0 * atr

    else:
        entry_low = max(0, price - 0.35 * atr)
        entry_high = price + 0.20 * atr

        stop = price - 1.5 * atr

        target1 = price + 1.5 * atr
        target2 = price + 2.5 * atr
        target3 = price + 3.5 * atr

    stop = max(0, stop)

    risk = max(entry_high - stop, 0.01)
    reward = max(target1 - entry_high, 0)

    risk_reward = reward / risk

    if score >= 70 and (
        breakout or (
            price > ema21 and vol_ratio >= 1.2
        )
    ):
        signal = "KANDIDAT"

    elif score >= 45:
        signal = "PANTAU"

    else:
        signal = "BELUM MEMENUHI"

    result = {
        "Kode": ticker,
        "Mode": mode,
        "Harga": price,
        "Perubahan %": change_pct,
        "Skor": int(score),
        "Sinyal": signal,
        "RSI14": rsi,
        "Vol Ratio": vol_ratio,
        "Breakout": "YA" if breakout else "Tidak",
        "EMA9": ema9,
        "EMA21": ema21,
        "EMA50": ema50,
        "Nilai Transaksi Rata2 20H": avg_turnover,
        "Nilai Transaksi Hari Ini": today_turnover,
        "Skor Akumulasi Proxy": accumulation_score,
        "Indikasi": (
            "; ".join(accumulation_notes)
            if accumulation_notes
            else "Belum ada pola kuat"
        ),
        "Entry Bawah": entry_low,
        "Entry Atas": entry_high,
        "Stop Loss": stop,
        "Target 1": target1,
        "Target 2": target2,
        "Target 3": target3,
        "Risk/Reward T1": risk_reward,
        "ATR14": atr,
        "Catatan": (
            "; ".join(notes)
            if notes
            else "Belum ada konfirmasi kuat"
        ),
        "Volume": float(last["Volume"]),
        "Tanggal Data": str(df.index[-1].date())
    }

    return (result, df), None


# ==========================================================
# A. GEMINI AI
# ==========================================================

def run_gemini(row, mode, model):
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY belum diatur pada "
            "Streamlit Cloud → Settings → Secrets."
        )

    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""
Anda adalah asisten analisis teknikal saham Indonesia.

Gunakan hanya data yang diberikan. Jangan mengarang berita,
laporan keuangan, data broker summary, atau order book.

Mode: {mode}
Data saham: {row.to_dict()}

Buat analisis dalam bahasa Indonesia sederhana dengan bagian:

1. Ringkasan kondisi saham.
2. Tren EMA9, EMA21, dan EMA50.
3. Momentum RSI.
4. Volume dan breakout.
5. Indikasi harga-volume dan keterbatasannya.
6. Skenario bullish, netral, dan bearish.
7. Area entry, stop loss, target 1, 2, dan 3.
8. Kondisi yang membatalkan skenario.
9. Risiko likuiditas, gap, slippage, dan biaya transaksi.
10. Kesimpulan edukatif: KANDIDAT, PANTAU, atau HINDARI DULU.

Target dan stop loss adalah estimasi, bukan kepastian.
Jangan menjamin keuntungan. Jika data kurang, nyatakan.
"""

    response = client.models.generate_content(
        model=model.strip() or "gemini-3.8-flash",
        contents=prompt
    )

    return response.text or "Gemini tidak mengembalikan teks."


# ==========================================================
# E. TELEGRAM
# ==========================================================

def send_telegram(token, chat_id, message):
    import requests

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(str(result))

    return True


# ==========================================================
# TAMPILAN APLIKASI
# ==========================================================

st.title("📈 BOT AI SAHAM BEI 6.0")

st.caption(
    "AI Gemini · Scanner · Harga-Volume · Sinyal Teknikal · "
    "Target Harga · Telegram · Money Management"
)

st.warning(
    "Aplikasi ini adalah alat bantu analisis, bukan jaminan "
    "keuntungan. Data Yahoo Finance dapat terlambat atau tidak "
    "lengkap. Skor akumulasi hanya proxy harga-volume, bukan "
    "bukti aktivitas bandar."
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("⚙️ Pengaturan")

    mode = st.radio(
        "Pilih strategi",
        ["Trading Harian", "Swing Trading"],
        index=1
    )

    max_tickers = st.slider(
        "Maksimum saham",
        min_value=20,
        max_value=531,
        value=100,
        step=10
    )

    use_default = st.checkbox(
        "Gunakan daftar ticker bawaan",
        value=True
    )

    uploaded_file = st.file_uploader(
        "Upload daftar saham CSV",
        type=["csv"]
    )

    manual_text = st.text_area(
        "Tambah ticker manual",
        placeholder="BBCA BBRI TLKM ANTM",
        height=85
    )

    st.divider()

    st.subheader("💰 Money Management")

    capital = st.number_input(
        "Modal (Rp)",
        min_value=100_000,
        value=3_000_000,
        step=100_000
    )

    risk_pct = st.slider(
        "Risiko maksimal (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )

    st.caption(
        "Simulasi menggunakan 1 lot = 100 saham. "
        "Biaya transaksi dan slippage belum dihitung."
    )

    st.divider()

    st.subheader("🤖 Gemini AI")

    gemini_model = st.text_input(
        "Nama model",
        value="gemini-3.8-flash"
    )

    st.divider()

    st.subheader("🔔 Telegram")

    telegram_token_default = st.secrets.get(
    "TELEGRAM_BOT_TOKEN", ""
)

telegram_chat_id_default = st.secrets.get(
    "TELEGRAM_CHAT_ID", ""
)

telegram_token = st.text_input(
    "Bot Token",
    value=telegram_token_default,
    type="password"
)

telegram_chat_id = st.text_input(
    "Chat ID",
    value=telegram_chat_id_default
)

telegram_min_score = st.slider(
    "Skor minimum notifikasi",
    min_value=60,
    max_value=100,
    value=75,
    step=5
)

tickers = get_tickers(
    uploaded_file,
    manual_text,
    use_default,
    max_tickers
)

st.info(
    f"Mode: **{mode}** | "
    f"Ticker yang akan dicoba: **{len(tickers)}**"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "results" not in st.session_state:
    st.session_state["results"] = []

if "failures" not in st.session_state:
    st.session_state["failures"] = []

if "frames" not in st.session_state:
    st.session_state["frames"] = {}

if "scan_time" not in st.session_state:
    st.session_state["scan_time"] = None

if "scan_mode" not in st.session_state:
    st.session_state["scan_mode"] = mode

if "last_ai_analysis" not in st.session_state:
    st.session_state["last_ai_analysis"] = ""

if "last_ai_code" not in st.session_state:
    st.session_state["last_ai_code"] = ""


# ==========================================================
# MULAI SCAN
# ==========================================================

col_scan, col_clear = st.columns([3, 1])

with col_scan:
    start_scan = st.button(
        "🔍 MULAI SCAN",
        type="primary",
        use_container_width=True
    )

with col_clear:
    clear_scan = st.button(
        "🧹 Hapus hasil",
        use_container_width=True
    )

if clear_scan:
    st.session_state["results"] = []
    st.session_state["failures"] = []
    st.session_state["frames"] = {}
    st.session_state["scan_time"] = None
    st.session_state["last_ai_analysis"] = ""
    st.rerun()


if start_scan:
    results = []
    failures = []
    frames = {}

    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(tickers):
        status.write(
            f"Memindai {i + 1}/{len(tickers)}: {ticker}"
        )

        try:
            output, error = analyze_stock(ticker, mode)

            if output is None:
                failures.append({
                    "Kode": ticker,
                    "Alasan": error
                })
            else:
                result, frame = output
                results.append(result)
                frames[ticker] = frame

        except Exception as exc:
            failures.append({
                "Kode": ticker,
                "Alasan": str(exc)[:180]
            })

        progress.progress(
            (i + 1) / max(len(tickers), 1)
        )

        time.sleep(0.02)

    st.session_state["results"] = results
    st.session_state["failures"] = failures
    st.session_state["frames"] = frames

    st.session_state["scan_time"] = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    st.session_state["scan_mode"] = mode
    st.session_state["last_ai_analysis"] = ""

    status.success(
        f"Selesai. Berhasil: {len(results)} | "
        f"Gagal/tidak tersedia: {len(failures)}"
    )

    progress.empty()


# ==========================================================
# TAMPILKAN HASIL
# ==========================================================

results = st.session_state["results"]
failures = st.session_state["failures"]

if results:

    all_df = pd.DataFrame(results)

    all_df = all_df.sort_values(
        ["Skor", "Skor Akumulasi Proxy"],
        ascending=False
    ).reset_index(drop=True)

    st.caption(
        f"Scan terakhir: {st.session_state['scan_time']} | "
        f"Mode saat scan: {st.session_state['scan_mode']}"
    )

    tabs = st.tabs([
        "📊 Dashboard",
        "🔎 Scanner",
        "🤖 AI Analisis",
        "🐋 Akumulasi",
        "🎯 Sinyal & Target",
        "💰 Modal",
        "🔔 Telegram",
        "⚠️ Gagal Scan"
    ])

    # ======================================================
    # DASHBOARD
    # ======================================================

    with tabs[0]:

        st.subheader("Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Berhasil dipindai",
            len(all_df)
        )

        c2.metric(
            "Kandidat",
            int((all_df["Sinyal"] == "KANDIDAT").sum())
        )

        c3.metric(
            "Breakout",
            int((all_df["Breakout"] == "YA").sum())
        )

        c4.metric(
            "Proxy akumulasi ≥ 60",
            int((all_df["Skor Akumulasi Proxy"] >= 60).sum())
        )

        st.markdown("### Saham dengan skor tertinggi")

        st.dataframe(
            all_df.head(15),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Unduh semua hasil CSV",
            data=csv_bytes(all_df),
            file_name="hasil_scan_bei_6.csv",
            mime="text/csv"
        )

    # ======================================================
    # F. SCANNER
    # ======================================================

    with tabs[1]:

        st.subheader("Scanner Saham")

        signal_filter = st.multiselect(
            "Filter sinyal",
            [
                "KANDIDAT",
                "PANTAU",
                "BELUM MEMENUHI"
            ],
            default=["KANDIDAT", "PANTAU"]
        )

        min_score = st.slider(
            "Skor minimum",
            min_value=0,
            max_value=100,
            value=40
        )

        only_breakout = st.checkbox(
            "Hanya saham breakout"
        )

        min_turnover = st.number_input(
            "Nilai transaksi rata-rata minimum (Rp)",
            min_value=0,
            value=0,
            step=250_000_000
        )

        filtered = all_df[
            all_df["Sinyal"].isin(signal_filter)
            & (all_df["Skor"] >= min_score)
            & (
                all_df["Nilai Transaksi Rata2 20H"]
                >= min_turnover
            )
        ].copy()

        if only_breakout:
            filtered = filtered[
                filtered["Breakout"] == "YA"
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Unduh hasil filter CSV",
            data=csv_bytes(filtered),
            file_name="hasil_filter_bei_6.csv",
            mime="text/csv"
        )

        st.markdown("### Detail saham")

        selected = st.selectbox(
            "Pilih saham",
            all_df["Kode"].tolist(),
            key="scanner_selected"
        )

        row = all_df[
            all_df["Kode"] == selected
        ].iloc[0]

        st.write(
            f"**Harga:** {fmt_rp(row['Harga'])}"
        )

        st.write(
            f"**Sinyal:** {row['Sinyal']} | "
            f"**Skor:** {row['Skor']}/100"
        )

        st.write(
            f"**Perubahan:** {row['Perubahan %']:.2f}%"
        )

        st.write(
            f"**Catatan:** {row['Catatan']}"
        )

        frame = st.session_state["frames"].get(selected)

        if frame is not None and not frame.empty:

            st.markdown("### Grafik harga dan EMA")

            st.line_chart(
                frame[
                    ["Close", "EMA9", "EMA21", "EMA50"]
                ].tail(90)
            )

            st.markdown("### Grafik volume")

            st.bar_chart(
                frame[["Volume"]].tail(30)
            )

    # ======================================================
    # A. ANALISIS GEMINI
    # ======================================================

    with tabs[2]:

        st.subheader("Analisis Saham dengan Gemini AI")

        ai_code = st.selectbox(
            "Pilih saham untuk dianalisis",
            all_df["Kode"].tolist(),
            key="ai_code"
        )

        ai_row = all_df[
            all_df["Kode"] == ai_code
        ].iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Harga",
            fmt_rp(ai_row["Harga"])
        )

        b.metric(
            "Skor",
            f"{int(ai_row['Skor'])}/100"
        )

        c.metric(
            "Sinyal",
            ai_row["Sinyal"]
        )

        st.write(
            f"RSI: {ai_row['RSI14']:.2f} | "
            f"Volume ratio: {ai_row['Vol Ratio']:.2f}"
        )

        if st.button(
            "🧠 Analisis dengan Gemini",
            key="run_ai"
        ):

            try:

                with st.spinner(
                    "Gemini sedang menganalisis..."
                ):

                    analysis = run_gemini(
                        ai_row,
                        st.session_state["scan_mode"],
                        gemini_model
                    )

                st.session_state["last_ai_analysis"] = analysis
                st.session_state["last_ai_code"] = ai_code

            except Exception as exc:

                st.error(
                    f"Analisis Gemini gagal: {exc}"
                )

        if st.session_state["last_ai_analysis"]:

            st.markdown(
                f"### Hasil analisis: "
                f"{st.session_state['last_ai_code']}"
            )

            st.markdown(
                st.session_state["last_ai_analysis"]
            )

    # ======================================================
    # B. AKUMULASI HARGA-VOLUME
    # ======================================================

    with tabs[3]:

        st.subheader(
            "Indikasi Akumulasi Harga-Volume"
        )

        st.warning(
            "Ini hanya indikator perkiraan berbasis harga dan "
            "volume harian. Bukan data broker summary, bukan "
            "order book, dan bukan bukti aktivitas bandar."
        )

        accum_df = all_df.sort_values(
            "Skor Akumulasi Proxy",
            ascending=False
        )

        st.dataframe(
            accum_df[
                [
                    "Kode",
                    "Harga",
                    "Perubahan %",
                    "Vol Ratio",
                    "Skor Akumulasi Proxy",
                    "Indikasi",
                    "Nilai Transaksi Rata2 20H"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Unduh data indikasi akumulasi",
            data=csv_bytes(accum_df),
            file_name="indikasi_akumulasi_bei.csv",
            mime="text/csv"
        )

    # ======================================================
    # C DAN D. SINYAL DAN TARGET
    # ======================================================

    with tabs[4]:

        st.subheader(
            "Sinyal Trading dan Target Harga"
        )

        target_code = st.selectbox(
            "Pilih saham",
            all_df["Kode"].tolist(),
            key="target_code"
        )

        target_row = all_df[
            all_df["Kode"] == target_code
        ].iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "Harga terakhir",
            fmt_rp(target_row["Harga"])
        )

        b.metric(
            "Sinyal",
            target_row["Sinyal"]
        )

        c.metric(
            "Skor",
            f"{int(target_row['Skor'])}/100"
        )

        st.write(
            f"**Entry indikatif:** "
            f"{fmt_rp(target_row['Entry Bawah'])} – "
            f"{fmt_rp(target_row['Entry Atas'])}"
        )

        st.write(
            f"**Stop loss:** "
            f"{fmt_rp(target_row['Stop Loss'])}"
        )

        st.write(
            f"**Target 1:** "
            f"{fmt_rp(target_row['Target 1'])}"
        )

        st.write(
            f"**Target 2:** "
            f"{fmt_rp(target_row['Target 2'])}"
        )

        st.write(
            f"**Target 3:** "
            f"{fmt_rp(target_row['Target 3'])}"
        )

        st.write(
            f"**Risk/reward T1:** "
            f"1:{target_row['Risk/Reward T1']:.2f}"
        )

        st.info(
            "Harga target adalah skenario berbasis ATR, bukan "
            "prediksi pasti. Periksa kembali harga berjalan, "
            "likuiditas, gap, dan biaya transaksi."
        )

    # ======================================================
    # MONEY MANAGEMENT
    # ======================================================

    with tabs[5]:

        st.subheader("Simulasi Modal dan Risiko")

        mm_code = st.selectbox(
            "Pilih saham",
            all_df["Kode"].tolist(),
            key="mm_code"
        )

        mm_row = all_df[
            all_df["Kode"] == mm_code
        ].iloc[0]

        entry = float(mm_row["Entry Atas"])
        stop = float(mm_row["Stop Loss"])

        risk_budget = capital * risk_pct / 100

        risk_per_share = max(
            entry - stop,
            0.01
        )

        shares_by_risk = math.floor(
            risk_budget / risk_per_share
        )

        shares_by_capital = (
            math.floor(capital / entry)
            if entry > 0 else 0
        )

        shares = max(
            0,
            min(shares_by_risk, shares_by_capital)
        )

        lots = shares // 100
        shares_used = lots * 100

        used_capital = shares_used * entry

        potential_loss = (
            shares_used * risk_per_share
        )

        potential_gain = (
            shares_used
            * max(float(mm_row["Target 1"]) - entry, 0)
        )

        a, b, c = st.columns(3)

        a.metric(
            "Jumlah lot",
            lots
        )

        b.metric(
            "Modal terpakai",
            fmt_rp(used_capital)
        )

        c.metric(
            "Potensi rugi ke stop",
            fmt_rp(potential_loss)
        )

        st.write(
            f"Batas risiko: **{fmt_rp(risk_budget)}**"
        )

        st.write(
            f"Potensi untung ke T1: "
            f"**{fmt_rp(potential_gain)}**"
        )

        if lots == 0:
            st.warning(
                "Dengan modal dan batas risiko ini, "
                "simulasi menghasilkan 0 lot. "
                "Jangan memaksakan transaksi."
            )

    # ======================================================
    # E. TELEGRAM
    # ======================================================

    with tabs[6]:

        st.subheader("Notifikasi Telegram")

        st.write(
            "Isi token bot dan Chat ID melalui sidebar. "
            "Token hanya digunakan ketika Anda menekan tombol kirim."
        )

        if st.button(
            "📨 Kirim sinyal ke Telegram",
            key="send_telegram"
        ):

            if not telegram_token or not telegram_chat_id:

                st.error(
                    "Isi Bot token dan Chat ID di sidebar."
                )

            else:

                eligible = all_df[
                    (all_df["Skor"] >= telegram_min_score)
                    & (all_df["Sinyal"] == "KANDIDAT")
                ].sort_values(
                    "Skor",
                    ascending=False
                ).head(15)

                if eligible.empty:

                    st.info(
                        "Tidak ada saham yang memenuhi syarat."
                    )

                else:

                    lines = [
                        f"📈 BOT AI SAHAM BEI {VERSION}",
                        f"Mode: {st.session_state['scan_mode']}",
                        f"Scan: {st.session_state['scan_time']}",
                        ""
                    ]

                    for _, r in eligible.iterrows():

                        lines.append(
                            f"{r['Kode']} | "
                            f"Skor {int(r['Skor'])} | "
                            f"{r['Sinyal']}\n"
                            f"Harga: {fmt_rp(r['Harga'])}\n"
                            f"Entry: {fmt_rp(r['Entry Bawah'])}"
                            f" - {fmt_rp(r['Entry Atas'])}\n"
                            f"SL: {fmt_rp(r['Stop Loss'])}\n"
                            f"T1: {fmt_rp(r['Target 1'])}\n"
                        )

                    message = "\n".join(lines)

                    # Telegram membatasi pesan teks.
                    message = message[:3900]

                    try:

                        send_telegram(
                            telegram_token.strip(),
                            telegram_chat_id.strip(),
                            message
                        )

                        st.success(
                            f"Notifikasi terkirim untuk "
                            f"{len(eligible)} saham."
                        )

                    except Exception as exc:

                        st.error(
                            f"Gagal mengirim Telegram: {exc}"
                        )

        st.caption(
            "Versi ini mengirim pesan saat tombol ditekan. "
            "Bot belum melakukan pemantauan otomatis di latar belakang."
        )

    # ======================================================
    # DAFTAR GAGAL
    # ======================================================

    with tabs[7]:

        st.subheader("Saham Gagal Dipindai")

        if failures:

            fail_df = pd.DataFrame(failures)

            st.dataframe(
                fail_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Unduh daftar saham gagal",
                data=csv_bytes(fail_df),
                file_name="saham_gagal_scan.csv",
                mime="text/csv"
            )

        else:

            st.success(
                "Tidak ada kegagalan yang tercatat."
            )

else:

    st.info(
        "Tekan MULAI SCAN untuk memindai saham. "
        "Untuk pengujian awal, gunakan 20–30 ticker."
    )

    st.markdown("""
    ### Fitur versi 6.0

    - A: Analisis saham dengan Gemini AI.
    - B: Indikasi akumulasi berdasarkan harga dan volume.
    - C: Sinyal teknikal.
    - D: Target harga dan stop loss berbasis ATR.
    - E: Notifikasi Telegram manual.
    - F: Scanner saham, filter likuiditas, CSV, dan laporan gagal.
    """)

st.divider()

st.caption(
    f"BOT AI SAHAM BEI {VERSION} | "
    "Data dapat terlambat atau tidak lengkap. "
    "Sinyal, target, dan stop loss bukan jaminan hasil. "
    "Mode Trading Harian masih memakai candle harian."
)
