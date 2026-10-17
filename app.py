Nama file: "app.py"

Salin seluruh kode pada blok berikut ke file "app.py" di GitHub.

import os
import time
import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BOT AI SAHAM BEI 5.0",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# BOT AI SAHAM BEI 5.0
# Scanner teknikal, breakout, harga-volume,
# money management, dan Gemini AI.
# ==================================================

DEFAULT_TICKERS = """
AADI ABMM ACES ACST ADES ADHI ADMF ADMR ADRO AGAR AGII AGRO AGRS
AKRA AMAG AMAR AMMN AMRT ANJT ANTM ARCI ARII ARNA ARTO ASGR ASII
ASLC ASRI ASSA AUTO AVIA BABP BACA BAJA BANK BAPA BATA BAYU BBCA
BBHI BBKP BBMD BBNI BBRI BBSI BBTN BBYB BCAP BCIC BDMN BEBS BEEF
BEKS BELI BFIN BHAT BHIT BIKE BINA BIRD BISI BJBR BJTM BKSL BMAS
BMRI BNBA BNGA BNII BNLI BOBA BOGA BOLT BRAM BRIS BRMS BRNA BRPT
BRRC BSDE BSIM BSSR BTEK BTON BTPS BUDI BUKA BULL BUMI BUVA BWPT
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
        return "Rp{:,.0f}".format(float(value)).replace(",", ".")
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

    # Jika tidak ada penurunan harga selama periode,
    # RSI secara praktis mendekati 100.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    return rsi


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker, period="1y", interval="1d"):
    symbol = (
        ticker if ticker.endswith(".JK")
        else ticker + ".JK"
    )

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).title()
        for column in data.columns
    ]

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if not all(column in data.columns for column in required):
        return pd.DataFrame()

    data = data[required].copy()

    data = data.dropna(subset=["Close"])

    return data


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

        except Exception as error:
            st.warning(
                "CSV tidak dapat dibaca: " + str(error)
            )

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


def analyze_stock(ticker, mode):
    period = "3mo" if mode == "Trading Harian" else "1y"

    df = fetch_history(
        ticker,
        period=period,
        interval="1d"
    )

    if df.empty or len(df) < 55:
        return None, "Data kosong atau kurang dari 55 hari"

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
        volume / df["VolAvg20"].replace(0, np.nan)
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

    if price <= 0:
        return None, "Harga tidak valid"

    prev_price = float(previous["Close"])

    change_pct = (
        (price / prev_price - 1) * 100
        if prev_price > 0 else 0
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

    today_turnover = price * float(last["Volume"])

    prev_high10 = float(last["PrevHigh10"])
    prev_high20 = float(last["PrevHigh20"])

    if not np.isfinite(prev_high10):
        prev_high10 = np.nan

    if not np.isfinite(prev_high20):
        prev_high20 = np.nan

    # ----------------------------------------------
    # Skor teknikal 0-100
    # ----------------------------------------------

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

    breakout10 = (
        pd.notna(prev_high10)
        and price > prev_high10
    )

    breakout20 = (
        pd.notna(prev_high20)
        and price > prev_high20
    )

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

    # ----------------------------------------------
    # Skor harga-volume sebagai indikasi, bukan
    # bukti pasti aktivitas bandar/institusi.
    # ----------------------------------------------

    accumulation_score = 0
    accumulation_notes = []

    change5 = (
        (price / float(close.iloc[-6]) - 1) * 100
        if float(close.iloc[-6]) > 0
        else 0
    )

    if vol_ratio >= 1.3 and -2 <= change_pct <= 2:
        accumulation_score += 35
        accumulation_notes.append(
            "Volume meningkat ketika harga relatif datar"
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

    # ----------------------------------------------
    # Estimasi entry, stop, dan target berbasis ATR
    # ----------------------------------------------

    if mode == "Trading Harian":
        entry_low = max(0, price - 0.25 * atr)
        entry_high = price + 0.15 * atr
        stop = price - 1.0 * atr
        target1 = price + 1.0 * atr
        target2 = price + 1.5 * atr
        target3 = price + 2.0 * atr
        breakout = breakout10

    else:
        entry_low = max(0, price - 0.35 * atr)
        entry_high = price + 0.20 * atr
        stop = price - 1.5 * atr
        target1 = price + 1.5 * atr
        target2 = price + 2.5 * atr
        target3 = price + 3.5 * atr
        breakout = breakout20

    # Jika volatilitas ekstrem menghasilkan stop
    # tidak masuk akal, harga tetap ditampilkan
    # dengan catatan untuk memeriksa grafik manual.
    stop = max(0, stop)

    risk = max(entry_high - stop, 0.01)
    reward = max(target1 - entry_high, 0)

    rr = reward / risk if risk > 0 else 0

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
        "Risk/Reward T1": rr,
        "ATR14": atr,
        "Catatan": (
            "; ".join(notes)
            if notes
            else "Belum ada konfirmasi kuat"
        )
    }

    return (result, df), None


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ==================================================
# TAMPILAN APLIKASI
# ==================================================

st.title("📈 BOT AI SAHAM BEI 5.0")

st.caption(
    "Trading Harian + Swing Trading | "
    "EMA | RSI | Volume | Breakout | Risk Management"
)

st.warning(
    "Bot ini merupakan alat bantu analisis, bukan jaminan "
    "keuntungan. Data Yahoo Finance dapat terlambat, tidak "
    "lengkap, atau gagal untuk ticker tertentu. Skor harga-volume "
    "bukan bukti pasti adanya aktivitas bandar."
)

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
        max_value=300,
        value=100,
        step=10
    )

    use_default = st.checkbox(
        "Gunakan daftar saham bawaan",
        value=True
    )

    uploaded_file = st.file_uploader(
        "Upload daftar saham CSV",
        type=["csv"]
    )

    manual_text = st.text_area(
        "Tambah ticker manual",
        placeholder="BBCA BBRI TLKM ANTM",
        height=90
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
        "Risiko maksimal per transaksi (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )

    st.caption(
        "Simulasi memakai 1 lot = 100 saham dan belum "
        "memperhitungkan biaya transaksi atau slippage."
    )

    st.divider()
    st.subheader("🤖 Gemini AI")

    gemini_model = st.text_input(
        "Nama model",
        value="gemini-2.5-flash"
    )

tickers = get_tickers(
    uploaded_file,
    manual_text,
    use_default,
    max_tickers
)

st.info(
    f"Mode **{mode}** · "
    f"Ticker yang dicoba: **{len(tickers)}**"
)

if st.button(
    "🔍 MULAI SCAN",
    type="primary",
    use_container_width=True
):
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

        time.sleep(0.03)

    st.session_state["results"] = results
    st.session_state["failures"] = failures
    st.session_state["frames"] = frames

    status.success(
        f"Selesai. Berhasil: {len(results)} | "
        f"Gagal/tidak tersedia: {len(failures)}"
    )

    progress.empty()


if "results" not in st.session_state:
    st.session_state["results"] = []

if "failures" not in st.session_state:
    st.session_state["failures"] = []

if "frames" not in st.session_state:
    st.session_state["frames"] = {}


# ==================================================
# HASIL SCANNER
# ==================================================

if st.session_state["results"]:
    df = pd.DataFrame(st.session_state["results"])

    df = df.sort_values(
        ["Skor", "Skor Akumulasi Proxy"],
        ascending=False
    ).reset_index(drop=True)

    st.subheader("📊 Hasil Scanner")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Berhasil dipindai", len(df))

    c2.metric(
        "Kandidat",
        int((df["Sinyal"] == "KANDIDAT").sum())
    )

    c3.metric(
        "Breakout",
        int((df["Breakout"] == "YA").sum())
    )

    c4.metric(
        "Skor akumulasi proxy ≥ 60",
        int((df["Skor Akumulasi Proxy"] >= 60).sum())
    )

    signal_filter = st.multiselect(
        "Filter sinyal",
        ["KANDIDAT", "PANTAU", "BELUM MEMENUHI"],
        default=["KANDIDAT", "PANTAU"]
    )

    min_score = st.slider(
        "Skor minimum",
        0,
        100,
        40
    )

    only_breakout = st.checkbox(
        "Hanya tampilkan saham breakout"
    )

    filtered = df[
        df["Sinyal"].isin(signal_filter)
        & (df["Skor"] >= min_score)
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
        "⬇️ Download hasil scan CSV",
        data=csv_bytes(filtered),
        file_name="hasil_scan_bei_5.csv",
        mime="text/csv"
    )

    st.subheader("🔎 Detail Saham")

    options = (
        filtered["Kode"].tolist()
        if not filtered.empty
        else df["Kode"].tolist()
    )

    selected = st.selectbox(
        "Pilih saham",
        options
    )

    row = df[df["Kode"] == selected].iloc[0]

    left, right = st.columns(2)

    with left:
        st.markdown("### Rencana transaksi")

        st.write(
            f"**Harga terakhir:** {fmt_rp(row['Harga'])}"
        )

        st.write(
            f"**Area entry:** "
            f"{fmt_rp(row['Entry Bawah'])} – "
            f"{fmt_rp(row['Entry Atas'])}"
        )

        st.write(
            f"**Stop loss:** {fmt_rp(row['Stop Loss'])}"
        )

        st.write(
            f"**Target 1:** {fmt_rp(row['Target 1'])}"
        )

        st.write(
            f"**Target 2:** {fmt_rp(row['Target 2'])}"
        )

        st.write(
            f"**Target 3:** {fmt_rp(row['Target 3'])}"
        )

        st.write(
            f"**Risk/reward:** 1:{row['Risk/Reward T1']:.2f}"
        )

        st.write(
            f"**Catatan teknikal:** {row['Catatan']}"
        )

    with right:
        st.markdown("### Simulasi modal")

        entry = float(row["Entry Atas"])
        stop = float(row["Stop Loss"])

        risk_budget = capital * risk_pct / 100

        risk_per_share = max(entry - stop, 0.01)

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
            * max(float(row["Target 1"]) - entry, 0)
        )

        st.write(
            f"**Modal tersedia:** {fmt_rp(capital)}"
        )

        st.write(
            f"**Batas risiko:** {fmt_rp(risk_budget)}"
        )

        st.write(
            f"**Estimasi jumlah:** {lots} lot"
        )

        st.write(
            f"**Modal terpakai:** {fmt_rp(used_capital)}"
        )

        st.write(
            f"**Potensi rugi:** {fmt_rp(potential_loss)}"
        )

        st.write(
            f"**Potensi untung T1:** {fmt_rp(potential_gain)}"
        )

        if lots == 0:
            st.info(
                "Dengan modal dan batas risiko ini, "
                "ukuran posisi yang sesuai adalah 0 lot. "
                "Jangan memaksakan transaksi."
            )

    st.subheader("📈 Grafik Harga")

    frame = st.session_state["frames"].get(selected)

    if frame is not None and not frame.empty:
        st.line_chart(
            frame[["Close", "EMA9", "EMA21", "EMA50"]].tail(90)
        )

        st.subheader("Volume")

        st.bar_chart(
            frame[["Volume"]].tail(30)
        )

    st.subheader("🏦 Indikasi Harga-Volume")

    st.write(
        f"**Skor indikasi:** "
        f"{int(row['Skor Akumulasi Proxy'])}/100"
    )

    st.write(row["Indikasi"])

    st.caption(
        "Skor harga-volume tidak mengidentifikasi broker atau "
        "pelaku pasar. Konfirmasi menggunakan data broker summary "
        "atau order book dari sumber yang sesuai jika tersedia."
    )

    # ==================================================
    # GEMINI AI
    # ==================================================

    st.subheader("🤖 Analisis Gemini AI")

    if st.button("Analisis saham terpilih dengan Gemini"):
        api_key = None

        try:
            api_key = st.secrets.get(
                "GEMINI_API_KEY",
                None
            )
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.error(
                "API key belum ditemukan. Buka Streamlit Cloud, "
                "Settings → Secrets, lalu tambahkan GEMINI_API_KEY. "
                "Jangan membagikan API key."
            )

        else:
            try:
                from google import genai

                client = genai.Client(
                    api_key=api_key
                )

                prompt = f"""
Analisis data saham Indonesia berikut untuk tujuan edukasi.

Mode: {mode}
Data: {row.to_dict()}

Bahas:
1. Tren dan momentum.
2. Kualitas breakout.
3. Makna volume dan keterbatasan data.
4. Area entry, stop loss, dan target sebagai skenario.
5. Risiko dan kondisi yang membatalkan skenario.
6. Hindari mengarang informasi yang tidak ada dalam data.
7. Jangan menjamin keuntungan atau menyebutnya sebagai
   rekomendasi pasti.

Gunakan bahasa Indonesia sederhana.
"""

                with st.spinner("Gemini sedang menganalisis..."):
                    response = client.models.generate_content(
                        model=gemini_model.strip() or "gemini-2.5-flash",
                        contents=prompt
                    )

                st.markdown(
                    response.text
                    or "Gemini tidak mengembalikan teks."
                )

            except Exception as exc:
                st.error(
                    f"Analisis Gemini gagal: {exc}"
                )

                st.info(
                    "Periksa nama model, API key, kuota, "
                    "dan koneksi internet."
                )

    with st.expander("Lihat seluruh data hasil scan"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# DAFTAR TICKER GAGAL
# ==================================================

if st.session_state["failures"]:
    with st.expander("⚠️ Saham gagal/tidak tersedia"):
        failed_df = pd.DataFrame(
            st.session_state["failures"]
        )

        st.dataframe(
            failed_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download daftar gagal CSV",
            data=csv_bytes(failed_df),
            file_name="saham_gagal_scan.csv",
            mime="text/csv"
        )


st.divider()

st.caption(
    "BOT AI SAHAM BEI 5.0 | Data dapat terlambat atau tidak lengkap. "
    "Harga entry, stop loss, target, dan skor merupakan estimasi "
    "berdasarkan data historis, bukan jaminan hasil."
)
