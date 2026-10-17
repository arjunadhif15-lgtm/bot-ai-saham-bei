import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(
    page_title="Bot AI Saham BEI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 BOT SCANNER SAHAM BEI")
st.caption("Pemindai saham berbasis indikator teknikal")

st.warning(
    "Sinyal bukan jaminan keuntungan. "
    "Gunakan data ini sebagai bahan analisis, "
    "bukan rekomendasi beli atau jual."
)

default_tickers = (
    "BBCA.JK,BBRI.JK,BMRI.JK,BBNI.JK,BBTN.JK,"
    "ANTM.JK,MDKA.JK,BRMS.JK,PTBA.JK,ADRO.JK,"
    "ITMG.JK,INCO.JK,UNTR.JK,ASII.JK,TLKM.JK,"
    "GOTO.JK,BRPT.JK,BREN.JK,DEWA.JK,BUKA.JK,"
    "WOOD.JK,KPIG.JK,BBYB.JK,COCO.JK,FORE.JK,"
    "NCKL.JK,EMTK.JK,AVIA.JK,VKTR.JK,IOTF.JK"
)

tickers_text = st.text_area(
    "Daftar kode saham (.JK)",
    value=default_tickers,
    help="Pisahkan kode saham dengan koma."
)

col1, col2 = st.columns(2)

with col1:
    min_volume_ratio = st.number_input(
        "Minimal rasio volume",
        min_value=0.5,
        max_value=5.0,
        value=1.2,
        step=0.1
    )

with col2:
    capital = st.number_input(
        "Modal simulasi (Rp)",
        min_value=100000,
        value=20000000,
        step=1000000
    )

def calculate_rsi(close, period=14):
    delta = close.diff()

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

    return rsi

def scan_stock(ticker):
    data = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty or len(data) < 30:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    volume = data["Volume"].squeeze()

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    rsi = calculate_rsi(close)

    avg_volume = volume.rolling(20).mean()
    volume_ratio = volume / avg_volume.replace(0, np.nan)

    previous_high = high.shift(1).rolling(20).max()

    price = float(close.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    current_volume_ratio = float(volume_ratio.iloc[-1])

    trend_up = ema9.iloc[-1] > ema21.iloc[-1]

    breakout = high.iloc[-1] > previous_high.iloc[-1]

    volume_strong = (
        current_volume_ratio >= min_volume_ratio
    )

    signal = "BELUM ADA SINYAL"

    if (
        trend_up
        and breakout
        and volume_strong
        and 50 <= current_rsi <= 70
    ):
        signal = "BREAKOUT POTENSIAL"

    elif trend_up and 45 <= current_rsi <= 65:
        signal = "PANTAU"

    stop_loss = price * 0.97
    target_1 = price * 1.05
    target_2 = price * 1.08

    lot = int(capital // (price * 100))
    lot = max(lot, 0)

    return {
        "Kode": ticker,
        "Harga": round(price, 2),
        "RSI": round(current_rsi, 2),
        "EMA 9": round(float(ema9.iloc[-1]), 2),
        "EMA 21": round(float(ema21.iloc[-1]), 2),
        "Volume x": round(current_volume_ratio, 2),
        "Sinyal": signal,
        "Stop simulasi": round(stop_loss, 2),
        "Target +5%": round(target_1, 2),
        "Target +8%": round(target_2, 2),
        "Maks. lot simulasi": lot
    }

if st.button("🔎 MULAI SCAN SAHAM", type="primary"):
    tickers = list(dict.fromkeys(
        t.strip().upper()
        for t in tickers_text.split(",")
        if t.strip()
    ))

    if not tickers:
        st.error("Masukkan minimal satu kode saham.")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(tickers):
            status.write(
                f"Memeriksa {ticker} "
                f"({i + 1}/{len(tickers)})..."
            )

            try:
                result = scan_stock(ticker)

                if result is not None:
                    results.append(result)

            except Exception:
                pass

            progress.progress((i + 1) / len(tickers))

        status.empty()
        progress.empty()

        if results:
            df = pd.DataFrame(results)

if not df.empty:
    if "Sinyal" in df.columns:
        urutan_sinyal = {
            "BREAKOUT POTENSIAL": 3,
            "PANTAU": 2,
            "BELUM ADA SINYAL": 1
        }

        df["Prioritas"] = (
            df["Sinyal"].map(urutan_sinyal).fillna(0)
        )

        df = df.sort_values(
            by="Prioritas",
            ascending=False
        )

    st.subheader("🏆 TOP 10 Saham Potensial")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📊 Hasil Pemindaian Semua Saham")
    st.dataframe(
        df.drop(columns=["Prioritas"], errors="ignore"),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Belum ada data saham yang berhasil dipindai.")
            df = pd.DataFrame(results
          
                df.sort_values(
                    by=["Sinyal", "Volume x"],
                    ascending=[True, False]
                ),
                use_container_width=True,
                hide_index=True
            )

            opportunities = df[
                df["Sinyal"] == "BREAKOUT POTENSIAL"
            ]

            st.subheader("🎯 Saham dengan Sinyal Breakout")

            if not opportunities.empty:
                st.dataframe(
                    opportunities,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "Belum ada saham yang memenuhi "
                    "seluruh kriteria. Jangan memaksakan entry."
                )

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Unduh hasil scan CSV",
                data=csv,
                file_name="hasil_scan_saham.csv",
                mime="text/csv"
            )

        else:
            st.error(
                "Data saham tidak berhasil diperoleh. "
                "Coba lagi nanti atau periksa kode saham."
            )

st.divider()

st.caption(
    "Data bersumber dari Yahoo Finance melalui yfinance. "
    "Data dapat terlambat, tidak tersedia, atau berbeda "
    "dari data resmi bursa. Target dan stop hanyalah simulasi."
  )
