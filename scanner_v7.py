import os
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =====================================
# BOT AI SAHAM BEI V7.0
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "sent_notifications_v7.json")

MAX_TICKERS = None
MIN_SCORE = 70
MIN_VOLUME_RATIO = 1.2
MIN_REWARD_RISK = 1.5
TOP_N = 10

TIMEZONE = ZoneInfo("Asia/Jakarta")

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN",
    "ANTM", "MDKA", "BRMS", "PTBA", "ADRO",
    "ITMG", "INCO", "UNTR", "ASII", "TLKM",
    "GOTO", "BRPT", "BREN", "DEWA", "BUKA",
    "WOOD", "KPIG", "BBYB", "COCO", "FORE",
    "NCKL", "EMTK", "AVIA", "VKTR", "IOTF"
]


def load_tickers():
    """Membaca daftar saham dari aplikasi jika tersedia."""
    try:
        import app
        raw = getattr(
            app, "DEFAULT_TICKERS", DEFAULT_TICKERS
        )
    except Exception:
        raw = DEFAULT_TICKERS

    if isinstance(raw, str):
        items = raw.replace(",", " ").split()
    else:
        items = list(raw)

    result = []

    for item in items:
        ticker = str(item).strip().upper()

        if ticker.endswith(".JK"):
            ticker = ticker[:-3]

        if ticker and ticker not in result:
            result.append(ticker)

    return result


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as file:
            json.dump(
                state, file, ensure_ascii=False, indent=2
            )
    except OSError as error:
        print("Gagal menyimpan status:", error)


def rupiah(value):
    return "Rp" + f"{value:,.0f}".replace(",", ".")


def fetch_history(ticker):
    """Mengambil harga saham dari Yahoo Finance."""
    try:
        df = yf.download(
            f"{ticker}.JK",
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]

        df.columns = [str(col).title() for col in df.columns]

        required = ["High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            return None

        df = df[required].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if len(df) < 60:
            return None

        return df

    except Exception as error:
        print(f"{ticker}: gagal mengambil data: {error}")
        return None


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

    return (100 - 100 / (1 + rs)).fillna(50)


def analyze_stock(ticker, df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    rsi = calculate_rsi(close)
    avg_volume = volume.rolling(20).mean()
    resistance = high.shift(1).rolling(20).max()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(14).mean()

    price = float(close.iloc[-1])
    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])

    avg_vol = float(avg_volume.iloc[-1])
    current_vol = float(volume.iloc[-1])

    resistance_now = resistance.iloc[-1]

    if pd.isna(resistance_now):
        resistance_now = np.nan
    else:
        resistance_now = float(resistance_now)

    atr_now = atr.iloc[-1]

    if pd.isna(atr_now) or atr_now <= 0:
        atr_now = price * 0.03
    else:
        atr_now = float(atr_now)

    volume_ratio = (
        current_vol / avg_vol if avg_vol > 0 else 0
    )

    trend_up = (
        price > ema21_now and ema9_now > ema21_now
    )

    breakout = (
        pd.notna(resistance_now)
        and price > resistance_now
    )

    false_breakout = (
        pd.notna(resistance_now)
        and float(high.iloc[-1]) > resistance_now
        and price < resistance_now
    )

    distance = (
        (resistance_now - price) / price * 100
        if pd.notna(resistance_now) and price > 0
        else np.nan
    )

    prebreakout = (
        not breakout
        and not false_breakout
        and pd.notna(distance)
        and 0 <= distance <= 3
        and price > ema21_now
        and volume_ratio >= 0.8
        and rsi_now >= 45
    )

    momentum_5d = (
        (price / float(close.iloc[-6]) - 1) * 100
        if len(close) >= 6 and close.iloc[-6] > 0
        else 0
    )

    momentum = (
        not breakout
        and not false_breakout
        and trend_up
        and momentum_5d > 0
        and pd.notna(distance)
        and 0 <= distance <= 5
        and volume_ratio >= 0.8
        and 45 <= rsi_now <= 70
    )

    # Indikasi teknikal, bukan bukti transaksi bandar.
    flat_10d = (
        len(close) >= 11
        and abs(
            (price / float(close.iloc[-11]) - 1) * 100
        ) <= 4
    )

    recent_volume = float(volume.tail(5).mean())

    previous_volume = (
        float(volume.iloc[-25:-5].mean())
        if len(volume) >= 25 else 0
    )

    accumulation = (
        flat_10d
        and previous_volume > 0
        and recent_volume >= previous_volume * 1.15
        and price > ema21_now
    )

    score = 0

    if trend_up:
        score += 20

    if price > ema50_now:
        score += 10

    if breakout:
        score += 25

    if prebreakout:
        score += 15

    if volume_ratio >= MIN_VOLUME_RATIO:
        score += 15
    elif volume_ratio >= 0.8:
        score += 5

    if 50 <= rsi_now <= 65:
        score += 10
    elif 45 <= rsi_now <= 70:
        score += 5

    if price > ema9_now:
        score += 5

    if accumulation:
        score += 10

    score = min(score, 100)

    # Filter agar tidak mengejar harga terlalu tinggi.
    above_ema9 = (
        (price - ema9_now) / ema9_now * 100
        if ema9_now > 0 else 0
    )

    watch_only = (
        rsi_now > 70 or above_ema9 > 3
    )

    # Area entry.
    if price > ema9_now:
        entry_low = min(price, ema9_now)
    else:
        entry_low = price * 0.995

    entry_high = price

    # Stop loss dan target berdasarkan ATR.
    stop_loss = max(price - 1.5 * atr_now, 0)
    risk = price - stop_loss

    target_1 = price + max(
        2.3 * atr_now,
        risk * MIN_REWARD_RISK
    )

    target_2 = price + max(
        3 * atr_now,
        risk * 2
    )

    rr1 = (
        (target_1 - price) / risk
        if risk > 0 else 0
    )

    rr2 = (
        (target_2 - price) / risk
        if risk > 0 else 0
    )

    risk_ok = (
        rr1 >= MIN_REWARD_RISK
        and not false_breakout
        and not watch_only
    )

    if false_breakout:
        signal = "WASPADA FALSE BREAKOUT"
    elif (
        breakout and trend_up
        and volume_ratio >= MIN_VOLUME_RATIO
        and 50 <= rsi_now <= 70
    ):
        signal = "BREAKOUT TERKONFIRMASI"
    elif prebreakout:
        signal = "CALON BREAKOUT"
    elif accumulation:
        signal = "INDIKASI AKUMULASI TEKNIKAL"
    elif momentum:
        signal = "MOMENTUM AWAL"
    elif trend_up:
        signal = "PANTAU"
    else:
        signal = "BELUM ADA SINYAL"

    candidate = (
        score >= MIN_SCORE
        and risk_ok
        and not false_breakout
        and (
            breakout
            or prebreakout
            or accumulation
            or momentum
            or (
                trend_up
                and volume_ratio >= MIN_VOLUME_RATIO
            )
        )
    )

    return {
        "ticker": ticker,
        "price": price,
        "score": score,
        "signal": signal,
        "rsi": rsi_now,
        "volume_ratio": volume_ratio,
        "momentum_5d": momentum_5d,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "rr1": rr1,
        "rr2": rr2,
        "is_candidate": candidate
    }


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("Telegram belum dikonfigurasi.")
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=20
        )

        if response.ok:
            return True

        print("Telegram gagal:", response.status_code)

    except requests.RequestException as error:
        print("Telegram gagal:", error)

    return False


def make_message(x):
    return (
        f"BOT AI SAHAM BEI V7.0\n"
        f"Saham: {x['ticker']}.JK\n"
        f"Sinyal: {x['signal']}\n"
        f"Skor: {x['score']}/100\n"
        f"Harga: {rupiah(x['price'])}\n"
        f"Area entry: {rupiah(x['entry_low'])} - "
        f"{rupiah(x['entry_high'])}\n"
        f"Stop Loss: {rupiah(x['stop_loss'])}\n"
        f"TP1: {rupiah(x['target_1'])} "
        f"(RR {x['rr1']:.2f}R)\n"
        f"TP2: {rupiah(x['target_2'])} "
        f"(RR {x['rr2']:.2f}R)\n"
        f"RSI: {x['rsi']:.1f}\n"
        f"Rasio volume: {x['volume_ratio']:.2f}x\n"
        f"Broker lokal/asing: belum terhubung.\n"
        f"Data broker aktual belum tersedia.\n"
        f"Catatan: sinyal bukan jaminan harga naik."
    )


def scan(send_alerts=True):
    tickers = load_tickers()
    state = load_state()
    today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    results = []

    print(f"BOT AI SAHAM BEI V7.0: {len(tickers)} saham")

    for number, ticker in enumerate(tickers, start=1):
        print(f"[{number}/{len(tickers)}] Memindai {ticker}.JK")

        df = fetch_history(ticker)

        if df is not None:
            try:
                result = analyze_stock(ticker, df)
                results.append(result)

                print(
                    f"{result['signal']} | Skor {result['score']} | "
                    f"Harga {rupiah(result['price'])}"
                )

                if send_alerts and result["is_candidate"]:
                    if state.get(ticker) != today:
                        if send_telegram(make_message(result)):
                            state[ticker] = today
                            save_state(state)
                    else:
                        print(
                            "Notifikasi saham ini sudah dikirim hari ini."
                        )

            except Exception as error:
                print(f"Gagal menganalisis {ticker}: {error}")

        time.sleep(0.25)

    results.sort(
        key=lambda item: (
            item["score"], item["volume_ratio"]
        ),
        reverse=True
    )

    print("\n=== 10 SAHAM TERATAS ===")

    for x in results[:TOP_N]:
        print(
            f"{x['ticker']}.JK | {x['signal']} | "
            f"Skor {x['score']} | Harga {rupiah(x['price'])} | "
            f"Entry {rupiah(x['entry_low'])}-"
            f"{rupiah(x['entry_high'])} | "
            f"SL {rupiah(x['stop_loss'])} | "
            f"TP1 {rupiah(x['target_1'])} | "
            f"TP2 {rupiah(x['target_2'])}"
        )

    print(
        "\nData broker lokal/asing belum terhubung. "
        "Yahoo Finance tidak menyediakan broker summary."
    )

    return results


if __name__ == "__main__":
    scan()
