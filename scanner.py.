import os
import re
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "sent_notifications.json")

MAX_TICKERS = 100
MIN_SCORE = 70
MODE = "Swing Trading"


def load_tickers():
    path = os.path.join(BASE_DIR, "app.py")

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    match = re.search(
        r'DEFAULT_TICKERS\s*=\s*"""(.*?)"""\s*\.split\(\)',
        source,
        re.DOTALL
    )

    if not match:
        raise RuntimeError(
            "DEFAULT_TICKERS tidak ditemukan di app.py"
        )

    tickers = list(dict.fromkeys(match.group(1).split()))

    return tickers[:MAX_TICKERS]


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        raise RuntimeError(
            "File sent_notifications.json tidak bisa dibaca. "
            "Periksa file tersebut sebelum menjalankan ulang."
        )


def save_state(state):
    temporary_file = STATE_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    os.replace(temporary_file, STATE_FILE)


def fetch_history(ticker):
    symbol = ticker + ".JK"

    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    required = ["High", "Low", "Close", "Volume"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    return df[required].dropna(subset=["Close"]).copy()


def analyze_stock(ticker):
    df = fetch_history(ticker)

    if df.empty or len(df) < 55:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)

    avg_volume = volume.rolling(20).mean()
    vol_ratio = volume / avg_volume.replace(0, np.nan)

    prev_high10 = high.shift(1).rolling(10).max()
    prev_high20 = high.shift(1).rolling(20).max()

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

    if not np.isfinite(price) or price <= 0:
        return None

    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e50 = float(ema50.iloc[-1])

    r = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50

    vr = (
        float(vol_ratio.iloc[-1])
        if pd.notna(vol_ratio.iloc[-1])
        else 0
    )

    a = (
        float(atr.iloc[-1])
        if pd.notna(atr.iloc[-1])
        else price * 0.03
    )

    a = max(a, price * 0.005)

    turnover = float(
        (close * volume).tail(20).mean()
    )

    breakout10 = (
        pd.notna(prev_high10.iloc[-1])
        and price > float(prev_high10.iloc[-1])
    )

    breakout20 = (
        pd.notna(prev_high20.iloc[-1])
        and price > float(prev_high20.iloc[-1])
    )

    score = 0
    notes = []

    if price > e9 > e21:
        score += 20
        notes.append("Tren pendek positif")
    elif price > e21:
        score += 12
        notes.append("Harga di atas EMA21")

    if price > e50:
        score += 10
        notes.append("Harga di atas EMA50")

    if 50 <= r <= 68:
        score += 15
        notes.append("Momentum RSI mendukung")
    elif 42 <= r < 50:
        score += 8

    if vr >= 1.8:
        score += 20
        notes.append("Volume melonjak")
    elif vr >= 1.2:
        score += 12
        notes.append("Volume di atas rata-rata")

    if breakout20:
        score += 20
        notes.append("Breakout high 20 hari")
    elif breakout10:
        score += 12
        notes.append("Breakout high 10 hari")

    if turnover >= 5_000_000_000:
        score += 15
        notes.append("Nilai transaksi rata-rata tinggi")
    elif turnover >= 1_000_000_000:
        score += 10
        notes.append("Nilai transaksi cukup")
    elif turnover >= 250_000_000:
        score += 5

    score = min(score, 100)

    breakout = breakout20

    if score >= 70 and (
        breakout or (price > e21 and vr >= 1.2)
    ):
        signal = "KANDIDAT"
    elif score >= 45:
        signal = "PANTAU"
    else:
        signal = "BELUM MEMENUHI"

    entry_low = max(0, price - 0.35 * a)
    entry_high = price + 0.20 * a
    stop = max(0, price - 1.5 * a)

    target1 = price + 1.5 * a
    target2 = price + 2.5 * a
    target3 = price + 3.5 * a

    previous_price = float(close.iloc[-2])

    change_pct = (
        (price / previous_price - 1) * 100
        if previous_price > 0
        else 0
    )

    return {
        "kode": ticker,
        "harga": price,
        "perubahan": change_pct,
        "skor": score,
        "sinyal": signal,
        "rsi": r,
        "volume_ratio": vr,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "target3": target3,
        "catatan": "; ".join(notes) if notes else "Belum ada konfirmasi kuat",
        "tanggal_data": str(df.index[-1].date())
    }


def rupiah(value):
    return "Rp{:,.0f}".format(value).replace(",", ".")


def send_telegram(token, chat_id, message):
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": message
        },
        timeout=25
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(str(result))

    return True


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum diatur."
        )

    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    today = now.strftime("%Y-%m-%d")

    tickers = load_tickers()
    state = load_state()

    print(f"Waktu Indonesia: {now}")
    print(f"Jumlah ticker: {len(tickers)}")
    print(f"Mode: {MODE}")

    results = []
    failures = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        try:
            result = analyze_stock(ticker)

            if result is not None:
                results.append(result)
            else:
                failures.append(ticker)

        except Exception as exc:
            failures.append(ticker)
            print(f"Gagal {ticker}: {str(exc)[:150]}")

        time.sleep(0.15)

    eligible = [
        row for row in results
        if row["sinyal"] == "KANDIDAT"
        and row["skor"] >= MIN_SCORE
    ]

    print(f"Berhasil dianalisis: {len(results)}")
    print(f"Gagal/tidak tersedia: {len(failures)}")
    print(f"Kandidat sinyal beli: {len(eligible)}")

    sent_count = 0

    for row in eligible:
        ticker = row["kode"]

        # Jangan mengirim ulang ticker yang sudah diberitahukan hari ini.
        if state.get(ticker) == today:
            print(f"Sudah dikirim hari ini: {ticker}")
            continue

        message = (
            "📈 BOT AI SAHAM BEI 6.0\n"
            "🔔 SINYAL KANDIDAT\n\n"
            f"Saham: {ticker}\n"
            f"Skor: {row['skor']}/100\n"
            f"Harga terakhir: {rupiah(row['harga'])}\n"
            f"Perubahan: {row['perubahan']:.2f}%\n"
            f"RSI: {row['rsi']:.2f}\n"
            f"Volume ratio: {row['volume_ratio']:.2f}\n\n"
            f"Entry indikatif: {rupiah(row['entry_low'])}"
            f" - {rupiah(row['entry_high'])}\n"
            f"Stop loss: {rupiah(row['stop'])}\n"
            f"Target 1: {rupiah(row['target1'])}\n"
            f"Target 2: {rupiah(row['target2'])}\n"
            f"Target 3: {rupiah(row['target3'])}\n\n"
            f"Catatan: {row['catatan']}\n"
            f"Tanggal data: {row['tanggal_data']}\n"
            f"Waktu scan: {now.strftime('%d-%m-%Y %H:%M WIB')}\n\n"
            "Data candle harian, bukan harga real-time. "
            "Sinyal bukan jaminan keuntungan."
        )

        try:
            send_telegram(token, chat_id, message)

            # Simpan status setelah Telegram mengonfirmasi pengiriman.
            state[ticker] = today
            save_state(state)

            sent_count += 1
            print(f"Notifikasi terkirim: {ticker}")

        except Exception as exc:
            print(f"Telegram gagal untuk {ticker}: {str(exc)[:150]}")

    print(f"Notifikasi berhasil dikirim: {sent_count}")
    print("Pemindaian selesai.")


if __name__ == "__main__":
    main()
