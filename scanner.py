import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# PENGATURAN BOT
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "sent_notifications.json")

MIN_SCORE = 70
MIN_VOLUME_RATIO = 1.2
MAX_RSI_ENTRY = 70.0  # RSI di atas ini diberi label PANTAU DULU
MAX_ABOVE_EMA9_PCT = 3.0  # Harga >3% di atas EMA9 diberi label PANTAU DULU
MAX_TICKERS = None  # None = pindai semua saham di DEFAULT_TICKERS
REQUEST_DELAY_SECONDS = 0.25
YF_PERIOD = "1y"
YF_INTERVAL = "1d"
TIMEZONE = "Asia/Jakarta"


# ============================================================
# MEMBACA DAFTAR SAHAM DARI app.py
# ============================================================
def load_tickers():
    """Mengambil semua kode saham dari DEFAULT_TICKERS di app.py."""
    app_path = os.path.join(BASE_DIR, "app.py")

    if not os.path.isfile(app_path):
        raise FileNotFoundError(
            "app.py tidak ditemukan di folder yang sama dengan scanner.py"
        )

    with open(app_path, "r", encoding="utf-8") as file:
        source = file.read()

    match = re.search(
        r'DEFAULT_TICKERS\s*=\s*"""(.*?)"""\s*\.split\(\)',
        source,
        re.DOTALL,
    )

    if not match:
        raise RuntimeError(
            'DEFAULT_TICKERS tidak ditemukan. Pastikan format di app.py adalah '
            'DEFAULT_TICKERS = """ ... """.split()'
        )

    tickers = []

    for ticker in match.group(1).split():
        ticker = ticker.strip().upper()

        if ticker and ticker not in tickers:
            tickers.append(ticker)

    if MAX_TICKERS is not None:
        tickers = tickers[:MAX_TICKERS]

    if not tickers:
        raise RuntimeError("Daftar DEFAULT_TICKERS kosong.")

    return tickers


# ============================================================
# STATUS NOTIFIKASI
# MAKSIMAL 1 NOTIFIKASI PER SAHAM PER HARI
# ============================================================
def load_state():
    if not os.path.isfile(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (json.JSONDecodeError, OSError):
        print(
            "PERINGATAN: sent_notifications.json tidak terbaca. "
            "Memulai status kosong."
        )
        return {}


def save_state(state):
    """Simpan status secara aman."""
    temp_path = STATE_FILE + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)

    os.replace(temp_path, STATE_FILE)


# ============================================================
# MENGAMBIL DATA HARGA HARIAN
# ============================================================
def fetch_history(ticker):
    yahoo_ticker = f"{ticker}.JK"

    try:
        data = yf.download(
            yahoo_ticker,
            period=YF_PERIOD,
            interval=YF_INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )

    except Exception as exc:
        print(f"GAGAL mengambil data {ticker}: {exc}")
        return None

    if data is None or data.empty:
        print(f"TIDAK ADA DATA: {ticker}")
        return None

    # Menangani kolom MultiIndex dari yfinance.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(col).title() for col in data.columns]

    required = ["High", "Low", "Close", "Volume"]

    if any(col not in data.columns for col in required):
        print(f"DATA TIDAK LENGKAP: {ticker}")
        return None

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=["High", "Low", "Close", "Volume"]
    )

    data = data[
        (data["Close"] > 0) &
        (data["Volume"] >= 0)
    ]

    if len(data) < 60:
        print(
            f"DATA BELUM CUKUP (kurang dari 60 candle): {ticker}"
        )
        return None

    return data


# ============================================================
# ANALISIS TEKNIKAL DAN PERHITUNGAN SKOR
# ============================================================
def analyze_stock(ticker, data):
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # EMA
    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    ema50 = close.ewm(
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

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).clip(0, 100)

    # Rasio volume terhadap rata-rata 20 hari
    avg_volume20 = volume.rolling(20).mean()

    if (
        pd.notna(avg_volume20.iloc[-1])
        and avg_volume20.iloc[-1] > 0
    ):
        volume_ratio = (
            volume.iloc[-1] / avg_volume20.iloc[-1]
        )
    else:
        volume_ratio = 0

    # Breakout terhadap tertinggi 20 candle sebelumnya
    prior_high20 = (
        high.shift(1).rolling(20).max().iloc[-1]
    )

    price = float(close.iloc[-1])
    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])

    high20 = (
        float(prior_high20)
        if pd.notna(prior_high20)
        else np.nan
    )

    # Kondisi teknikal
    trend_up = ema9_now > ema21_now
    trend_medium_up = ema21_now > ema50_now

    breakout20 = (
        pd.notna(high20)
        and price > high20
    )

    volume_strong = volume_ratio >= MIN_VOLUME_RATIO
    price_above_ema21 = price > ema21_now
    distance_above_ema9_pct = ((price / ema9_now) - 1) * 100 if ema9_now > 0 else 0.0
    overheated_rsi = rsi_now > MAX_RSI_ENTRY
    price_stretched = distance_above_ema9_pct > MAX_ABOVE_EMA9_PCT
    watch_only = overheated_rsi or price_stretched

    # ATR 14 untuk perkiraan risiko
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = true_range.rolling(14).mean().iloc[-1]

    if pd.notna(atr14):
        atr14 = float(atr14)
    else:
        atr14 = price * 0.03

    # ========================================================
    # SKOR TEKNIKAL
    # Maksimum skor: 100
    # ========================================================
    score = 0

    if trend_up:
        score += 20

    if trend_medium_up:
        score += 15

    if breakout20:
        score += 25

    if volume_strong:
        score += 15

    if 50 <= rsi_now <= 65:
        score += 15

    elif 45 <= rsi_now < 50:
        score += 5

    elif 65 < rsi_now <= 70:
        score += 10

    if price > ema9_now:
        score += 10

    # ========================================================
    # SYARAT KANDIDAT PILIHAN A
    # Skor >= 70 DAN memenuhi kondisi teknikal.
    # ========================================================
    technical_condition = (
        breakout20
        or (
            price_above_ema21
            and volume_strong
        )
    )

    is_candidate = (
        score >= MIN_SCORE
        and technical_condition
    )

    if is_candidate and watch_only:
        reasons = []
        if overheated_rsi:
            reasons.append(f"RSI tinggi ({rsi_now:.1f})")
        if price_stretched:
            reasons.append(f"harga {distance_above_ema9_pct:.1f}% di atas EMA9")
        signal = "PANTAU DULU — " + "; ".join(reasons)
    elif is_candidate:
        signal = "KANDIDAT ENTRY"

    elif score >= MIN_SCORE:
        signal = (
            "SKOR MEMENUHI, "
            "SYARAT TEKNIKAL BELUM LENGKAP"
        )

    else:
        signal = "BELUM ADA SINYAL"

    # ========================================================
    # AREA ENTRY, STOP LOSS, DAN TARGET PROFIT
    # Entry adalah perkiraan teknikal, bukan order otomatis.
    # ========================================================
    stop_loss = max(
        0,
        price - (1.5 * atr14)
    )

    target_1 = price + (2 * atr14)
    target_2 = price + (3 * atr14)

    if watch_only:
        entry_status = "PANTAU DULU — jangan mengejar harga; tunggu kondisi lebih ideal"
    elif breakout20:
        # Breakout terdeteksi: area entry indikatif di sekitar
        # harga penutupan sampai sedikit di atasnya.
        entry_low = price
        entry_high = price + (0.25 * atr14)
        entry_status = (
            "Breakout terdeteksi; periksa harga pasar terkini"
        )

    elif price_above_ema21 and volume_strong:
        # Belum breakout: perkiraan area pullback di sekitar
        # EMA21 hingga harga penutupan.
        entry_low = max(
            ema21_now,
            price - (0.5 * atr14)
        )
        entry_high = price
        entry_status = (
            "Belum breakout; tunggu konfirmasi dan pullback"
        )

    else:
        entry_low = np.nan
        entry_high = np.nan
        entry_status = (
            "Belum ada area entry yang memenuhi syarat"
        )

    return {
        "ticker": ticker,
        "price": price,
        "score": int(score),
        "signal": signal,
        "is_candidate": bool(is_candidate),
        "rsi": rsi_now,
        "distance_above_ema9_pct": float(distance_above_ema9_pct),
        "watch_only": bool(watch_only),
        "volume_ratio": float(volume_ratio),
        "ema9": ema9_now,
        "ema21": ema21_now,
        "ema50": ema50_now,
        "breakout20": bool(breakout20),
        "trend_up": bool(trend_up),
        "volume_strong": bool(volume_strong),
        "entry_low": float(entry_low),
        "entry_high": float(entry_high),
        "entry_status": entry_status,
        "stop_loss": float(stop_loss),
        "target_1": float(target_1),
        "target_2": float(target_2),
        "date": data.index[-1].strftime("%Y-%m-%d"),
    }


# ============================================================
# MENGIRIM PESAN TELEGRAM
# ============================================================
def send_telegram(message):
    token = os.getenv(
        "TELEGRAM_BOT_TOKEN", ""
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID", ""
    ).strip()

    if not token or not chat_id:
        print(
            "TELEGRAM TIDAK DIKIRIM: "
            "TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID "
            "belum diatur."
        )
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        if response.ok:
            payload = response.json()

            if payload.get("ok") is True:
                return True

        print(
            f"Telegram gagal: HTTP {response.status_code} "
            f"- {response.text[:300]}"
        )

    except requests.RequestException as exc:
        print(
            f"Telegram gagal mengirim pesan: {exc}"
        )

    return False


# ============================================================
# FORMAT HARGA RUPIAH
# ============================================================
def format_rupiah(value):
    return (
        f"Rp{value:,.0f}"
        .replace(",", ".")
    )


# ============================================================
# FORMAT PESAN KANDIDAT
# ============================================================
def make_telegram_message(result):
    breakout_text = (
        "Ya"
        if result["breakout20"]
        else "Tidak"
    )

    if (
        np.isfinite(result["entry_low"])
        and np.isfinite(result["entry_high"])
    ):
        entry_text = (
            f"{format_rupiah(result['entry_low'])}"
            f" - "
            f"{format_rupiah(result['entry_high'])}"
        )
    else:
        entry_text = "Belum tersedia"

    heading = "👀 PANTAU DULU — JANGAN MENGEJAR HARGA" if result.get("watch_only", False) else "📈 KANDIDAT ENTRY SAHAM BEI"

    return (
        f"{heading}: "
        f"{result['ticker']}\n"

        f"Skor teknikal: "
        f"{result['score']}/100\n"

        f"Harga penutupan data harian: "
        f"{format_rupiah(result['price'])}\n"

        f"RSI(14): "
        f"{result['rsi']:.1f}\n"

        f"Jarak harga dari EMA9: "
        f"{result.get('distance_above_ema9_pct', 0):.1f}%\n"

        f"Rasio volume: "
        f"{result['volume_ratio']:.2f}x\n"

        f"Breakout 20 hari: "
        f"{breakout_text}\n\n"

        f"🎯 RENCANA TRANSAKSI INDIKATIF\n"

        f"Area entry: {entry_text}\n"

        f"Status entry: "
        f"{result['entry_status']}\n\n"

        f"Stop loss indikatif: "
        f"{format_rupiah(result['stop_loss'])}\n"

        f"Target 1 indikatif: "
        f"{format_rupiah(result['target_1'])}\n"

        f"Target 2 indikatif: "
        f"{format_rupiah(result['target_2'])}\n\n"

        f"Tanggal candle: "
        f"{result['date']}\n\n"

        "Catatan: harga berdasarkan data harian "
        "Yahoo Finance dan dapat terlambat atau "
        "tidak lengkap. Area entry, stop loss, "
        "dan target merupakan perkiraan teknikal, "
        "bukan jaminan profit atau instruksi "
        "otomatis untuk membeli. Jika berstatus PANTAU DULU, "
        "hindari mengejar harga dan tunggu evaluasi ulang. Periksa harga "
        "terkini sebelum melakukan transaksi."
    )


# ============================================================
# PROSES UTAMA
# ============================================================
def main():
    now = datetime.now(
        ZoneInfo(TIMEZONE)
    )

    today = now.strftime("%Y-%m-%d")

    tickers = load_tickers()
    state = load_state()

    print("=" * 60)
    print("BOT SCANNER SAHAM BEI")
    print(
        f"Waktu mulai WIB: "
        f"{now:%Y-%m-%d %H:%M:%S}"
    )
    print(
        f"Jumlah kode untuk dipindai: "
        f"{len(tickers)}"
    )
    print(
        f"Skor minimum kandidat: "
        f"{MIN_SCORE}"
    )
    print(
        f"Syarat volume kuat: rasio >= "
        f"{MIN_VOLUME_RATIO:.1f}x"
    )
    print(
        "Notifikasi: maksimal 1 kali "
        "per saham per hari"
    )
    print("=" * 60)

    scanned = 0
    failures = 0
    candidates = 0
    notifications_sent = 0

    for index, ticker in enumerate(
        tickers,
        start=1
    ):
        print(
            f"[{index}/{len(tickers)}] "
            f"Memindai {ticker}...",
            flush=True,
        )

        data = fetch_history(ticker)

        if data is None:
            failures += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        try:
            result = analyze_stock(
                ticker,
                data
            )

        except Exception as exc:
            failures += 1
            print(
                f"ERROR analisis {ticker}: {exc}"
            )
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        scanned += 1

        print(
            f"  Harga="
            f"{format_rupiah(result['price'])} | "
            f"Skor={result['score']} | "
            f"Sinyal={result['signal']} | "
            f"RSI={result['rsi']:.1f} | "
            f"Volume="
            f"{result['volume_ratio']:.2f}x",
            flush=True,
        )

        if result["is_candidate"]:
            candidates += 1

            if state.get(ticker) == today:
                print(
                    f"  Telegram dilewati: {ticker} "
                    "sudah mendapat notifikasi hari ini."
                )

            else:
                message = make_telegram_message(
                    result
                )

                if send_telegram(message):
                    state[ticker] = today
                    save_state(state)

                    notifications_sent += 1

                    print(
                        f"  NOTIFIKASI TERKIRIM: "
                        f"{ticker}"
                    )

                else:
                    print(
                        f"  NOTIFIKASI GAGAL: {ticker}; "
                        "akan bisa dicoba lagi "
                        "pada proses berikutnya."
                    )

        time.sleep(REQUEST_DELAY_SECONDS)

    print("=" * 60)
    print("RINGKASAN SCAN")
    print(
        f"Jumlah kode dalam daftar : "
        f"{len(tickers)}"
    )
    print(
        f"Berhasil dianalisis      : "
        f"{scanned}"
    )
    print(
        f"Gagal/tidak ada data     : "
        f"{failures}"
    )
    print(
        f"Kandidat skor >= {MIN_SCORE} "
        "+ syarat teknikal: {candidates}"
    )
    print(
        f"Notifikasi terkirim kali ini: "
        f"{notifications_sent}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
