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
# PENGATURAN SEDERHANA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEZONE = "Asia/Jakarta"
YF_PERIOD = "1y"
YF_INTERVAL = "1d"
TOP_N = 10
MIN_SCORE = 60
MIN_VOLUME_RATIO = 1.2
REQUEST_DELAY_SECONDS = 0.3


def load_tickers():
    """Membaca DEFAULT_TICKERS dari app.py."""
    app_path = os.path.join(BASE_DIR, "app.py")
    if not os.path.isfile(app_path):
        raise FileNotFoundError("app.py tidak ditemukan di folder yang sama.")

    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()

    match = re.search(
        r'DEFAULT_TICKERS\s*=\s*"""(.*?)"""\s*\.split\(\)',
        source,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            'DEFAULT_TICKERS tidak ditemukan. Format yang diharapkan: '
            'DEFAULT_TICKERS = """BBCA.JK BBRI.JK ...""".split()'
        )

    tickers = list(dict.fromkeys(
        item.strip().upper() for item in match.group(1).split() if item.strip()
    ))
    if not tickers:
        raise RuntimeError("Daftar DEFAULT_TICKERS kosong.")
    return tickers


def fetch_history(ticker):
    """Mengambil data harian dari Yahoo Finance."""
    try:
        data = yf.download(
            ticker,
            period=YF_PERIOD,
            interval=YF_INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as exc:
        print(f"GAGAL mengambil {ticker}: {exc}")
        return None

    if data is None or data.empty:
        print(f"Tidak ada data untuk {ticker}")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(col).title() for col in data.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in data.columns for col in needed):
        print(f"Data OHLCV tidak lengkap: {ticker}")
        return None

    data = data[needed].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[(data["Close"] > 0) & (data["Volume"] >= 0)]
    return data if len(data) >= 60 else None


def analyze_stock(ticker, data):
    """Skor teknikal ringkas, Entry 1 breakout, Entry 2 pullback support + Fibonacci."""
    close, high, low = data["Close"], data["High"], data["Low"]
    open_price, volume = data["Open"], data["Volume"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50).clip(0, 100)

    avg_vol20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = float(volume.iloc[-1] / avg_vol20) if pd.notna(avg_vol20) and avg_vol20 > 0 else 0.0

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    ema9_now, ema21_now, ema50_now = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])

    resistance = high.shift(1).rolling(20).max().iloc[-1]
    resistance = float(resistance) if pd.notna(resistance) else np.nan
    support = float(low.iloc[-20:].min())

    trend_up = ema9_now > ema21_now
    medium_trend_up = ema21_now > ema50_now
    volume_strong = volume_ratio >= MIN_VOLUME_RATIO
    breakout = bool(np.isfinite(resistance) and price > resistance)
    near_breakout = bool(
        np.isfinite(resistance) and price <= resistance
        and price >= resistance * 0.97 and price > ema21_now
        and volume_ratio >= 0.8 and rsi_now >= 45
    )

    # ATR untuk mengukur jarak stop loss/target.
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr = float(atr) if pd.notna(atr) and atr > 0 else price * 0.03

    # Fibonacci dari swing high/low 60 candle terakhir.
    swing_high = float(high.iloc[-60:].max())
    swing_low = float(low.iloc[-60:].min())
    swing_range = swing_high - swing_low
    fib_levels = []
    if swing_range > 0:
        fib_levels = [
            swing_high - swing_range * 0.382,
            swing_high - swing_range * 0.500,
            swing_high - swing_range * 0.618,
        ]

    # Entry 1 selalu ditampilkan sebagai AREA RENCANA breakout.
    # Jika resistance tidak tersedia, gunakan EMA21 sebagai acuan sementara.
    entry1_anchor = resistance if np.isfinite(resistance) else ema21_now
    entry1_low = max(0.01, entry1_anchor)
    entry1_high = max(entry1_low, entry1_anchor + 0.25 * atr)
    entry1_confirmed = bool(
        breakout and volume_strong and trend_up and 45 <= rsi_now <= 70
        and price <= ema9_now * 1.03
    )

    # Entry 2 selalu ditampilkan sebagai AREA RENCANA pullback.
    # Utamakan pertemuan support dan Fibonacci; jika tidak dekat, gunakan support.
    fib_near_support = None
    zone_center = support
    if fib_levels:
        closest_fib = min(fib_levels, key=lambda level: abs(level - support))
        if abs(closest_fib - support) <= 1.5 * atr:
            fib_near_support = closest_fib
            zone_center = (support + closest_fib) / 2
    entry2_low = max(0.01, zone_center - 0.25 * atr)
    entry2_high = max(entry2_low, zone_center + 0.25 * atr)

    # Konfirmasi pantulan hanya status tambahan; tidak menghapus angka area Entry 2.
    touched_zone = float(low.iloc[-1]) <= entry2_high and price >= entry2_low
    bounce_confirmed = bool(
        touched_zone and price > float(open_price.iloc[-1]) and price > prev_close
    )

    score = 0
    if trend_up:
        score += 20
    if medium_trend_up:
        score += 10
    if breakout:
        score += 25
    elif near_breakout:
        score += 15
    if volume_strong:
        score += 15
    if 50 <= rsi_now <= 65:
        score += 15
    elif 45 <= rsi_now <= 70:
        score += 8
    if price > ema9_now:
        score += 5
    if bounce_confirmed:
        score += 5
    score = min(100, score)

    if breakout and volume_strong and trend_up and rsi_now <= 70:
        signal = "BREAKOUT"
    elif near_breakout:
        signal = "CALON BREAKOUT"
    elif bounce_confirmed:
        signal = "PANTULAN SUPPORT"
    elif score >= MIN_SCORE:
        signal = "PANTAU"
    else:
        signal = "BELUM ADA SINYAL"

    stop_loss = max(0.01, price - 1.5 * atr)
    target1 = price + 2.0 * atr
    target2 = price + 3.0 * atr

    return {
        "ticker": ticker,
        "price": price,
        "score": int(score),
        "signal": signal,
        "rsi": rsi_now,
        "volume_ratio": volume_ratio,
        "resistance": resistance,
        "support": support,
        "fib": fib_near_support,
        "entry1_low": entry1_low,
        "entry1_high": entry1_high,
        "entry1_confirmed": entry1_confirmed,
        "entry2_low": entry2_low,
        "entry2_high": entry2_high,
        "bounce_confirmed": bounce_confirmed,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "date": data.index[-1].strftime("%d-%m-%Y"),
    }


def rupiah(value):
    try:
        value = float(value)
        if np.isfinite(value) and value > 0:
            return "Rp" + f"{value:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        pass
    return "Belum tersedia"


def make_top_message(results):
    now = datetime.now(ZoneInfo(TIMEZONE))
    ranked = sorted(results, key=lambda r: (r["score"], r["volume_ratio"]), reverse=True)[:TOP_N]
    lines = [
        f"📊 SCREENER SAHAM BEI — TOP {len(ranked)}",
        f"🕒 {now:%d-%m-%Y %H:%M} WIB",
        "Entry 1 = area rencana breakout; Entry 2 = area rencana pullback support/Fibonacci.",
        "",
    ]

    for i, r in enumerate(ranked, 1):
        e1 = f"{rupiah(r['entry1_low'])}–{rupiah(r['entry1_high'])}"
        e2 = f"{rupiah(r['entry2_low'])}–{rupiah(r['entry2_high'])}"
        e1_status = "siap dipertimbangkan" if r["entry1_confirmed"] else "area rencana; tunggu konfirmasi"
        e2_status = "pantulan terdeteksi" if r["bounce_confirmed"] else "area rencana; tunggu harga + pantulan"
        fib = rupiah(r["fib"]) if r["fib"] is not None else "Belum cocok"
        lines.extend([
            f"{i}. {r['ticker']} | Skor {r['score']}/100 | {r['signal']}",
            f"💰 Harga: {rupiah(r['price'])} | RSI {r['rsi']:.1f} | Volume {r['volume_ratio']:.2f}x",
            f"🎯 Entry 1 (breakout): {e1}",
            f"   Status Entry 1: {e1_status}",
            f"🔄 Entry 2 (pullback): {e2}",
            f"   Status Entry 2: {e2_status}",
            f"📉 Support: {rupiah(r['support'])} | Fibonacci dekat support: {fib}",
            f"🛑 SL: {rupiah(r['stop_loss'])} | 🎯 T1: {rupiah(r['target1'])} | T2: {rupiah(r['target2'])}",
            "",
        ])

    lines.append(
        "Catatan: sinyal bersifat indikatif, data Yahoo Finance bisa terlambat. "
        "Kedua area adalah angka rencana, bukan jaminan beli. Entry 1 perlu konfirmasi breakout; "
        "Entry 2 perlu pantulan. Periksa harga terbaru dan fraksi harga BEI sebelum transaksi."
    )
    message = "\n".join(lines)
    return message[:3950] + "\nPesan dipersingkat." if len(message) > 4000 else message


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram dilewati: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID belum diatur.")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("ok") is True
    except Exception as exc:
        print(f"Gagal mengirim Telegram: {exc}")
        return False


def main():
    tickers = load_tickers()
    results = []
    failures = 0
    print(f"Mulai scanner: {len(tickers)} saham | {datetime.now(ZoneInfo(TIMEZONE)):%d-%m-%Y %H:%M} WIB")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Memindai {ticker}...", flush=True)
        data = fetch_history(ticker)
        if data is None:
            failures += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        try:
            result = analyze_stock(ticker, data)
            results.append(result)
            print(
                f"{ticker}: harga {rupiah(result['price'])}, skor {result['score']}, "
                f"sinyal {result['signal']}"
            )
        except Exception as exc:
            failures += 1
            print(f"Analisis gagal untuk {ticker}: {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Selesai. Berhasil dianalisis: {len(results)} | Gagal/tidak ada data: {failures}")
    if not results:
        print("Tidak ada hasil yang bisa dikirim.")
        return

    top = sorted(results, key=lambda r: (r["score"], r["volume_ratio"]), reverse=True)[:TOP_N]
    for i, r in enumerate(top, 1):
        print(f"{i}. {r['ticker']} | Skor {r['score']} | {r['signal']} | {rupiah(r['price'])}")

    if send_telegram(make_top_message(results)):
        print("TOP screener terkirim ke Telegram.")
    else:
        print("TOP screener tidak terkirim ke Telegram.")


if __name__ == "__main__":
    main()
