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

MIN_SCORE = 75
MIN_VOLUME_RATIO = 1.2
MAX_RSI_ENTRY = 70.0  # RSI di atas ini diberi label PANTAU DULU
MAX_ABOVE_EMA9_PCT = 3.0  # Harga >3% di atas EMA9 diberi label PANTAU DULU
MAX_TICKERS = None  # None = pindai semua saham di DEFAULT_TICKERS
REQUEST_DELAY_SECONDS = 0.25
YF_PERIOD = "1y"
YF_INTERVAL = "1d"
TIMEZONE = "Asia/Jakarta"

# Fitur tambahan F-M
TOP_N = 10
MAX_RISK_PER_TRADE_PCT = 1.0
MIN_REWARD_RISK = 1.5
DEFAULT_CAPITAL = 20_000_000
RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0").strip() == "1"
BACKTEST_TICKERS_LIMIT = 30
BACKTEST_FORWARD_DAYS = 5


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
# MODE UJI: NOTIFIKASI BOLEH BERULANG SETIAP KALI SCANNER DIJALANKAN
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

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

    # Volume hari ini dibanding rata-rata 20 hari.
    avg_volume20 = volume.rolling(20).mean()
    avg_vol_now = float(avg_volume20.iloc[-1]) if pd.notna(avg_volume20.iloc[-1]) else 0.0
    volume_ratio = float(volume.iloc[-1] / avg_vol_now) if avg_vol_now > 0 else 0.0

    # Resistance 20 hari: tertinggi 20 candle SEBELUM candle terakhir.
    prior_high20_series = high.shift(1).rolling(20).max()
    resistance20 = float(prior_high20_series.iloc[-1]) if pd.notna(prior_high20_series.iloc[-1]) else np.nan

    price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2])
    today_high = float(high.iloc[-1])
    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])

    trend_up = ema9_now > ema21_now
    trend_medium_up = ema21_now > ema50_now
    price_above_ema21 = price > ema21_now
    price_above_ema9 = price > ema9_now
    distance_above_ema9_pct = ((price / ema9_now) - 1) * 100 if ema9_now > 0 else 0.0

    breakout20 = bool(pd.notna(resistance20) and price > resistance20)
    near_resistance_pct = (
        ((resistance20 - price) / resistance20) * 100
        if pd.notna(resistance20) and resistance20 > 0
        else np.nan
    )
    # Calon breakout: harga belum menembus resistance, tetapi maksimal 3% di bawahnya.
    prebreakout = bool(
        pd.notna(near_resistance_pct)
        and 0 <= near_resistance_pct <= 3.0
        and price_above_ema21
        and volume_ratio >= 0.8
        and rsi_now >= 45
    )

    volume_strong = volume_ratio >= MIN_VOLUME_RATIO

    # Indikasi akumulasi (bukan bukti aktivitas bandar):
    # harga relatif datar 10 sesi, volume naik, dan harga bertahan di atas EMA21.
    change_10d_pct = (
        ((price / float(close.iloc[-11])) - 1) * 100
        if len(close) >= 11 and float(close.iloc[-11]) > 0
        else 999.0
    )
    avg_volume5 = float(volume.rolling(5).mean().iloc[-1]) if pd.notna(volume.rolling(5).mean().iloc[-1]) else 0.0
    accumulation = bool(
        abs(change_10d_pct) <= 4.0
        and avg_vol_now > 0
        and avg_volume5 >= avg_vol_now * 1.15
        and price_above_ema21
        and not (rsi_now > 75)
    )

    # Skor indikasi akumulasi 0-100; ini hanya proksi harga/volume,
    # bukan bukti bahwa bandar atau investor tertentu sedang membeli.
    accumulation_score = 0

    # Stabilitas harga 10 sesi: maksimal 25 poin.
    if abs(change_10d_pct) <= 2.0:
        accumulation_score += 25
    elif abs(change_10d_pct) <= 4.0:
        accumulation_score += 18
    elif abs(change_10d_pct) <= 6.0:
        accumulation_score += 8

    # Volume rata-rata 5 sesi dibanding rata-rata 20 sesi: maksimal 25.
    volume_lift = avg_volume5 / avg_vol_now if avg_vol_now > 0 else 0.0
    if volume_lift >= 1.5:
        accumulation_score += 25
    elif volume_lift >= 1.25:
        accumulation_score += 20
    elif volume_lift >= 1.10:
        accumulation_score += 12
    elif volume_lift >= 1.0:
        accumulation_score += 5

    # Posisi harga relatif terhadap EMA21: maksimal 20 poin.
    if price_above_ema21:
        accumulation_score += 20
    elif price >= ema21_now * 0.98:
        accumulation_score += 10

    # Rasio sesi yang tidak ditutup turun dalam 9 perubahan terakhir: maksimal 15.
    recent_change = close.iloc[-10:].diff().dropna()
    if len(recent_change) > 0:
        non_down_ratio = float((recent_change >= 0).sum()) / len(recent_change)
        if non_down_ratio >= 0.65:
            accumulation_score += 15
        elif non_down_ratio >= 0.50:
            accumulation_score += 8

    # Konfirmasi tren jangka pendek: maksimal 15 poin.
    if trend_up and price_above_ema9:
        accumulation_score += 15
    elif price_above_ema9:
        accumulation_score += 8

    accumulation_score = min(100, int(accumulation_score))
    accumulation_strength = (
        "KUAT" if accumulation_score >= 75
        else "SEDANG" if accumulation_score >= 55
        else "LEMAH"
    )

    # False breakout: candle sempat melewati resistance, tetapi ditutup kembali di bawahnya,
    # atau salah satu dari 3 candle terakhir menembus resistance lalu harga kini kembali di bawah.
    false_breakout = False
    if pd.notna(resistance20) and resistance20 > 0:
        rejected_today = today_high > resistance20 and price <= resistance20
        recent_highs = high.iloc[-3:]
        recent_closes = close.iloc[-3:]
        recent_rejection = any(
            float(h) > resistance20 and float(c) <= resistance20
            for h, c in zip(recent_highs, recent_closes)
        )
        returned_below = previous_price > resistance20 and price <= resistance20
        false_breakout = bool(rejected_today or recent_rejection or returned_below)

    overheated_rsi = rsi_now > MAX_RSI_ENTRY
    price_stretched = distance_above_ema9_pct > MAX_ABOVE_EMA9_PCT
    watch_only = overheated_rsi or price_stretched

    # ATR 14
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_value = true_range.rolling(14).mean().iloc[-1]
    atr14 = float(atr_value) if pd.notna(atr_value) and float(atr_value) > 0 else price * 0.03

    # Skor berbobot dengan maksimum tepat 100.
    score = 0
    if trend_up:
        score += 15
    if trend_medium_up:
        score += 10
    if breakout20:
        score += 20
    if prebreakout:
        score += 15
    if volume_strong:
        score += 15
    if 50 <= rsi_now <= 65:
        score += 10
    elif 45 <= rsi_now < 50 or 65 < rsi_now <= 70:
        score += 5
    if price_above_ema9:
        score += 5
    if accumulation:
        score += 5
    if breakout20 and volume_strong and trend_up and 45 <= rsi_now <= 70:
        score += 5
    score = min(100, int(score))

    # Kategori sinyal; false breakout diberi prioritas sebagai peringatan risiko.
    if false_breakout:
        signal = "WASPADA FALSE BREAKOUT"
    elif breakout20 and volume_strong and trend_up:
        signal = "BREAKOUT TERKONFIRMASI"
    elif prebreakout:
        signal = "CALON BREAKOUT"
    elif accumulation:
        signal = "INDIKASI AKUMULASI"
    elif score >= MIN_SCORE and price_above_ema21 and volume_strong:
        signal = "KANDIDAT ENTRY"
    else:
        signal = "PANTAU DULU" if watch_only else "BELUM ADA SINYAL"

    # Penentuan is_candidate dipindahkan ke bawah, setelah semua variabel dihitung.

    # Area entry indikatif. Hindari mengejar harga saat RSI/harga terlalu tinggi.
    entry_low = np.nan
    entry_high = np.nan
    if not watch_only and not false_breakout:
        if breakout20:
            entry_low = price
            entry_high = price + (0.25 * atr14)
            entry_status = "Breakout terdeteksi; verifikasi harga dan volume terkini"
        elif prebreakout:
            entry_low = max(ema21_now, price - (0.5 * atr14))
            entry_high = min(resistance20, price + (0.25 * atr14)) if pd.notna(resistance20) else price
            entry_status = "Calon breakout; tunggu konfirmasi tembus resistance"
        elif price_above_ema21 and volume_strong:
            entry_low = max(ema21_now, price - (0.5 * atr14))
            entry_high = price
            entry_status = "Belum breakout; tunggu konfirmasi dan pullback"
        else:
            entry_status = "Belum ada area entry yang memenuhi syarat"
    elif false_breakout:
        entry_status = "Hindari entry dahulu; harga menunjukkan risiko false breakout"
    else:
        entry_status = "PANTAU DULU — jangan mengejar harga; tunggu kondisi lebih ideal"

    stop_loss = max(0.0, price - (1.5 * atr14))
    target_1 = price + (2.3 * atr14)
    target_2 = price + (3 * atr14)

    # G: momentum awal sebelum breakout
    momentum_5d_pct = (price / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 and float(close.iloc[-6]) > 0 else 0.0
    momentum_10d_pct = (price / float(close.iloc[-11]) - 1) * 100 if len(close) >= 11 and float(close.iloc[-11]) > 0 else 0.0
    momentum_early = bool(not breakout20 and not false_breakout and price_above_ema21 and trend_up and momentum_5d_pct > 0 and pd.notna(near_resistance_pct) and 0 <= near_resistance_pct <= 5.0 and volume_ratio >= 0.8 and 45 <= rsi_now <= 70)
    momentum_score = 0
    if momentum_5d_pct > 0: momentum_score += 25
    if momentum_10d_pct > 0: momentum_score += 15
    if trend_up: momentum_score += 20
    if price_above_ema21: momentum_score += 15
    if volume_ratio >= 1.0: momentum_score += 10
    if pd.notna(near_resistance_pct) and 0 <= near_resistance_pct <= 5: momentum_score += 15
    momentum_score = min(100, momentum_score)

    # H/K: risk/reward dan ukuran posisi simulasi berbasis modal Rp20 juta.
    risk_per_share = max(price - stop_loss, 0.0)
    reward_per_share = max(target_1 - price, 0.0)
    reward_risk = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    risk_reward_ok = bool(reward_risk >= MIN_REWARD_RISK)
    risk_budget = DEFAULT_CAPITAL * MAX_RISK_PER_TRADE_PCT / 100
    risk_lot = int(risk_budget // (risk_per_share * 100)) if risk_per_share > 0 else 0
    capital_lot = int(DEFAULT_CAPITAL // (price * 100)) if price > 0 else 0
    suggested_lot = max(0, min(risk_lot, capital_lot))
    capital_used = suggested_lot * price * 100
    estimated_trade_risk = suggested_lot * risk_per_share * 100
    risk_filter_pass = bool(risk_reward_ok and not false_breakout and not watch_only)

    # Kandidat dihitung setelah momentum_early dan risk_filter_pass tersedia.
    # Mode notifikasi berulang: kandidat dapat dikirim lagi pada setiap pemindaian.
    is_candidate = bool(
        (
            not false_breakout
            and score >= MIN_SCORE
            and risk_filter_pass
            and (
                breakout20
                or prebreakout
                or accumulation
                or momentum_early
                or (price_above_ema21 and volume_strong)
            )
        )

    )

    return {
        "ticker": ticker,
        "price": price,
        "score": score,
        "signal": signal,
        "is_candidate": is_candidate,
        "rsi": rsi_now,
        "distance_above_ema9_pct": float(distance_above_ema9_pct),
        "watch_only": bool(watch_only),
        "volume_ratio": float(volume_ratio),
        "ema9": ema9_now,
        "ema21": ema21_now,
        "ema50": ema50_now,
        "breakout20": bool(breakout20),
        "prebreakout": bool(prebreakout),
        "accumulation": bool(accumulation),
        "accumulation_score": int(accumulation_score),
        "accumulation_strength": accumulation_strength,
        "false_breakout": bool(false_breakout),
        "resistance20": float(resistance20) if pd.notna(resistance20) else np.nan,
        "distance_to_resistance_pct": float(near_resistance_pct) if pd.notna(near_resistance_pct) else np.nan,
        "change_10d_pct": float(change_10d_pct),
        "momentum_5d_pct": float(momentum_5d_pct),
        "momentum_10d_pct": float(momentum_10d_pct),
        "momentum_early": bool(momentum_early),
        "momentum_score": int(momentum_score),
        "reward_risk": float(reward_risk),
        "risk_reward_ok": bool(risk_reward_ok),
        "risk_filter_pass": bool(risk_filter_pass),
        "suggested_lot": int(suggested_lot),
        "capital_used": float(capital_used),
        "estimated_trade_risk": float(estimated_trade_risk),
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

    heading = f"🔎 {result['signal']} — {result['ticker']}"

    return (
        f"{heading}\n"

        f"Skor teknikal: "
        f"{result['score']}/100\n"

        f"Skor momentum awal: {result.get('momentum_score', 0)}/100\n"
        f"Skor akumulasi: {result.get('accumulation_score', 0)}/100 ({result.get('accumulation_strength', 'LEMAH')})\n"

        f"Harga penutupan data harian: "
        f"{format_rupiah(result['price'])}\n"

        f"RSI(14): "
        f"{result['rsi']:.1f}\n"

        f"Jarak harga dari EMA9: "
        f"{result.get('distance_above_ema9_pct', 0):.1f}%\n"

        f"Rasio volume: "
        f"{result['volume_ratio']:.2f}x\n"

        f"Breakout 20 hari: "
        f"{breakout_text}\n"

        f"Jarak ke resistance: "
        f"{result.get('distance_to_resistance_pct', float('nan')):.1f}%\n"

        f"Indikasi akumulasi: "
        f"{'Ya' if result.get('accumulation', False) else 'Belum kuat'}\n"

        f"Skor indikasi akumulasi: "
        f"{result.get('accumulation_score', 0)}/100 "
        f"({result.get('accumulation_strength', 'LEMAH')})\n"

        f"Perubahan harga 10 sesi: "
        f"{result.get('change_10d_pct', 0):.2f}%\n"

        f"Risiko false breakout: "
        f"{'TERDETEKSI' if result.get('false_breakout', False) else 'Tidak terdeteksi'}\n\n"

        f"🎯 RENCANA TRANSAKSI INDIKATIF\n"

        f"Area entry: {entry_text}\n"

        f"Status entry: "
        f"{result['entry_status']}\n\n"

        f"Stop loss indikatif: "
        f"{format_rupiah(result['stop_loss'])}\n"

        f"Target 1 indikatif: "
        f"{format_rupiah(result['target_1'])}\n"

        f"Target 2 indikatif: "
        f"{format_rupiah(result['target_2'])}\n"
        f"Reward/risk ke Target 1: {result.get('reward_risk', 0):.2f}x\n"
        f"Lot simulasi (modal Rp{DEFAULT_CAPITAL:,.0f}, risiko max {MAX_RISK_PER_TRADE_PCT:.1f}%): {result.get('suggested_lot', 0)} lot\n"
        f"Perkiraan modal terpakai: {format_rupiah(result.get('capital_used', 0))}\n"
        f"Estimasi risiko ke stop loss: {format_rupiah(result.get('estimated_trade_risk', 0))}\n\n"

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
# J: BACKTESTING HISTORIS SEDERHANA
# ============================================================
def run_backtest(tickers):
    print("\n" + "=" * 60)
    print("BACKTESTING HISTORIS SEDERHANA")
    print("Simulasi kasar: belum memasukkan biaya transaksi/slippage.")
    tested = wins = losses = skipped = 0
    returns = []
    sample = tickers[:BACKTEST_TICKERS_LIMIT]
    for idx, ticker in enumerate(sample, start=1):
        print(f"[BT {idx}/{len(sample)}] {ticker}", flush=True)
        data = fetch_history(ticker)
        if data is None or len(data) < 100:
            skipped += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        start_i = max(60, len(data) - 80)
        for i in range(start_i, len(data) - BACKTEST_FORWARD_DAYS):
            hist = data.iloc[:i + 1]
            try:
                res = analyze_stock(ticker, hist)
            except Exception:
                continue
            eligible = res["score"] >= MIN_SCORE and res["risk_filter_pass"] and not res["false_breakout"] and (res["breakout20"] or res["prebreakout"] or res["accumulation"] or res["momentum_early"])
            if not eligible:
                continue
            entry = float(hist["Close"].iloc[-1])
            future = data.iloc[i + 1:i + 1 + BACKTEST_FORWARD_DAYS]
            if future.empty or entry <= 0:
                continue
            pct = (float(future["Close"].iloc[-1]) / entry - 1) * 100
            returns.append(pct)
            tested += 1
            if pct > 0: wins += 1
            else: losses += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    print("-" * 60)
    print(f"Saham diperiksa       : {len(sample)}")
    print(f"Saham dilewati        : {skipped}")
    print(f"Sinyal historis diuji : {tested}")
    print(f"Hasil positif         : {wins}")
    print(f"Hasil nol/negatif     : {losses}")
    print(f"Win rate sederhana    : {wins / tested * 100 if tested else 0:.1f}%")
    print(f"Rata-rata perubahan   : {float(np.mean(returns)) if returns else 0:.2f}%")
    print("Catatan: belum memodelkan urutan stop loss/target intrahari.")
    print("=" * 60)


def print_top_lists(results):
    if not results:
        print("Belum ada hasil yang bisa dirangking.")
        return
    ranked = sorted(results, key=lambda r: (r.get("score", 0), r.get("momentum_score", 0), r.get("accumulation_score", 0)), reverse=True)
    print("\n" + "=" * 60)
    print(f"TOP {min(TOP_N, len(ranked))} SAHAM BERDASARKAN SKOR")
    print("Ticker | Skor | Momentum | Akumulasi | RSI | Volume | Sinyal")
    for r in ranked[:TOP_N]:
        print(f"{r['ticker']} | {r['score']} | {r.get('momentum_score', 0)} | {r.get('accumulation_score', 0)} | {r['rsi']:.1f} | {r['volume_ratio']:.2f}x | {r['signal']}")
    momentum = sorted([r for r in results if r.get("momentum_early")], key=lambda r: (r.get("momentum_score", 0), r.get("score", 0)), reverse=True)
    accum = sorted([r for r in results if r.get("accumulation_score", 0) >= 55], key=lambda r: (r.get("accumulation_score", 0), r.get("score", 0)), reverse=True)
    print("\nMOMENTUM AWAL: " + (", ".join(r["ticker"] for r in momentum[:10]) or "Tidak ada"))
    print("INDIKASI AKUMULASI: " + (", ".join(r["ticker"] for r in accum[:10]) or "Tidak ada"))
    print("=" * 60)



def make_top10_telegram_message(results):
    """Buat satu pesan Telegram TOP 10 yang ringkas dan mudah dibaca."""
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("score", 0),
            r.get("momentum_score", 0),
            r.get("accumulation_score", 0),
        ),
        reverse=True,
    )[:TOP_N]

    def price_or_wait(value):
        try:
            number = float(value)
            if np.isfinite(number) and number > 0:
                return format_rupiah(number)
        except (TypeError, ValueError):
            pass
        return "Belum tersedia"

    now = datetime.now(ZoneInfo(TIMEZONE))
    lines = [
        f"🏆 TOP {len(ranked)} SAHAM BEI",
        f"📅 {now:%d-%m-%Y %H:%M} WIB",
        "Urutan: skor teknikal → momentum → akumulasi",
        "",
    ]

    for i, r in enumerate(ranked, start=1):
        warning = " ⚠️ FALSE BREAKOUT" if r.get("false_breakout", False) else ""
        entry_low = price_or_wait(r.get("entry_low"))
        entry_high = price_or_wait(r.get("entry_high"))
        if entry_low != "Belum tersedia" and entry_high != "Belum tersedia":
            entry1 = f"{entry_low}–{entry_high}"
        else:
            entry1 = "Tunggu konfirmasi"

        entry_status = str(r.get("entry_status", "Perlu konfirmasi"))
        if r.get("false_breakout", False):
            entry2 = "Hindari dulu; risiko false breakout"
        elif r.get("watch_only", False):
            entry2 = "Tunggu harga lebih ideal"
        elif "pullback" in entry_status.lower() or "pantul" in entry_status.lower():
            entry2 = "Tunggu pullback/pantulan, jangan kejar harga"
        else:
            entry2 = "Tunggu konfirmasi dan pantulan support"

        lines.extend([
            f"{i}. {r.get('ticker', '?')} | Skor {r.get('score', 0)}/100{warning}",
            f"💰 Harga {price_or_wait(r.get('price'))} | {r.get('signal', 'PANTAU')}",
            f"🎯 Entry 1: {entry1}",
            f"🔄 Entry 2: {entry2}",
            f"🛑 SL: {price_or_wait(r.get('stop_loss'))} | 🎯 T1: {price_or_wait(r.get('target_1'))} | T2: {price_or_wait(r.get('target_2'))}",
            f"📊 Momentum {r.get('momentum_score', 0)}/100 | Akumulasi {r.get('accumulation_score', 0)}/100",
            "",
        ])

    lines.append(
        "ℹ️ TOP 10 adalah peringkat skor, bukan perintah beli. "
        "Data Yahoo Finance bisa terlambat. Entry 1/2, SL, dan target hanya perkiraan teknikal; "
        "cek harga terkini dan konfirmasi support sebelum transaksi."
    )
    message = "\n".join(lines)
    if len(message) > 4000:
        compact_lines = [line for line in lines if not line.startswith("📊 Momentum")]
        message = "\n".join(compact_lines)
    if len(message) > 4000:
        message = message[:3950] + "\n\nPesan dipersingkat agar sesuai batas Telegram."
    return message
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
        "Notifikasi: mode berulang aktif "
        "(setiap pemindaian jika memenuhi syarat)"
    )
    print("=" * 60)

    scanned = 0
    failures = 0
    candidates = 0
    notifications_sent = 0
    all_results = []

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
        all_results.append(result)

        print(
            f"  Harga="
            f"{format_rupiah(result['price'])} | "
            f"Skor={result['score']} | "
            f"Sinyal={result['signal']} | "
            f"RSI={result['rsi']:.1f} | "
            f"Volume={result['volume_ratio']:.2f}x | "
            f"Akumulasi={result.get('accumulation_score', 0)}/100 | "
            f"Momentum={result.get('momentum_score', 0)}/100 | "
            f"R/R={result.get('reward_risk', 0):.2f}x",
            flush=True,
        )

        if result["is_candidate"]:
            candidates += 1

        # Notifikasi ditunda sampai semua saham selesai dipindai.
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
        f"+ syarat teknikal: {candidates}"
    )
    print("=" * 60)
    print_top_lists(all_results)

    # Kirim maksimal 10 saham dengan skor tertinggi, walaupun kandidat entry = 0.
    if all_results:
        top_ranked = sorted(
            all_results,
            key=lambda r: (
                r.get("score", 0),
                r.get("momentum_score", 0),
                r.get("accumulation_score", 0),
            ),
            reverse=True,
        )[:TOP_N]
        print(
            f"Mengirim TOP {len(top_ranked)} saham terbaik ke Telegram "
            "(peringkat skor, bukan hanya kandidat entry)."
        )
        if send_telegram(make_top10_telegram_message(top_ranked)):
            notifications_sent = 1
            print("NOTIFIKASI TOP 10 TERKIRIM: satu pesan Telegram.")
        else:
            print("NOTIFIKASI TOP 10 GAGAL dikirim.")
    else:
        print("Tidak ada hasil analisis; TOP 10 tidak dikirim.")

    print(f"Pesan Telegram terkirim kali ini: {notifications_sent}")
    print("=" * 60)
    if RUN_BACKTEST:
        run_backtest(tickers)


if __name__ == "__main__":
    main()
