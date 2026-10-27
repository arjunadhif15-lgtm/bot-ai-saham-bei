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
# Filter likuiditas TOP 10 (estimasi nilai transaksi = Close x Volume)
MIN_AVG_TURNOVER20_IDR = 1_000_000_000  # harus di atas Rp1 miliar/hari
MIN_LATEST_TURNOVER_IDR = 5_000_000_000  # harus di atas Rp5 miliar pada candle terakhir
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

    # Validasi dasar OHLC: harga harus positif dan High tidak boleh di bawah Low.
    data = data[
        (data["High"] > 0) &
        (data["Low"] > 0) &
        (data["Close"] > 0) &
        (data["High"] >= data["Low"]) &
        (data["Volume"] >= 0) &
        (data["Close"] >= data["Low"] * 0.98) &
        (data["Close"] <= data["High"] * 1.02)
    ]

    if len(data) < 60:
        print(
            f"DATA BELUM CUKUP (kurang dari 60 candle): {ticker}"
        )
        return None

    # Deteksi anomali harga terbaru terhadap riwayat dekat.
    # Ini bukan pengganti pengecekan harga broker; hanya menolak data yang sangat janggal.
    recent_closes = data["Close"].tail(20)
    latest_close = float(data["Close"].iloc[-1])
    median_close = float(recent_closes.median()) if not recent_closes.empty else 0.0
    if median_close > 0 and not (0.35 <= latest_close / median_close <= 3.0):
        print(
            f"HARGA DITOLAK: {ticker} harga terakhir {latest_close:.2f} "
            f"terlalu jauh dari median 20 sesi {median_close:.2f}; "
            "periksa data Yahoo Finance."
        )
        return None

    return data


# ============================================================
# DETEKSI KOREKSI BERURUTAN
# ============================================================
def detect_current_correction_stage(close, high, low, ema50, atr14, lookback=90):
    """Deteksi koreksi aktif ke-1/ke-2 dalam struktur uptrend terkini.

    Koreksi dihitung setelah penurunan minimum dari puncak lokal dan dianggap
    selesai jika harga pulih/menembus kembali puncak sebelumnya. Ini pendekatan
    teknikal heuristik, bukan identifikasi swing yang pasti.
    """
    n = len(close)
    if n < 30 or atr14 <= 0:
        return 0, 0.0

    start = max(1, n - lookback)
    threshold_pct = max(0.025, min(0.05, 0.8 * atr14 / max(float(close.iloc[-1]), 0.01)))
    completed = 0
    peak = float(high.iloc[start - 1])
    active = False
    trough = float(low.iloc[start - 1])
    current_drawdown_pct = 0.0

    for i in range(start, n):
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])
        cl = float(close.iloc[i])
        ema50_i = float(ema50.iloc[i]) if pd.notna(ema50.iloc[i]) else np.nan

        if not active:
            if hi >= peak:
                peak = hi
            dd = (peak - cl) / peak if peak > 0 else 0.0
            # Hanya hitung koreksi yang tidak terlalu dalam; penurunan >12%
            # dianggap struktur berisiko/berpotensi perubahan tren, bukan pullback biasa.
            if threshold_pct <= dd <= 0.12:
                active = True
                trough = lo
                current_drawdown_pct = dd * 100
        else:
            trough = min(trough, lo)
            current_drawdown_pct = ((peak - cl) / peak) * 100 if peak > 0 else 0.0
            # Koreksi lebih dari 12% tidak diperlakukan sebagai pullback normal.
            if current_drawdown_pct > 12.0:
                active = False
                completed = 0
                peak = hi
                trough = lo
                current_drawdown_pct = 0.0
            # Koreksi selesai saat penutupan kembali ke puncak sebelumnya.
            elif cl >= peak * 1.002:
                completed += 1
                active = False
                peak = max(hi, cl)
                trough = lo
                current_drawdown_pct = 0.0
            # Tren dianggap rusak bila penutupan jatuh jauh di bawah EMA50.
            elif pd.notna(ema50_i) and cl < ema50_i * 0.95:
                active = False
                completed = 0
                peak = hi
                trough = lo
                current_drawdown_pct = 0.0

    if not active:
        return 0, 0.0
    return min(completed + 1, 3), max(0.0, current_drawdown_pct)


def build_entry_zones(price, atr14, ema9_now, ema21_now,
                      support20, fib382, fib500, fib618):
    """Hitung dua zona harga indikatif: koreksi pertama dan koreksi kedua.

    Zona tetap ditampilkan meski koreksi belum aktif agar pengguna dapat
    menyiapkan level pantauan. Zona bukan sinyal beli; konfirmasi pantulan
    tetap diperlukan.
    """
    price = float(price)
    atr14 = float(atr14) if np.isfinite(atr14) and atr14 > 0 else price * 0.03
    tolerance = max(0.35 * atr14, price * 0.005)
    levels = [
        ("EMA9", ema9_now), ("EMA21", ema21_now),
        ("Support20", support20), ("Fib38.2%", fib382),
        ("Fib50%", fib500), ("Fib61.8%", fib618),
    ]

    # Hanya gunakan level di bawah harga sekarang dan dalam rentang yang
    # masih relevan. Level yang jauh di atas harga bukan support pullback.
    candidates = []
    for name, value in levels:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if (np.isfinite(value) and value > 0
                and value < price
                and value >= price * 0.80):
            candidates.append((name, value))

    # Kelompokkan level teknikal yang berdekatan menjadi zona konfluensi.
    groups = []
    unused = candidates[:]
    while unused:
        anchor_name, anchor_value = max(unused, key=lambda item: item[1])
        group = [(name, value) for name, value in unused
                 if abs(value - anchor_value) <= tolerance]
        groups.append(group)
        group_keys = {(name, value) for name, value in group}
        unused = [item for item in unused if item not in group_keys]

    clusters = []
    for group in groups:
        center = float(np.mean([value for _, value in group]))
        names = "/".join(name for name, _ in group)
        clusters.append((center, names, len(group) >= 2))
    clusters.sort(key=lambda item: item[0], reverse=True)

    # Hindari dua entry yang sebenarnya merupakan zona yang sama.
    distinct = []
    min_separation = max(0.65 * atr14, price * 0.008)
    for cluster in clusters:
        if all(abs(cluster[0] - old[0]) >= min_separation for old in distinct):
            distinct.append(cluster)

    # Entry 1 adalah zona paling dekat di bawah harga sekarang.
    if distinct:
        e1_anchor, e1_names, e1_confluence = distinct[0]
        e1_source = ("Konfluensi " if e1_confluence else "Level tunggal ") + e1_names
    else:
        # Fallback terukur agar rentang tetap tersedia, tanpa menyebutnya sinyal.
        e1_anchor = min(ema21_now, price - 0.35 * atr14)
        e1_anchor = max(price * 0.85, min(float(e1_anchor), price - 0.05 * atr14))
        e1_source = "zona estimasi EMA21/ATR; konfluensi belum cukup"

    # Entry 2 memakai cluster lebih rendah yang berbeda dari Entry 1.
    lower_clusters = [c for c in distinct if c[0] < e1_anchor - min_separation * 0.5]
    if lower_clusters:
        e2_anchor, e2_names, e2_confluence = lower_clusters[0]
        e2_source = ("Konfluensi " if e2_confluence else "Level tunggal ") + e2_names
    else:
        deeper_candidates = [
            float(v) for v in (support20, fib618, ema21_now)
            if np.isfinite(v) and 0 < float(v) < e1_anchor - min_separation * 0.5
        ]
        if deeper_candidates:
            e2_anchor = max(deeper_candidates)
            e2_source = "zona support/Fibonacci berikutnya"
        else:
            e2_anchor = e1_anchor - max(atr14, price * 0.015)
            e2_source = "zona estimasi lebih dalam berbasis ATR"
        e2_anchor = max(price * 0.80, min(e2_anchor, e1_anchor - min_separation * 0.5))

    width = max(0.18 * atr14, price * 0.003)
    e1_low = max(price * 0.80, e1_anchor - width)
    e1_high = min(price * 0.999, e1_anchor + width)
    e2_low = max(price * 0.80, e2_anchor - width)
    e2_high = min(e1_low - 0.01, e2_anchor + width)

    # Pastikan batas bawah/atas positif dan berurutan setelah pembatasan.
    e1_low = max(0.01, min(e1_low, e1_high))
    e2_low = max(0.01, min(e2_low, e2_high))
    if e2_high <= e2_low:
        e2_high = e2_low + max(0.01, price * 0.002)

    return (
        float(e1_low), float(e1_high), e1_source,
        float(e2_low), float(e2_high), e2_source,
    )


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

    # Estimasi nilai transaksi harian (bukan turnover resmi bursa): Close x Volume.
    # Rata-rata 20 sesi dihitung dari nilai transaksi setiap candle.
    daily_turnover = (close * volume).replace([np.inf, -np.inf], np.nan)
    latest_turnover = (
        float(daily_turnover.iloc[-1])
        if len(daily_turnover) and pd.notna(daily_turnover.iloc[-1])
        else np.nan
    )
    avg_turnover20 = (
        float(daily_turnover.tail(20).mean())
        if len(daily_turnover.tail(20)) == 20 and daily_turnover.tail(20).notna().all()
        else np.nan
    )
    liquidity_filter_pass = bool(
        np.isfinite(avg_turnover20)
        and np.isfinite(latest_turnover)
        and avg_turnover20 > MIN_AVG_TURNOVER20_IDR
        and latest_turnover > MIN_LATEST_TURNOVER_IDR
    )

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

    # False breakout diperiksa memakai resistance yang memang tersedia pada tiap candle,
    # bukan memakai resistance hari ini untuk menilai candle-candle lama (mengurangi bias).
    # Peringatan aktif jika ada penolakan breakout baru-baru ini dan harga saat ini masih
    # berada di bawah/tepat pada resistance.
    false_breakout = False
    breakout_confirmed = False
    prior_resistance_series = high.shift(1).rolling(20).max()
    if pd.notna(resistance20) and resistance20 > 0:
        rejected_today = today_high > resistance20 and price <= resistance20

        recent_rejection = False
        recent_start = max(0, len(data) - 3)
        for pos in range(recent_start, len(data)):
            candle_resistance = prior_resistance_series.iloc[pos]
            if pd.isna(candle_resistance) or candle_resistance <= 0:
                continue
            candle_high = float(high.iloc[pos])
            candle_close = float(close.iloc[pos])
            if candle_high > float(candle_resistance) and candle_close <= float(candle_resistance):
                recent_rejection = True
                break

        previous_resistance = prior_resistance_series.iloc[-2] if len(prior_resistance_series) >= 2 else np.nan
        returned_below = bool(
            pd.notna(previous_resistance)
            and previous_price > float(previous_resistance)
            and price <= resistance20
        )
        false_breakout = bool(
            rejected_today
            or returned_below
            or (recent_rejection and price <= resistance20)
        )

        # Label breakout yang lebih ketat: penutupan melewati resistance dengan buffer,
        # volume kuat, tren pendek naik, dan penutupan berada di bagian atas rentang candle.
        candle_range = max(today_high - float(low.iloc[-1]), 0.0)
        close_position = (price - float(low.iloc[-1])) / candle_range if candle_range > 0 else 0.0
        breakout_confirmed = bool(
            breakout20
            and price >= resistance20 * 1.003
            and volume_strong
            and trend_up
            and close_position >= 0.60
            and not false_breakout
        )

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
    elif breakout_confirmed:
        signal = "BREAKOUT TERKONFIRMASI"
    elif breakout20:
        signal = "BREAKOUT: TUNGGU KONFIRMASI"
    elif prebreakout:
        signal = "CALON BREAKOUT"
    elif accumulation:
        signal = "INDIKASI AKUMULASI"
    elif score >= MIN_SCORE and price_above_ema21 and volume_strong:
        signal = "KANDIDAT ENTRY"
    else:
        signal = "PANTAU DULU" if watch_only else "BELUM ADA SINYAL"

    # Penentuan is_candidate dipindahkan ke bawah, setelah semua variabel dihitung.

    # Entry 1 = zona indikatif koreksi pertama; Entry 2 = zona indikatif koreksi kedua.
    # Harga ditampilkan sejak awal agar pengguna bisa menyiapkan alarm harga.
    # Status tetap membedakan zona rencana dengan sinyal entry yang terkonfirmasi.
    support20 = float(low.tail(20).min()) if len(low) >= 20 else float(low.min())

    # Fibonacci dari swing low sebelum swing high terbaru.
    fib_window = min(60, len(data))
    high_window = high.tail(fib_window)
    low_window = low.tail(fib_window)
    swing_high_idx = high_window.idxmax()
    swing_high = float(high_window.loc[swing_high_idx])
    low_before_high = low_window.loc[:swing_high_idx]
    swing_low = float(low_before_high.min()) if len(low_before_high) else float(low_window.min())
    swing_range = swing_high - swing_low
    if swing_range > 0 and swing_high_idx != low_window.index[0]:
        fib382 = swing_high - (0.382 * swing_range)
        fib500 = swing_high - (0.500 * swing_range)
        fib618 = swing_high - (0.618 * swing_range)
    else:
        fib382 = fib500 = fib618 = np.nan

    correction_stage, correction_drawdown_pct = detect_current_correction_stage(
        close, high, low, ema50, atr14
    )
    (
        entry1_low, entry1_high, entry1_source,
        entry2_low, entry2_high, entry2_source,
    ) = build_entry_zones(
        price, atr14, ema9_now, ema21_now, support20, fib382, fib500, fib618
    )

    # Keterangan singkat untuk Telegram.
    structure_ok = bool(trend_medium_up and price >= ema50_now * 0.97 and not false_breakout)
    if false_breakout:
        entry_status = "Jangan entry dulu: false breakout"
        entry2_status = "Tunggu tren kembali kuat"
    elif not structure_ok:
        entry_status = "Zona rencana saja; tren belum kuat"
        entry2_status = "Belum aktif; cek ulang support"
    elif correction_stage == 1:
        entry_status = "Koreksi 1 terdeteksi; tunggu pantulan"
        entry2_status = "Belum aktif; tunggu koreksi berikutnya"
    elif correction_stage == 2:
        entry_status = "Zona pantauan; bukan sinyal beli baru"
        entry2_status = "Koreksi 2 terdeteksi; tunggu pantulan"
    elif correction_stage >= 3:
        entry_status = "Koreksi sudah lanjut; evaluasi ulang"
        entry2_status = "Jangan buru-buru entry"
    else:
        entry_status = "Rencana saja; tunggu harga masuk dan memantul"
        entry2_status = "Belum aktif; tunggu koreksi lanjutan"

    # Field lama dipertahankan agar fungsi lain tetap kompatibel.
    entry_low = entry1_low if np.isfinite(entry1_low) else entry2_low
    entry_high = entry1_high if np.isfinite(entry1_high) else entry2_high

    # Stop loss berbasis zona Entry 2 dan support terdekat yang masih relevan.
    # Support yang terlalu jauh tidak dipakai agar risiko tidak membengkak tanpa alasan.
    sl_buffer = max(0.30 * atr14, price * 0.005)
    support_candidates = []
    for level in (support20, fib618, ema21_now, ema50_now):
        try:
            level = float(level)
        except (TypeError, ValueError):
            continue
        if np.isfinite(level) and 0 < level < entry2_low:
            support_candidates.append(level)
    nearby_supports = [s for s in support_candidates if (entry2_low - s) <= 2.0 * atr14]
    structural_stop_base = max(nearby_supports) if nearby_supports else entry2_low
    stop_loss = max(0.0, min(entry2_low, structural_stop_base) - sl_buffer)

    # Target memakai resistance historis di atas harga jika ada; jika tidak,
    # gunakan fallback berbasis ATR. Target harus tetap di atas harga saat scan.
    historical_highs = high.iloc[:-1].tail(60) if len(high) > 1 else high.iloc[:0]
    resistance_candidates = sorted({
        float(v) for v in historical_highs
        if pd.notna(v) and float(v) >= price + 0.80 * atr14
    })
    if resistance_candidates:
        target_1 = resistance_candidates[0]
    else:
        target_1 = price + 1.8 * atr14
    if target_1 <= price:
        target_1 = price + 1.8 * atr14

    next_resistances = [r for r in resistance_candidates if r >= target_1 + 0.50 * atr14]
    if next_resistances:
        target_2 = next_resistances[0]
    else:
        target_2 = max(price + 3.0 * atr14, target_1 + 0.75 * atr14)
    if target_2 <= target_1:
        target_2 = target_1 + max(0.50 * atr14, price * 0.01)

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

    # Rasio risiko/imbalan dihitung konservatif dari titik tengah Entry 1.
    entry1_mid = (entry1_low + entry1_high) / 2 if np.isfinite(entry1_low) and np.isfinite(entry1_high) else price
    risk_per_share = max(entry1_mid - stop_loss, 0.0)
    reward_per_share = max(target_1 - entry1_mid, 0.0)
    reward_risk = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    entry2_mid = (entry2_low + entry2_high) / 2 if np.isfinite(entry2_low) and np.isfinite(entry2_high) else np.nan
    entry2_risk = max(entry2_mid - stop_loss, 0.0) if np.isfinite(entry2_mid) else 0.0
    entry2_reward = max(target_1 - entry2_mid, 0.0) if np.isfinite(entry2_mid) else 0.0
    reward_risk_entry2 = entry2_reward / entry2_risk if entry2_risk > 0 else 0.0
    risk_reward_ok = bool(reward_risk >= MIN_REWARD_RISK and target_1 > entry1_mid > stop_loss)
    risk_budget = DEFAULT_CAPITAL * MAX_RISK_PER_TRADE_PCT / 100
    risk_lot = int(risk_budget // (risk_per_share * 100)) if risk_per_share > 0 else 0
    capital_lot = int(DEFAULT_CAPITAL // (entry1_mid * 100)) if entry1_mid > 0 else 0
    suggested_lot = max(0, min(risk_lot, capital_lot))
    capital_used = suggested_lot * entry1_mid * 100
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
        "latest_turnover_idr": float(latest_turnover) if np.isfinite(latest_turnover) else np.nan,
        "avg_turnover20_idr": float(avg_turnover20) if np.isfinite(avg_turnover20) else np.nan,
        "liquidity_filter_pass": bool(liquidity_filter_pass),
        "ema9": ema9_now,
        "ema21": ema21_now,
        "ema50": ema50_now,
        "breakout20": bool(breakout20),
        "breakout_confirmed": bool(breakout_confirmed),
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
        "reward_risk_entry2": float(reward_risk_entry2),
        "risk_reward_ok": bool(risk_reward_ok),
        "risk_filter_pass": bool(risk_filter_pass),
        "suggested_lot": int(suggested_lot),
        "capital_used": float(capital_used),
        "estimated_trade_risk": float(estimated_trade_risk),
        "trend_up": bool(trend_up),
        "volume_strong": bool(volume_strong),
        "entry_low": float(entry_low) if np.isfinite(entry_low) else np.nan,
        "entry_high": float(entry_high) if np.isfinite(entry_high) else np.nan,
        "entry1_low": float(entry1_low) if np.isfinite(entry1_low) else np.nan,
        "entry1_high": float(entry1_high) if np.isfinite(entry1_high) else np.nan,
        "entry2_low": float(entry2_low) if np.isfinite(entry2_low) else np.nan,
        "entry2_high": float(entry2_high) if np.isfinite(entry2_high) else np.nan,
        "support20": float(support20),
        "fib382": float(fib382) if np.isfinite(fib382) else np.nan,
        "fib500": float(fib500) if np.isfinite(fib500) else np.nan,
        "fib618": float(fib618) if np.isfinite(fib618) else np.nan,
        "entry_status": entry_status,
        "entry2_status": entry2_status,
        "entry1_source": entry1_source,
        "entry2_source": entry2_source,
        "correction_stage": int(correction_stage),
        "correction_drawdown_pct": float(correction_drawdown_pct),
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

        f"Status: "
        f"{result['entry_status']}\n\n"

        f"SL: "
        f"{format_rupiah(result['stop_loss'])}\n"

        f"T1: "
        f"{format_rupiah(result['target_1'])}\n"

        f"T2: "
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



def make_top10_telegram_messages(results):
    """Buat pesan TOP 10 Telegram dengan format sederhana dan baris baru asli."""
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
    header = [
        f"🏆 TOP {len(ranked)} SAHAM BEI",
        f"📅 {now:%d-%m-%Y} | {now:%H:%M} WIB",
    ]

    divider = "━━━━━━━━━━━━━━"
    blocks = []

    for i, r in enumerate(ranked, start=1):
        warning = " ⚠️ FALSE BREAKOUT" if r.get("false_breakout", False) else ""

        e1l = price_or_wait(r.get("entry1_low", r.get("entry_low")))
        e1h = price_or_wait(r.get("entry1_high", r.get("entry_high")))
        entry1 = (
            f"{e1l}–{e1h.removeprefix('Rp')}"
            if e1l != "Belum tersedia" and e1h != "Belum tersedia"
            else "Belum tersedia"
        )

        e2l = price_or_wait(r.get("entry2_low"))
        e2h = price_or_wait(r.get("entry2_high"))
        entry2 = (
            f"{e2l}–{e2h.removeprefix('Rp')}"
            if e2l != "Belum tersedia" and e2h != "Belum tersedia"
            else "Belum tersedia"
        )

        blocks.append(chr(10).join([
            divider,
            f"{i}. {r.get('ticker', '?')} | Skor {r.get('score', 0)}/100{warning}",
            "",
            "💰 HARGA",
            f"{price_or_wait(r.get('price'))} — {r.get('signal', 'PANTAU')}",
            "",
            "🎯 RENCANA BELI",
            f"Entry 1 : {entry1}",
            f"Entry 2 : {entry2}",
            f"Status E1: {r.get('entry_status', 'Tunggu konfirmasi')}",
            f"Status E2: {r.get('entry2_status', 'Belum aktif')}",
            "",
            "📍 LEVEL PENTING",
            f"Support : {price_or_wait(r.get('support20'))}",
            f"Fibo 38%: {price_or_wait(r.get('fib382'))}",
            f"Fibo 50%: {price_or_wait(r.get('fib500'))}",
            f"Fibo 62%: {price_or_wait(r.get('fib618'))}",
            "",
            "🛡️ RISIKO & TARGET",
            f"Stop Loss : {price_or_wait(r.get('stop_loss'))}",
            f"Target 1  : {price_or_wait(r.get('target_1'))}",
            f"Target 2  : {price_or_wait(r.get('target_2'))}",
            f"Risk/Reward E1: {r.get('reward_risk', 0):.2f}x | E2: {r.get('reward_risk_entry2', 0):.2f}x",
            f"Kelayakan R/R: {'Memenuhi batas' if r.get('risk_reward_ok', False) else 'Risiko/imbalan belum memenuhi batas'}",
            "",
            "📊 PENILAIAN",
            f"Momentum : {r.get('momentum_score', 0)}/100",
            f"Akumulasi: {r.get('accumulation_score', 0)}/100",
        ]))

    footer = chr(10).join([
        "ℹ️ Bukan perintah beli.",
        "Zona entry perlu pantulan/konfirmasi; cek harga aktual.",
        "Target memakai resistance historis bila tersedia; selain itu estimasi ATR.",
        divider,
    ])

    messages = []
    current_blocks = []
    # Batas internal 3.400 karakter agar aman dari batas pesan Telegram.
    for block in blocks:
        candidate = (chr(10) * 2).join(header + current_blocks + [block, footer])
        if len(candidate) > 3400 and current_blocks:
            messages.append((chr(10) * 2).join(header + current_blocks + [footer]))
            current_blocks = [block]
        else:
            current_blocks.append(block)

    if current_blocks or not messages:
        messages.append((chr(10) * 2).join(header + current_blocks + [footer]))
    return messages

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
            f"Turnover={format_rupiah(result.get('latest_turnover_idr', 0))} | "
            f"Avg20={format_rupiah(result.get('avg_turnover20_idr', 0))} | "
            f"Likuiditas={'LOLOS' if result.get('liquidity_filter_pass', False) else 'GAGAL'} | "
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

    # Filter likuiditas diterapkan sebelum pemilihan TOP 10; ranking tetap sama.
    eligible_liquidity = [r for r in all_results if r.get("liquidity_filter_pass", False)]
    print(
        f"Lolos filter likuiditas: {len(eligible_liquidity)} dari {len(all_results)} saham "
        f"(rata-rata 20 sesi > Rp{MIN_AVG_TURNOVER20_IDR:,.0f}/hari dan "
        f"nilai transaksi candle terakhir > Rp{MIN_LATEST_TURNOVER_IDR:,.0f})."
    )
    if eligible_liquidity:
        top_ranked = sorted(
            eligible_liquidity,
            key=lambda r: (
                r.get("score", 0),
                r.get("momentum_score", 0),
                r.get("accumulation_score", 0),
            ),
            reverse=True,
        )[:TOP_N]
        print(
            f"Mengirim TOP {len(top_ranked)} saham yang lolos filter likuiditas "
            "(ranking teknikal → momentum → akumulasi tetap)."
        )
        telegram_messages = make_top10_telegram_messages(top_ranked)
        sent_count = 0
        for part_no, message in enumerate(telegram_messages, start=1):
            if send_telegram(message):
                sent_count += 1
                print(f"NOTIFIKASI TOP 10 BAGIAN {part_no}/{len(telegram_messages)} TERKIRIM.")
            else:
                print(f"NOTIFIKASI TOP 10 BAGIAN {part_no}/{len(telegram_messages)} GAGAL dikirim.")
        notifications_sent = sent_count
        if sent_count == len(telegram_messages):
            print(f"NOTIFIKASI TOP 10 TERKIRIM LENGKAP: {sent_count} pesan.")
        else:
            print(f"NOTIFIKASI TOP 10 BELUM LENGKAP: {sent_count}/{len(telegram_messages)} pesan.")
    else:
        print("Tidak ada saham yang memenuhi kedua filter likuiditas; TOP 10 tidak dikirim.")

    print(f"Pesan Telegram terkirim kali ini: {notifications_sent}")
    print("=" * 60)
    if RUN_BACKTEST:
        run_backtest(tickers)


if __name__ == "__main__":
    main()
