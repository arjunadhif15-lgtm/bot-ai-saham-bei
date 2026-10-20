import os
import ast
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
APP_FILE = os.path.join(BASE_DIR, "app.py")
STATE_FILE = os.path.join(BASE_DIR, "sent_notifications_v7.json")
TIMEZONE = ZoneInfo("Asia/Jakarta")
MIN_SCORE = 70
MIN_BREAKOUT_SCORE = 75
MIN_VOLUME_RATIO = 1.2
MIN_REWARD_RISK = 1.5
TOP_N = 10
REQUEST_DELAY_SECONDS = 0.25

# Hanya digunakan sebagai cadangan jika DEFAULT_TICKERS tidak dapat dibaca.
FALLBACK_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "BBTN", "ANTM", "MDKA", "BRMS",
    "PTBA", "ADRO", "ITMG", "INCO", "UNTR", "ASII", "TLKM", "GOTO",
    "BRPT", "BREN", "DEWA", "BUKA", "WOOD", "KPIG", "BBYB", "COCO",
    "FORE", "NCKL", "EMTK", "AVIA", "VKTR", "IOTF"
]


def _normalize_tickers(raw):
    if isinstance(raw, str):
        items = raw.replace(",", " ").split()
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []

    tickers = []
    for item in items:
        ticker = str(item).strip().upper()
        if ticker.endswith(".JK"):
            ticker = ticker[:-3]
        if ticker and all(ch.isalnum() or ch in ".-" for ch in ticker):
            if ticker not in tickers:
                tickers.append(ticker)
    return tickers



def _safe_ticker_value(node):
    """Evaluasi format ticker umum tanpa menjalankan kode dari app.py."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_safe_ticker_value(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _safe_ticker_value(node.left)
        right = _safe_ticker_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    if isinstance(node, ast.Call):
        # Mendukung pola: "BBCA,BBRI,...".split(",")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "split":
            value = _safe_ticker_value(node.func.value)
            args = [_safe_ticker_value(arg) for arg in node.args]
            if isinstance(value, str) and len(args) <= 2:
                return value.split(*args)
        # Mendukung list([...]), tuple([...]), sorted([...]) dan set([...])
        if isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple", "set", "sorted"}:
            if len(node.args) == 1 and not node.keywords:
                value = _safe_ticker_value(node.args[0])
                if node.func.id == "list":
                    return list(value)
                if node.func.id == "tuple":
                    return tuple(value)
                if node.func.id == "set":
                    return set(value)
                return sorted(value)
    raise ValueError("Ekspresi DEFAULT_TICKERS tidak didukung")

def load_tickers():
    """Baca assignment DEFAULT_TICKERS dari app.py tanpa mengimpor Streamlit."""
    if not os.path.isfile(APP_FILE):
        raise FileNotFoundError(
            f"{APP_FILE} tidak ditemukan. Pastikan app.py ada di root repositori."
        )

    with open(APP_FILE, "r", encoding="utf-8") as file:
        source = file.read()
    tree = ast.parse(source, filename=APP_FILE)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_TICKERS"
            for target in targets
        ):
            continue

        try:
            raw = _safe_ticker_value(node.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(
                "Format DEFAULT_TICKERS di app.py belum didukung. Gunakan daftar string, "
                "string dipisahkan koma, atau string dengan .split(',')."
            ) from error

        tickers = _normalize_tickers(raw)
        if not tickers:
            raise ValueError("DEFAULT_TICKERS ditemukan, tetapi tidak berisi ticker valid.")
        print(f"Berhasil membaca {len(tickers)} ticker dari app.py")
        return tickers

    raise ValueError("DEFAULT_TICKERS tidak ditemukan sebagai assignment di app.py.")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(state):
    """Simpan status secara atomik; status tetap perlu dipersistenkan antarrun CI."""
    temp_path = STATE_FILE + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, STATE_FILE)
    except OSError as error:
        print(f"Gagal menyimpan status notifikasi: {error}")


def rupiah(value):
    try:
        value = float(value)
        if not np.isfinite(value):
            return "-"
        return "Rp" + f"{value:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def fetch_history(ticker):
    """Ambil data harian Yahoo Finance; ini bukan jaminan harga real-time."""
    try:
        df = yf.download(
            f"{ticker}.JK", period="1y", interval="1d", auto_adjust=True,
            progress=False, threads=False, timeout=20
        )
        if df is None or df.empty:
            print(f"{ticker}: data kosong dari Yahoo Finance.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [str(col).title() for col in df.columns]

        required = ["High", "Low", "Close", "Volume"]
        if not all(column in df.columns for column in required):
            print(f"{ticker}: kolom harga/volume tidak lengkap.")
            return None

        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[(df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] >= 0)]
        if len(df) < 60:
            print(f"{ticker}: data valid kurang dari 60 hari.")
            return None

        last_timestamp = pd.Timestamp(df.index[-1])
        if last_timestamp.tzinfo is not None:
            last_timestamp = last_timestamp.tz_convert(TIMEZONE).tz_localize(None)
        last_date = last_timestamp.date()
        age_days = (datetime.now(TIMEZONE).date() - last_date).days
        if age_days > 7:
            print(f"{ticker}: data terakhir {last_date} ({age_days} hari lalu), dilewati.")
            return None

        df.attrs["data_date"] = str(last_date)
        return df
    except Exception as error:
        print(f"{ticker}: gagal mengambil data: {error}")
        return None


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100)
    return result.fillna(50)


def analyze_stock(ticker, df):
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = calculate_rsi(close)
    avg_volume = volume.rolling(20).mean()
    resistance = high.shift(1).rolling(20).max()
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()

    price = float(close.iloc[-1])
    ema9_now, ema21_now, ema50_now = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])
    avg_vol = float(avg_volume.iloc[-1]) if pd.notna(avg_volume.iloc[-1]) else 0.0
    current_vol = float(volume.iloc[-1])
    resistance_value = resistance.iloc[-1]
    resistance_now = float(resistance_value) if pd.notna(resistance_value) else np.nan
    atr_value = atr.iloc[-1]
    atr_now = float(atr_value) if pd.notna(atr_value) and float(atr_value) > 0 else price * 0.03
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0

    trend_up = price > ema21_now and ema9_now > ema21_now
    breakout = pd.notna(resistance_now) and price > resistance_now
    false_breakout = (
        pd.notna(resistance_now) and float(high.iloc[-1]) > resistance_now and price < resistance_now
    )
    distance = (resistance_now - price) / price * 100 if pd.notna(resistance_now) and price > 0 else np.nan
    prebreakout = (
        not breakout and not false_breakout and pd.notna(distance) and 0 <= distance <= 3
        and price > ema21_now and volume_ratio >= MIN_VOLUME_RATIO and rsi_now >= 50
    )
    momentum_5d = (price / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 and float(close.iloc[-6]) > 0 else 0.0
    momentum = (
        not breakout and not false_breakout and trend_up and momentum_5d > 0
        and pd.notna(distance) and 0 <= distance <= 5 and volume_ratio >= MIN_VOLUME_RATIO
        and 50 <= rsi_now <= 70
    )

    # Indikasi akumulasi teknikal saja; bukan data broker/bukti aktivitas bandar.
    flat_10d = len(close) >= 11 and abs((price / float(close.iloc[-11]) - 1) * 100) <= 4
    recent_volume = float(volume.tail(5).mean())
    previous_volume = float(volume.iloc[-25:-5].mean()) if len(volume) >= 25 else 0.0
    accumulation = (
        flat_10d and previous_volume > 0 and recent_volume >= previous_volume * 1.15
        and price > ema21_now
    )

    score = 0
    if trend_up: score += 20
    if price > ema50_now: score += 10
    if breakout: score += 25
    if prebreakout: score += 15
    if volume_ratio >= MIN_VOLUME_RATIO: score += 15
    elif volume_ratio >= 0.8: score += 5
    if 50 <= rsi_now <= 65: score += 10
    elif 45 <= rsi_now <= 70: score += 5
    if price > ema9_now: score += 5
    if accumulation: score += 10
    score = min(score, 100)

    above_ema9 = (price - ema9_now) / ema9_now * 100 if ema9_now > 0 else 0.0
    watch_only = rsi_now > 70 or above_ema9 > 3
    entry_low = min(price, ema9_now) if price > ema9_now else price * 0.995
    entry_high = price
    stop_loss = max(price - 1.5 * atr_now, 0.0)
    risk = price - stop_loss
    target_1 = price + max(2.3 * atr_now, risk * MIN_REWARD_RISK)
    target_2 = price + max(3.0 * atr_now, risk * 2.0)
    rr1 = (target_1 - price) / risk if risk > 0 else 0.0
    rr2 = (target_2 - price) / risk if risk > 0 else 0.0
    risk_ok = rr1 >= MIN_REWARD_RISK and not false_breakout and not watch_only

    if false_breakout:
        signal = "WASPADA FALSE BREAKOUT"
    elif (breakout and trend_up and volume_ratio >= MIN_VOLUME_RATIO
          and 50 <= rsi_now <= 70 and score >= MIN_BREAKOUT_SCORE):
        signal = "BREAKOUT TERKONFIRMASI"
    elif prebreakout and volume_ratio >= MIN_VOLUME_RATIO and score >= MIN_BREAKOUT_SCORE:
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
        score >= MIN_SCORE and risk_ok and not false_breakout
        and (breakout or prebreakout or accumulation or momentum
             or (trend_up and volume_ratio >= MIN_VOLUME_RATIO))
    )
    return {
        "ticker": ticker, "price": price, "score": score, "signal": signal,
        "rsi": rsi_now, "volume_ratio": volume_ratio, "momentum_5d": momentum_5d,
        "entry_low": entry_low, "entry_high": entry_high, "stop_loss": stop_loss,
        "target_1": target_1, "target_2": target_2, "rr1": rr1, "rr2": rr2,
        "is_candidate": candidate, "data_date": df.attrs.get("data_date", "tidak diketahui")
    }


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram belum dikonfigurasi; notifikasi tidak dikirim.")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message}, timeout=20
        )
        if response.ok:
            return True
        print(f"Telegram gagal: HTTP {response.status_code} - {response.text[:200]}")
    except requests.RequestException as error:
        print(f"Telegram gagal: {error}")
    return False


def make_message(item):
    return (
        "BOT AI SAHAM BEI V7.0\n"
        f"Saham: {item['ticker']}.JK\n"
        f"Tanggal data: {item['data_date']} (Yahoo Finance; dapat tertunda)\n"
        f"Sinyal: {item['signal']}\nSkor: {item['score']}/100\n"
        f"Harga penutupan data: {rupiah(item['price'])}\n"
        f"Area entry indikatif: {rupiah(item['entry_low'])} - {rupiah(item['entry_high'])}\n"
        f"Stop Loss indikatif: {rupiah(item['stop_loss'])}\n"
        f"TP1: {rupiah(item['target_1'])} (RR {item['rr1']:.2f}R)\n"
        f"TP2: {rupiah(item['target_2'])} (RR {item['rr2']:.2f}R)\n"
        f"RSI: {item['rsi']:.1f}\nRasio volume: {item['volume_ratio']:.2f}x\n"
        "Broker lokal/asing: belum terhubung; bukan data broker aktual.\n"
        "Catatan: verifikasi harga di Stockbit. Sinyal bukan jaminan harga naik."
    )


def scan(send_alerts=True):
    tickers = load_tickers()
    state = load_state()
    today = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    results = []
    print(f"BOT AI SAHAM BEI V7.0: {len(tickers)} saham")
    print("Sumber data harga: Yahoo Finance; bukan harga real-time terjamin.")

    for number, ticker in enumerate(tickers, start=1):
        print(f"[{number}/{len(tickers)}] Memindai {ticker}.JK")
        df = fetch_history(ticker)
        if df is not None:
            try:
                item = analyze_stock(ticker, df)
                results.append(item)
                print(
                    f"{item['signal']} | Skor {item['score']} | Harga {rupiah(item['price'])} | "
                    f"Tanggal data {item['data_date']}"
                )
                if send_alerts and item["is_candidate"]:
                    if state.get(ticker) != today:
                        if send_telegram(make_message(item)):
                            state[ticker] = today
                            save_state(state)
                    else:
                        print("Notifikasi saham ini sudah tercatat hari ini.")
            except Exception as error:
                print(f"Gagal menganalisis {ticker}: {error}")
        time.sleep(REQUEST_DELAY_SECONDS)

    results.sort(key=lambda item: (item["score"], item["volume_ratio"]), reverse=True)
    print("\n=== 10 SAHAM TERATAS ===")
    for item in results[:TOP_N]:
        print(
            f"{item['ticker']}.JK | {item['signal']} | Skor {item['score']} | "
            f"Harga {rupiah(item['price'])} | Data {item['data_date']} | "
            f"Entry {rupiah(item['entry_low'])}-{rupiah(item['entry_high'])} | "
            f"SL {rupiah(item['stop_loss'])} | TP1 {rupiah(item['target_1'])} | "
            f"TP2 {rupiah(item['target_2'])}"
        )
    print("\nData broker lokal/asing belum terhubung; Yahoo Finance tidak menyediakan broker summary.")
    print("Catatan: GitHub Actions tidak mempertahankan file JSON antar-run secara otomatis. "
          "Agar batas satu notifikasi per saham per hari berlaku, status harus dipersistenkan antarrun.")
    return results


if __name__ == "__main__":
    scan()
