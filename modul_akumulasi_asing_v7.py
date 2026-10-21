"""
BOT AI SAHAM BEI V7.0 — Modul Uji Akumulasi Asing 5 Hari
Versi awal: membaca file Ringkasan Saham BEI (.xlsx) yang diunduh manual,
menghitung net foreign dalam JUMLAH SAHAM, lalu menggabungkan indikator teknikal Yahoo Finance.

PENTING:
- Modul ini berdiri sendiri; tidak mengubah scanner_v7.py atau workflow V7.0.
- Simpan file Excel dalam folder data_foreign/ dengan nama:
  Ringkasan Saham-20261014.xlsx, ...20261015.xlsx, ...20261016.xlsx,
  ...20261019.xlsx, ...20261020.xlsx
- Foreign Buy/Sell pada file BEI ini berupa jumlah saham. Net foreign = Foreign Buy - Foreign Sell.
- Data Excel manual belum membuat pengambilan data BEI menjadi otomatis.
"""

from pathlib import Path
import re
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data_foreign")
OUTPUT_CSV = Path("hasil_akumulasi_asing_5_hari.csv")

MIN_POSITIVE_DAYS = 4
MIN_VOLUME_RATIO = 1.2
MIN_SCORE = 75
RSI_MIN = 50
RSI_MAX = 70


def read_beI_file(path: Path) -> pd.DataFrame:
    """Baca file BEI dan samakan nama kolom yang diperlukan."""
    df = pd.read_excel(path)
    # Normalisasi spasi pada nama kolom.
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Kode Saham", "Foreign Buy", "Foreign Sell"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom {sorted(missing)} tidak ditemukan di {path.name}")

    out = df[["Kode Saham", "Foreign Buy", "Foreign Sell"]].copy()
    out["Kode Saham"] = out["Kode Saham"].astype(str).str.strip().str.upper()
    out = out[out["Kode Saham"].str.fullmatch(r"[A-Z0-9]+", na=False)]
    for col in ["Foreign Buy", "Foreign Sell"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["Net Foreign"] = out["Foreign Buy"] - out["Foreign Sell"]
    out["Tanggal File"] = path.stem.replace("Ringkasan Saham-", "")
    return out


def load_five_days() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("Ringkasan Saham-*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file Excel. Buat folder {DATA_DIR}/ lalu masukkan "
            "file Ringkasan Saham BEI (.xlsx) di dalamnya."
        )

    daily = [read_beI_file(f) for f in files]
    all_data = pd.concat(daily, ignore_index=True)
    n_dates = all_data["Tanggal File"].nunique()
    if n_dates < 5:
        print(f"PERINGATAN: baru ada {n_dates} tanggal file; idealnya 5 hari perdagangan.")

    summary = all_data.groupby("Kode Saham", as_index=False).agg(
        NetForeign5D=("Net Foreign", "sum"),
        HariNetBuy=("Net Foreign", lambda s: int((s > 0).sum())),
        HariData=("Net Foreign", "count"),
        ForeignBuyTotal=("Foreign Buy", "sum"),
        ForeignSellTotal=("Foreign Sell", "sum"),
    )
    summary["PersenHariNetBuy"] = (
        summary["HariNetBuy"] / summary["HariData"].replace(0, pd.NA) * 100
    )
    return summary


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def get_technical(code: str) -> dict:
    ticker = f"{code}.JK"
    try:
        hist = yf.download(
            ticker, period="6mo", interval="1d",
            auto_adjust=False, progress=False, threads=False
        )
        if hist is None or hist.empty or len(hist) < 30:
            return {"Kode Saham": code, "DataTeknikal": "Tidak tersedia"}

        # yfinance kadang mengembalikan MultiIndex columns.
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        volume = pd.to_numeric(hist["Volume"], errors="coerce").reindex(close.index)
        if len(close) < 30:
            return {"Kode Saham": code, "DataTeknikal": "Data kurang"}

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        rsi14 = rsi(close)
        avgvol20 = volume.rolling(20).mean()
        volratio = volume / avgvol20.replace(0, pd.NA)
        prior_high20 = close.shift(1).rolling(20).max()

        price = float(close.iloc[-1])
        ema9_now = float(ema9.iloc[-1])
        ema21_now = float(ema21.iloc[-1])
        rsi_now = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else None
        volratio_now = float(volratio.iloc[-1]) if pd.notna(volratio.iloc[-1]) else None
        breakout = bool(price > float(prior_high20.iloc[-1])) if pd.notna(prior_high20.iloc[-1]) else False
        trend_up = ema9_now > ema21_now
        price_above_ema21 = price > ema21_now
        volume_strong = volratio_now is not None and volratio_now >= MIN_VOLUME_RATIO
        rsi_ok = rsi_now is not None and RSI_MIN <= rsi_now <= RSI_MAX

        score = 0
        score += 25 if trend_up else 0
        score += 15 if price_above_ema21 else 0
        score += 30 if breakout else 0
        score += 20 if volume_strong else 0
        if rsi_now is not None:
            if 50 <= rsi_now <= 65:
                score += 10
            elif 65 < rsi_now <= 70:
                score += 5

        strong = (
            trend_up and price_above_ema21 and volume_strong and rsi_ok
            and score >= MIN_SCORE
        )
        return {
            "Kode Saham": code,
            "HargaYahooTerakhir": round(price, 2),
            "EMA9": round(ema9_now, 2),
            "EMA21": round(ema21_now, 2),
            "RSI14": round(rsi_now, 2) if rsi_now is not None else None,
            "RasioVolume20D": round(volratio_now, 2) if volratio_now is not None else None,
            "Breakout20D": breakout,
            "SkorTeknikal": score,
            "SinyalTeknikalKuat": strong,
            "DataTeknikal": "OK",
        }
    except Exception as exc:
        return {"Kode Saham": code, "DataTeknikal": f"Gagal: {type(exc).__name__}"}


def main():
    foreign = load_five_days()
    foreign["AkumulasiAsingKonsisten"] = (
        (foreign["NetForeign5D"] > 0)
        & (foreign["HariNetBuy"] >= MIN_POSITIVE_DAYS)
        & (foreign["HariData"] >= 5)
    )

    # Ambil kandidat dari saham yang net foreign positif dan konsisten.
    candidates = foreign[foreign["AkumulasiAsingKonsisten"]].copy()
    if candidates.empty:
        print("Tidak ada kandidat yang memenuhi filter net foreign. Periksa file dan datanya.")
        foreign.sort_values("NetForeign5D", ascending=False).to_csv(OUTPUT_CSV, index=False)
        print(f"Hasil awal disimpan: {OUTPUT_CSV}")
        return

    print(f"Menguji teknikal untuk {len(candidates)} kandidat. Ini mungkin perlu beberapa menit.")
    technical = [get_technical(code) for code in candidates["Kode Saham"].tolist()]
    tech_df = pd.DataFrame(technical)
    result = candidates.merge(tech_df, on="Kode Saham", how="left")
    result["KandidatKuat"] = (
        result["AkumulasiAsingKonsisten"]
        & result["SinyalTeknikalKuat"].fillna(False)
        & (result["SkorTeknikal"].fillna(0) >= MIN_SCORE)
    )
    result = result.sort_values(
        ["KandidatKuat", "SkorTeknikal", "NetForeign5D"],
        ascending=[False, False, False]
    )
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    cols = [
        "Kode Saham", "NetForeign5D", "HariNetBuy", "HariData",
        "HargaYahooTerakhir", "RSI14", "RasioVolume20D",
        "Breakout20D", "SkorTeknikal", "KandidatKuat"
    ]
    print("\nHASIL TERATAS:")
    print(result[[c for c in cols if c in result.columns]].head(30).to_string(index=False))
    print(f"\nFile lengkap disimpan: {OUTPUT_CSV}")
    print("Catatan: harga teknikal berasal dari Yahoo Finance; validasi terhadap aplikasi broker diperlukan.")


if __name__ == "__main__":
    main()
