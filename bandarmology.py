#!/usr/bin/env python3
"""Bandarmology TOP 10: IDX Edge PRO -> Supabase. Modul terpisah dari scanner.py."""
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

TZ = ZoneInfo("Asia/Jakarta")
IDX = "https://stock.arjum.com/api"
RETRIES = 3
MIN_DAYS = 5


def log(msg):
    print(f"[BANDARMOLOGY] {msg}", flush=True)


def env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Secret/environment {name} belum diatur.")
    return value


def call(session, method, url, headers, params=None, body=None, timeout=30):
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.request(method, url, headers=headers, params=params,
                                json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                last = f"HTTP {r.status_code}: {r.text[:250]}"
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == RETRIES - 1:
                break
    raise RuntimeError(f"Request gagal: {last}")


def main():
    supa = env("SUPABASE_URL").rstrip("/")
    service = env("SUPABASE_SERVICE_ROLE_KEY")
    idx_key = env("IDX_EDGE_API_KEY")
    sh = {"apikey": service, "Authorization": f"Bearer {service}",
          "Accept": "application/json", "Content-Type": "application/json"}
    ih = {"X-API-Key": idx_key, "Accept": "application/json"}
    session = requests.Session()

    # Ambil maksimal 50 baris terbaru, lalu pilih 10 ticker unik dari tanggal terbaru.
    latest = call(session, "GET", f"{supa}/rest/v1/top10_scans", sh,
                  params={"select": "scan_date,scan_time,ticker",
                          "order": "scan_date.desc,scan_time.desc", "limit": "50"})
    if not isinstance(latest, list) or not latest:
        raise RuntimeError("top10_scans kosong/tidak valid; tidak ada data yang diubah.")
    scan_date = max(str(x.get("scan_date", "")) for x in latest)
    tickers = []
    for row in latest:
        if str(row.get("scan_date", "")) == scan_date:
            t = str(row.get("ticker", "")).strip().upper()
            if t and t not in tickers:
                tickers.append(t)
    if len(tickers) != 10:
        raise RuntimeError(
            f"Tanggal {scan_date} hanya memiliki {len(tickers)} ticker unik, "
            "bukan 10. Proses dihentikan; data lama aman."
        )

    end = date.fromisoformat(scan_date)
    start = end - timedelta(days=14)
    rows = []
    log(f"TOP 10 tanggal {scan_date}: {', '.join(tickers)}")

    for i, ticker in enumerate(tickers, 1):
        log(f"[{i}/10] Mengambil broker {ticker}")
        summary = call(
            session, "GET", f"{IDX}/broker-summary/{ticker}", ih,
            params={"net": "false", "broker_limit": 20, "level_limit": 25,
                    "all_data": "false", "flow": "all"})
        history = call(
            session, "GET", f"{IDX}/broker-accumulation/{ticker}", ih,
            params={"start_date": start.isoformat(), "end_date": end.isoformat()})
        if not isinstance(summary, dict) or not isinstance(history, dict):
            raise RuntimeError(f"{ticker}: format respons API tidak valid.")

        series = history.get("series")
        if not isinstance(series, list):
            raise RuntimeError(f"{ticker}: data series broker tidak ditemukan.")

        dates = set()
        totals = {}
        for broker in series:
            if not isinstance(broker, dict):
                continue
            code = str(broker.get("broker_code", "")).strip()
            if not code:
                continue
            total = 0.0
            pos = neg = count = 0
            for point in broker.get("points", []) or []:
                if not isinstance(point, dict):
                    continue
                day = str(point.get("date", "")).strip()
                if day:
                    dates.add(day)
                try:
                    val = float(point.get("nval") or 0)
                    if val != val or abs(val) == float("inf"):
                        val = 0.0
                except (TypeError, ValueError, OverflowError):
                    val = 0.0
                total += val
                count += 1
                pos += int(val > 0)
                neg += int(val < 0)
            totals[code] = {
                "broker_code": code,
                "broker_name": str(broker.get("broker_name") or code),
                "total_nval": round(total, 2),
                "positive_days": pos,
                "negative_days": neg,
                "observed_days": count,
            }

        if len(dates) < MIN_DAYS:
            raise RuntimeError(
                f"{ticker}: hanya {len(dates)} hari perdagangan tersedia "
                "(butuh minimal 5). Batch tidak disimpan."
            )

        buyers = sorted((x for x in totals.values() if x["total_nval"] > 0),
                        key=lambda x: x["total_nval"], reverse=True)
        sellers = sorted((x for x in totals.values() if x["total_nval"] < 0),
                         key=lambda x: x["total_nval"])
        buy_total = sum(x["total_nval"] for x in buyers)
        sell_total = abs(sum(x["total_nval"] for x in sellers))
        gross = buy_total + sell_total
        balance = 50.0 if gross == 0 else 50.0 + 50.0 * (buy_total - sell_total) / gross
        consistencies = [
            x["positive_days"] / x["observed_days"]
            for x in buyers[:5] if x["observed_days"] > 0
        ]
        consistency = sum(consistencies) / len(consistencies) if consistencies else 0
        score = round(max(0, min(100, 0.75 * balance + 25 * consistency)), 2)
        signal = ("INDIKASI AKUMULASI" if score >= 65 else
                  "INDIKASI DISTRIBUSI" if score <= 35 else "NETRAL / CAMPURAN")

        rows.append({
            "scan_date": scan_date,
            "ticker": ticker,
            "broker_summary": summary,
            "broker_accumulation": history,
            "top_buyers": buyers[:5],
            "top_sellers": sellers[:5],
            "net_buy_total": round(buy_total, 2),
            "net_sell_total": round(sell_total, 2),
            "bandar_score": score,
            "signal": signal,
            "analysis": {
                "method": "heuristik_net_flow_dan_konsistensi_broker",
                "balance_score": round(balance, 2),
                "buyer_consistency": round(consistency, 4),
                "trading_dates": sorted(dates),
                "trading_days_count": len(dates),
                "note": (
                    "Broker flow bukan identitas pemilik manfaat dan bukan bukti "
                    "pasti aktivitas bandar atau rekomendasi beli/jual."
                ),
            },
        })
        log(f"{ticker}: {signal}, skor {score}/100, {len(dates)} sesi")

    if len(rows) != 10:
        raise RuntimeError("Hasil tidak lengkap 10 saham; batch tidak disimpan.")

    # Upsert hanya setelah SEMUA 10 saham berhasil; kegagalan tidak menghapus data lama.
    save_headers = dict(sh)
    save_headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    response = session.post(
        f"{supa}/rest/v1/bandarmology_scans",
        headers=save_headers,
        params={"on_conflict": "scan_date,ticker"},
        json=rows,
        timeout=45,
    )
    response.raise_for_status()
    log(f"BERHASIL: {len(rows)} saham tersimpan untuk {scan_date}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"GAGAL: {exc}")
        sys.exit(1)
