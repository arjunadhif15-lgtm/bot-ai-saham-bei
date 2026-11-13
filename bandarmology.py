
#!/usr/bin/env python3
"""
Bandarmology TOP 10:
IDX Edge PRO -> Supabase.

Modul terpisah dari scanner.py.
Tidak menjalankan scanner dan tidak mengirim Telegram.
"""

import os
import sys
from datetime import date, timedelta
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
        raise RuntimeError(
            f"Secret/environment {name} belum diatur."
        )

    return value


def call(
    session,
    method,
    url,
    headers,
    params=None,
    body=None,
    timeout=30,
):
    last = None

    for attempt in range(RETRIES):
        try:
            response = session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=body,
                timeout=timeout,
            )

            if (
                response.status_code in (429, 500, 502, 503, 504)
                and attempt < RETRIES - 1
            ):
                last = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:250]}"
                )
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as exc:
            last = exc

            if attempt == RETRIES - 1:
                break

    raise RuntimeError(f"Request gagal: {last}")


def main():

    # ==================================================
    # 1. MEMBACA SECRET
    # ==================================================

    supa = env("SUPABASE_URL").rstrip("/")
    service = env("SUPABASE_SERVICE_ROLE_KEY")
    idx_key = env("IDX_EDGE_API_KEY")

    sh = {
        "apikey": service,
        "Authorization": f"Bearer {service}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    ih = {
        "X-API-Key": idx_key,
        "Accept": "application/json",
    }

    session = requests.Session()

    log("Memulai proses Bandarmology TOP 10.")

    # ==================================================
    # 2. MENGAMBIL TOP 10 DARI SUPABASE
    # ==================================================

    latest = call(
        session,
        "GET",
        f"{supa}/rest/v1/top10_scans",
        sh,
        params={
            "select": "scan_date,scan_time,ticker",
            "order": "scan_date.desc,scan_time.desc",
            "limit": "50",
        },
    )

    if not isinstance(latest, list) or not latest:
        raise RuntimeError(
            "top10_scans kosong/tidak valid. "
            "Tidak ada data yang diubah."
        )

    scan_date = max(
        str(row.get("scan_date", ""))
        for row in latest
    )

    tickers = []

    for row in latest:

        if str(row.get("scan_date", "")) == scan_date:

            ticker = str(
                row.get("ticker", "")
            ).strip().upper()

            if ticker and ticker not in tickers:
                tickers.append(ticker)

    if len(tickers) != 10:
        raise RuntimeError(
            f"Tanggal {scan_date} hanya memiliki "
            f"{len(tickers)} ticker unik, bukan 10. "
            "Proses dihentikan; data lama aman."
        )

    end = date.fromisoformat(scan_date)
    start = end - timedelta(days=14)

    rows = []

    log(
        f"TOP 10 tanggal {scan_date}: "
        f"{', '.join(tickers)}"
    )

    # ==================================================
    # 3. ANALISIS BROKER SETIAP SAHAM
    # ==================================================

    for i, ticker in enumerate(tickers, 1):

        log(
            f"[{i}/10] Mengambil data broker {ticker}"
        )

        # ----------------------------------------------
        # BROKER SUMMARY
        # ----------------------------------------------

        summary = call(
            session,
            "GET",
            f"{IDX}/broker-summary/{ticker}",
            ih,
            params={
                "net": "false",
                "broker_limit": 20,
                "level_limit": 25,
                "all_data": "false",
                "flow": "all",
            },
        )

        # ----------------------------------------------
        # BROKER ACCUMULATION
        # ----------------------------------------------

        history = call(
            session,
            "GET",
            f"{IDX}/broker-accumulation/{ticker}",
            ih,
            params={
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )

        if (
            not isinstance(summary, dict)
            or not isinstance(history, dict)
        ):
            raise RuntimeError(
                f"{ticker}: format respons API tidak valid."
            )

        series = history.get("series")

        if not isinstance(series, list):
            raise RuntimeError(
                f"{ticker}: data series broker tidak ditemukan."
            )

        # ----------------------------------------------
        # MENENTUKAN 5 HARI PERDAGANGAN TERAKHIR
        # ----------------------------------------------

        all_dates = set()

        for broker in series:

            if not isinstance(broker, dict):
                continue

            for point in broker.get("points", []) or []:

                if isinstance(point, dict):

                    day = str(
                        point.get("date", "")
                    ).strip()

                    if day:
                        all_dates.add(day)

        if len(all_dates) < MIN_DAYS:
            raise RuntimeError(
                f"{ticker}: hanya {len(all_dates)} "
                "hari perdagangan tersedia. "
                "Butuh minimal 5 hari. "
                "Batch tidak disimpan."
            )

        dates = sorted(all_dates)[-MIN_DAYS:]
        selected_dates = set(dates)

        # ----------------------------------------------
        # MENGHITUNG NET VALUE SETIAP BROKER
        # ----------------------------------------------

        totals = {}

        for broker in series:

            if not isinstance(broker, dict):
                continue

            code = str(
                broker.get("broker_code", "")
            ).strip()

            if not code:
                continue

            total = 0.0
            positive = 0
            negative = 0
            count = 0

            for point in broker.get("points", []) or []:

                if not isinstance(point, dict):
                    continue

                day = str(
                    point.get("date", "")
                ).strip()

                if day not in selected_dates:
                    continue

                try:
                    value = float(
                        point.get("nval") or 0
                    )

                    if (
                        value != value
                        or abs(value) == float("inf")
                    ):
                        value = 0.0

                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ):
                    value = 0.0

                total += value
                count += 1

                positive += int(value > 0)
                negative += int(value < 0)

            if count:

                totals[code] = {
                    "broker_code": code,
                    "broker_name": str(
                        broker.get("broker_name") or code
                    ),
                    "total_nval": round(total, 2),
                    "positive_days": positive,
                    "negative_days": negative,
                    "observed_days": count,
                }

        # ----------------------------------------------
        # TOP BUYERS DAN TOP SELLERS
        # ----------------------------------------------

        buyers = sorted(
            (
                item
                for item in totals.values()
                if item["total_nval"] > 0
            ),
            key=lambda item: item["total_nval"],
            reverse=True,
        )

        sellers = sorted(
            (
                item
                for item in totals.values()
                if item["total_nval"] < 0
            ),
            key=lambda item: item["total_nval"],
        )

        buy_total = sum(
            item["total_nval"]
            for item in buyers
        )

        sell_total = abs(
            sum(
                item["total_nval"]
                for item in sellers
            )
        )

        # ----------------------------------------------
        # PERHITUNGAN BANDAR SCORE
        # ----------------------------------------------

        gross = buy_total + sell_total

        if gross == 0:
            balance = 50.0
        else:
            balance = (
                50.0
                + 50.0 * (buy_total - sell_total) / gross
            )

        consistencies = [
            item["positive_days"] / item["observed_days"]
            for item in buyers[:5]
            if item["observed_days"] > 0
        ]

        if consistencies:
            consistency = (
                sum(consistencies) / len(consistencies)
            )
        else:
            consistency = 0.0

        score = round(
            max(
                0,
                min(
                    100,
                    0.75 * balance + 25 * consistency,
                ),
            ),
            2,
        )

        if score >= 65:
            signal = "INDIKASI AKUMULASI"

        elif score <= 35:
            signal = "INDIKASI DISTRIBUSI"

        else:
            signal = "NETRAL / CAMPURAN"

        # ----------------------------------------------
        # MENYUSUN DATA UNTUK SUPABASE
        # ----------------------------------------------

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

                "method": (
                    "heuristik_net_flow_dan_"
                    "konsistensi_broker"
                ),

                "balance_score": round(
                    balance,
                    2,
                ),

                "buyer_consistency": round(
                    consistency,
                    4,
                ),

                "trading_dates": dates,

                "trading_days_count": len(dates),

                "note": (
                    "Broker flow bukan identitas "
                    "pemilik manfaat dan bukan bukti "
                    "pasti aktivitas bandar atau "
                    "rekomendasi beli/jual."
                ),
            },
        })

        log(
            f"{ticker}: {signal}, "
            f"skor {score}/100, "
            f"{len(dates)} sesi"
        )

    # ==================================================
    # 4. VALIDASI SEBELUM PENYIMPANAN
    # ==================================================

    if len(rows) != 10:
        raise RuntimeError(
            "Hasil tidak lengkap 10 saham. "
            "Batch tidak disimpan."
        )

    log(
        "Seluruh 10 saham berhasil dianalisis. "
        "Memulai penyimpanan Supabase."
    )

    # ==================================================
    # 5. MENYIMPAN HASIL KE SUPABASE
    # ==================================================

    save_headers = dict(sh)

    save_headers["Prefer"] = (
        "resolution=merge-duplicates,return=minimal"
    )

    response = session.post(
        f"{supa}/rest/v1/bandarmology_scans",
        headers=save_headers,
        params={
            "on_conflict": "scan_date,ticker"
        },
        json=rows,
        timeout=45,
    )

    # ----------------------------------------------
    # DIAGNOSTIK ERROR SUPABASE
    # ----------------------------------------------

    if not response.ok:

        try:
            error_detail = response.json()

        except ValueError:
            error_detail = response.text[:1500]

        log(
            f"SUPABASE HTTP {response.status_code}"
        )

        log(
            f"DETAIL ERROR: {error_detail}"
        )

        raise RuntimeError(
            "Gagal menyimpan hasil Bandarmology "
            "ke Supabase. Periksa DETAIL ERROR di atas."
        )

    # ==================================================
    # 6. BERHASIL
    # ==================================================

    log(
        f"BERHASIL: {len(rows)} saham tersimpan "
        f"untuk {scan_date}."
    )

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())

    except Exception as exc:

        log(f"GAGAL: {exc}")

        sys.exit(1)
