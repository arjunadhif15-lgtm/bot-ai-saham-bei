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
HISTORY_DAYS = 20


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
    # Ambil rentang kalender yang cukup untuk memperoleh 20 sesi bursa
    # (termasuk akhir pekan dan kemungkinan hari libur).
    start = end - timedelta(days=45)
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

        # Tentukan tepat lima tanggal perdagangan terbaru dari seluruh seri broker.
        all_dates = set()
        for broker in series:
            if not isinstance(broker, dict):
                continue
            for point in broker.get("points", []) or []:
                if isinstance(point, dict):
                    day = str(point.get("date", "")).strip()
                    if day:
                        all_dates.add(day)

        if len(all_dates) < MIN_DAYS:
            raise RuntimeError(
                f"{ticker}: hanya {len(all_dates)} hari perdagangan tersedia "
                "(butuh minimal 5). Batch tidak disimpan."
            )

        ordered_dates = sorted(all_dates)
        dates = ordered_dates[-MIN_DAYS:]
        dates_20d = ordered_dates[-HISTORY_DAYS:]
        selected_dates = set(dates)
        selected_dates_20d = set(dates_20d)
        totals = {}
        for broker in series:
            if not isinstance(broker, dict):
                continue
            code = str(broker.get("broker_code", "")).strip()
            if not code:
                continue
            total = 0.0
            pos = neg = count = 0
            # Estimasi AVG akumulasi: bavg harian dibobot hanya dengan
            # nvol positif. nvol adalah volume NET, bukan volume beli bruto.
            weighted_buy_value = 0.0
            positive_net_volume = 0.0
            buy_days = 0
            for point in broker.get("points", []) or []:
                if not isinstance(point, dict):
                    continue
                day = str(point.get("date", "")).strip()
                if day not in selected_dates:
                    continue
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
                try:
                    net_volume = float(point.get("nvol") or 0)
                    buy_avg = float(point.get("bavg")) if point.get("bavg") is not None else 0.0
                    if (net_volume > 0 and buy_avg > 0 and
                            net_volume == net_volume and buy_avg == buy_avg and
                            abs(net_volume) != float("inf") and abs(buy_avg) != float("inf")):
                        weighted_buy_value += buy_avg * net_volume
                        positive_net_volume += net_volume
                        buy_days += 1
                except (TypeError, ValueError, OverflowError):
                    pass
            if count:
                totals[code] = {
                    "broker_code": code,
                    "broker_name": str(broker.get("broker_name") or code),
                    "total_nval": round(total, 2),
                    "positive_days": pos,
                    "negative_days": neg,
                    "observed_days": count,
                    "avg_buy_price_5d": round(weighted_buy_value / positive_net_volume, 2) if positive_net_volume > 0 else None,
                    "positive_net_volume_5d": round(positive_net_volume, 2),
                    "buy_days_5d": buy_days,
                    "avg_buy_method": "bavg_weighted_by_positive_net_volume_indicative",
                }

        buyers = sorted((x for x in totals.values() if x["total_nval"] > 0),
                        key=lambda x: x["total_nval"], reverse=True)
        sellers = sorted((x for x in totals.values() if x["total_nval"] < 0),
                         key=lambda x: x["total_nval"])
        buy_total = sum(x["total_nval"] for x in buyers)
        sell_total = abs(sum(x["total_nval"] for x in sellers))
        gross = buy_total + sell_total
        # 0-100: dominasi nilai bersih beli vs jual selama lima sesi.
        balance = 50.0 if gross == 0 else 50.0 + 50.0 * (buy_total - sell_total) / gross

        active = [x for x in totals.values() if x["observed_days"] > 0]
        consistencies = [
            x["positive_days"] / x["observed_days"]
            for x in buyers[:5] if x["observed_days"] > 0
        ]
        consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0
        breadth = (sum(1 for x in active if x["total_nval"] > 0) / len(active)) if active else 0.5

        # Skor dibuat transparan: dominasi aliran 55%, konsistensi top buyer 30%,
        # dan keluasan broker net-buy 15%. Bukan probabilitas atau prediksi harga.
        score = round(max(0, min(100, 0.55 * balance + 30 * consistency + 15 * breadth)), 2)
        if score >= 80:
            signal = "AKUMULASI KUAT"
        elif score >= 65:
            signal = "INDIKASI AKUMULASI"
        elif score <= 20:
            signal = "DISTRIBUSI KUAT"
        elif score <= 35:
            signal = "INDIKASI DISTRIBUSI"
        else:
            signal = "NETRAL / CAMPURAN"

        daily_flow = []
        for day in dates:
            day_buy = day_sell = 0.0
            day_positive = day_negative = 0
            for broker in series:
                if not isinstance(broker, dict):
                    continue
                for point in broker.get("points", []) or []:
                    if not isinstance(point, dict) or str(point.get("date", "")).strip() != day:
                        continue
                    try:
                        value = float(point.get("nval") or 0)
                        if value != value or abs(value) == float("inf"):
                            value = 0.0
                    except (TypeError, ValueError, OverflowError):
                        value = 0.0
                    if value > 0:
                        day_buy += value
                        day_positive += 1
                    elif value < 0:
                        day_sell += abs(value)
                        day_negative += 1
            daily_flow.append({
                "date": day,
                "net_buy": round(day_buy, 2),
                "net_sell": round(day_sell, 2),
                "net_flow": round(day_buy - day_sell, 2),
                "positive_brokers": day_positive,
                "negative_brokers": day_negative,
            })

        # Agregasi history 20 sesi: dihitung terpisah agar skor harian 5 sesi
        # tetap kompatibel dengan versi sebelumnya. AVG indikatif memakai
        # bavg dibobot nvol positif, bukan menganggap nvol sebagai volume beli bruto.
        totals_20d = {}
        for broker in series:
            if not isinstance(broker, dict):
                continue
            code = str(broker.get("broker_code", "")).strip()
            if not code:
                continue
            total_20 = 0.0
            pos_20 = neg_20 = observed_20 = buy_days_20 = 0
            weighted_value_20 = positive_volume_20 = 0.0
            for point in broker.get("points", []) or []:
                if not isinstance(point, dict) or str(point.get("date", "")).strip() not in selected_dates_20d:
                    continue
                try:
                    value = float(point.get("nval") or 0)
                    if value != value or abs(value) == float("inf"):
                        value = 0.0
                except (TypeError, ValueError, OverflowError):
                    value = 0.0
                total_20 += value
                observed_20 += 1
                pos_20 += int(value > 0)
                neg_20 += int(value < 0)
                try:
                    net_volume = float(point.get("nvol") or 0)
                    buy_avg = float(point.get("bavg")) if point.get("bavg") is not None else 0.0
                    if (net_volume > 0 and buy_avg > 0 and net_volume == net_volume and
                            buy_avg == buy_avg and abs(net_volume) != float("inf") and
                            abs(buy_avg) != float("inf")):
                        weighted_value_20 += buy_avg * net_volume
                        positive_volume_20 += net_volume
                        buy_days_20 += 1
                except (TypeError, ValueError, OverflowError):
                    pass
            if observed_20:
                totals_20d[code] = {
                    "broker_code": code,
                    "broker_name": str(broker.get("broker_name") or code),
                    "total_nval": round(total_20, 2),
                    "positive_days": pos_20,
                    "negative_days": neg_20,
                    "observed_days": observed_20,
                    "avg_buy_price_20d": round(weighted_value_20 / positive_volume_20, 2) if positive_volume_20 > 0 else None,
                    "positive_net_volume_20d": round(positive_volume_20, 2),
                    "buy_days_20d": buy_days_20,
                }

        buyers_20d = sorted((x for x in totals_20d.values() if x["total_nval"] > 0),
                            key=lambda x: x["total_nval"], reverse=True)
        sellers_20d = sorted((x for x in totals_20d.values() if x["total_nval"] < 0),
                             key=lambda x: x["total_nval"])
        daily_flow_20d = []
        for day in dates_20d:
            day_buy = day_sell = 0.0
            day_positive = day_negative = 0
            for broker in series:
                if not isinstance(broker, dict):
                    continue
                for point in broker.get("points", []) or []:
                    if not isinstance(point, dict) or str(point.get("date", "")).strip() != day:
                        continue
                    try:
                        value = float(point.get("nval") or 0)
                        if value != value or abs(value) == float("inf"):
                            value = 0.0
                    except (TypeError, ValueError, OverflowError):
                        value = 0.0
                    if value > 0:
                        day_buy += value
                        day_positive += 1
                    elif value < 0:
                        day_sell += abs(value)
                        day_negative += 1
            daily_flow_20d.append({
                "date": day,
                "net_buy": round(day_buy, 2),
                "net_sell": round(day_sell, 2),
                "net_flow": round(day_buy - day_sell, 2),
                "positive_brokers": day_positive,
                "negative_brokers": day_negative,
            })
        coverage_20d = round(min(1.0, sum(x["observed_days"] for x in totals_20d.values()) /
                                 max(1, len(totals_20d) * len(dates_20d))), 4)

        coverage = round(min(1.0, sum(x["observed_days"] for x in active) /
                             max(1, len(active) * len(dates))), 4)
        confidence = "TINGGI" if coverage >= 0.90 and len(active) >= 10 else (
            "SEDANG" if coverage >= 0.65 and len(active) >= 5 else "TERBATAS")
        top3_share = (sum(x["total_nval"] for x in buyers[:3]) / buy_total) if buy_total > 0 else 0.0

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
                "method": "skor_55_persen_dominasi_net_flow_30_persen_konsistensi_top_buyer_15_persen_breadth",
                "score_components": {
                    "net_flow_balance": round(balance, 2),
                    "top_buyer_consistency": round(consistency * 100, 2),
                    "broker_breadth": round(breadth * 100, 2),
                    "coverage": coverage,
                    "top3_buyer_share": round(top3_share, 4),
                },
                "buyer_consistency": round(consistency, 4),
                "broker_breadth": round(breadth, 4),
                "data_coverage": coverage,
                "confidence": confidence,
                "daily_flow": daily_flow,
                "history_20d": {
                    "trading_dates": dates_20d,
                    "trading_days_count": len(dates_20d),
                    "data_coverage": coverage_20d,
                    "top_buyers": buyers_20d[:10],
                    "top_sellers": sellers_20d[:10],
                    "daily_flow": daily_flow_20d,
                    "note": "Riwayat 20 sesi perdagangan terakhir (atau sesi tersedia). AVG indikatif memakai bavg dibobot volume net buy positif; bukan harga modal pasti broker."
                },
                "trading_dates": sorted(dates),
                "trading_days_count": len(dates),
                "note": (
                    "Skor adalah heuristik aliran broker lima sesi; riwayat 20 sesi ditampilkan terpisah, bukan probabilitas atau prediksi harga. "
                    "Broker flow bukan identitas pemilik manfaat dan bukan bukti pasti aktivitas bandar "
                    "atau rekomendasi beli/jual. Konfirmasi dengan harga, volume, likuiditas, dan risiko."
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
