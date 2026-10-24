"""
Проверка исполнения стопов на минутных свечах: для каждой сделки базового кандидата, закрытой по стопу,
скачиваем 1m-свечи фьючерса за день стопа (data.binance.vision, futures/um/daily/klines) и считаем,
где сработал бы STOP_MARKET (первая минута с high >= стоп; цена = max(open минуты, стоп)).

    python3 -m scripts.edge_lab.verify_listings_1m
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.edge_lab.common import CACHE, DAY, HOUR, OUT  # noqa: E402
from scripts.edge_lab.universe import _get  # noqa: E402
from scripts.edge_lab import verify_listings as V  # noqa: E402

D1M = CACHE / "listings" / "verify_1m"


def fetch_day(sym: str, day_ms: int) -> np.ndarray | None:
    """1m свечи [ts, o, h, l, c] за сутки; кеш в npz."""
    D1M.mkdir(parents=True, exist_ok=True)
    date = str(np.datetime64(day_ms, "ms").astype("datetime64[D]"))
    f = D1M / f"{sym}_{date}.npz"
    if f.exists():
        with np.load(f) as z:
            return z["a"] if z["a"].size else None
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{date}.zip"
    blob = _get(url)
    a = np.empty((0, 5))
    if blob:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            txt = zf.read(zf.namelist()[0]).decode()
        rows = [r.split(",") for r in txt.strip().splitlines() if r and r[0].isdigit()]
        a = np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows])
    np.savez(f, a=a)
    return a if a.size else None


def stop_fills(trades: list[dict], S: float) -> dict:
    jobs = [r for r in trades if r["reason"] == "stop"]
    hours = [(r["sym"], r["L"] + r["xb"] * HOUR) for r in jobs]
    with ThreadPoolExecutor(8) as ex:
        days = list(ex.map(lambda x: fetch_day(x[0], x[1] // DAY * DAY), hours))
    out = {}
    for r, (sym, th), a in zip(jobs, hours, days):
        if a is None:
            continue
        stop = r["pe"] * (1 + S)
        m = (a[:, 0] >= th) & (a[:, 0] < th + HOUR)
        b = a[m]
        hit = np.where(b[:, 2] >= stop)[0]
        if len(hit) == 0:
            continue
        k = hit[0]
        out[(sym, r["L"])] = {"fill_1m": float(max(b[k, 1], stop)), "fill_1h_raw": float(r["px"] / 1.003),
                              "minute": int(k), "close_1m": float(b[k, 4]), "stop": float(stop)}
    return out


def main():
    events = V.find_events()
    crypto = [e for e in events if e["cls"] == "crypto"]
    wins = {(e["sym"], e["L"]): V.event_window(e["sym"], e["L"]) for e in crypto}
    btc = V.BTC()
    res = {}
    for S in (0.30, 0.225, 0.375):
        tr = V.build_trades(crypto, wins, btc, 24, 168, S, 0.25)
        f = stop_fills(tr, S)
        ex = np.array([v["fill_1m"] / v["stop"] - 1 for v in f.values()])
        ex_h = np.array([v["fill_1h_raw"] / v["stop"] - 1 for v in f.values()])
        n_stop = sum(r["reason"] == "stop" for r in tr)
        res[str(S)] = {"n_stops": n_stop, "n_with_1m": len(f),
                       "gap_over_stop_1m_mean_pct": round(float(ex.mean()) * 100, 3),
                       "gap_over_stop_1m_median_pct": round(float(np.median(ex)) * 100, 3),
                       "gap_over_stop_1m_p90_pct": round(float(np.percentile(ex, 90)) * 100, 2),
                       "gap_over_stop_1m_max_pct": round(float(ex.max()) * 100, 2),
                       "gap_over_stop_1h_mean_pct": round(float(ex_h.mean()) * 100, 3),
                       "share_1m_gap_gt_0_3pct": round(float(np.mean(ex > 0.003)), 3)}
        print(S, res[str(S)], flush=True)
        if S == 0.30:
            fills = {f"{k[0]}|{k[1]}": v for k, v in f.items()}
            json.dump(fills, open(OUT / "listings" / "verify_1m_stopfills.json", "w"), indent=0)
    json.dump(res, open(OUT / "listings" / "verify_1m_summary.json", "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ------------------------------------------------------------------------------------------------
# aggTrades: реальная цена первой сделки >= стопа и цены в первые 0.2 / 1 / 5 с после срабатывания
# ------------------------------------------------------------------------------------------------
DAT = CACHE / "listings" / "verify_agg"


def agg_slice(sym: str, th: int) -> np.ndarray | None:
    import pandas as pd
    DAT.mkdir(parents=True, exist_ok=True)
    f = DAT / f"{sym}_{th}.npz"
    if f.exists():
        with np.load(f) as z:
            return z["a"] if z["a"].size else None
    date = str(np.datetime64(th, "ms").astype("datetime64[D]"))
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{date}.zip"
    blob = _get(url)
    a = np.empty((0, 3))
    if blob:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            raw = zf.read(zf.namelist()[0])
        first = raw[:64].decode(errors="ignore")
        hdr = 0 if first[:1].isalpha() else None
        df = pd.read_csv(io.BytesIO(raw), header=hdr, usecols=[1, 2, 5])
        df.columns = ["p", "q", "t"]
        df = df[(df["t"] >= th) & (df["t"] < th + HOUR)]
        a = df[["t", "p", "q"]].to_numpy(dtype=float)
    np.savez(f, a=a)
    return a if a.size else None


def agg_check(n_sample: int = 40):
    events = V.find_events()
    crypto = [e for e in events if e["cls"] == "crypto"]
    wins = {(e["sym"], e["L"]): V.event_window(e["sym"], e["L"]) for e in crypto}
    btc = V.BTC()
    S = 0.30
    tr = V.build_trades(crypto, wins, btc, 24, 168, S, 0.25)
    stops = [r for r in tr if r["reason"] == "stop" and r["t_entry"] >= V.ms("2024-01-01")]
    f1 = json.load(open(OUT / "listings" / "verify_1m_stopfills.json"))
    jump = sorted(stops, key=lambda r: -f1[f"{r['sym']}|{r['L']}"]["close_1m"] / f1[f"{r['sym']}|{r['L']}"]["stop"])
    pick = {id(r): r for r in jump[:10]}
    for r in stops[:: max(1, len(stops) // (n_sample - 10))]:
        pick[id(r)] = r
    pick = list(pick.values())
    jobs = [(r["sym"], r["L"] + r["xb"] * HOUR) for r in pick]
    with ThreadPoolExecutor(3) as ex:
        sl = list(ex.map(lambda x: agg_slice(*x), jobs))
    rows = []
    for r, (sym, th), a in zip(pick, jobs, sl):
        if a is None:
            continue
        stop = r["pe"] * (1 + S)
        k = np.where(a[:, 1] >= stop)[0]
        if len(k) == 0:
            continue
        k = k[0]
        t0 = a[k, 0]
        res = {"sym": sym, "t": int(th), "first": a[k, 1] / stop - 1}
        for lag in (200, 1000, 5000):
            m = (a[:, 0] >= t0) & (a[:, 0] <= t0 + lag)
            res[f"vwap_{lag}"] = float(np.sum(a[m, 1] * a[m, 2]) / np.sum(a[m, 2])) / stop - 1
            res[f"max_{lag}"] = float(a[m, 1].max()) / stop - 1
        rows.append(res)
    out = {"n": len(rows)}
    for key in ("first", "vwap_200", "max_200", "vwap_1000", "max_1000", "vwap_5000", "max_5000"):
        x = np.array([q[key] for q in rows])
        out[key] = {"mean_pct": round(float(x.mean()) * 100, 3), "median_pct": round(float(np.median(x)) * 100, 3),
                    "p90_pct": round(float(np.percentile(x, 90)) * 100, 3), "max_pct": round(float(x.max()) * 100, 2)}
    out["rows"] = rows
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    json.dump(out, open(OUT / "listings" / "verify_agg_stops.json", "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "agg":
    agg_check()
