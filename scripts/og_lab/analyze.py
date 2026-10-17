"""
Анализ поведения OG/USDT (OG Fan Token): связь с BTC, волатильность, пампы, возврат к среднему.

    PYTHONPATH=. python3 -m scripts.og_lab.analyze [--market spot|futures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.dca_lab.data import load_symbol

OUT = Path(__file__).resolve().parents[2] / "output" / "og_lab"
DAY = 1440


def resample(ts: np.ndarray, ohlc: np.ndarray, qvol: np.ndarray, minutes: int) -> dict[str, np.ndarray]:
    n = len(ts) // minutes * minutes
    o = ohlc[:n].reshape(-1, minutes, 4)
    return {
        "ts": ts[:n:minutes],
        "open": o[:, 0, 0],
        "high": o[:, :, 1].max(axis=1),
        "low": o[:, :, 2].min(axis=1),
        "close": o[:, -1, 3],
        "qvol": qvol[:n].reshape(-1, minutes).sum(axis=1),
    }


def aligned_returns(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common, ia, ib = np.intersect1d(a["ts"], b["ts"], return_indices=True)
    ra = np.diff(np.log(a["close"][ia]))
    rb = np.diff(np.log(b["close"][ib]))
    return common[1:], ra, rb


def autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="spot", choices=("spot", "futures"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    og = load_symbol("OGUSDT", args.market)
    btc = load_symbol("BTCUSDT")
    report: dict = {"market": args.market}

    # Начало на ровной границе суток, чтобы бары разных монет совпадали.
    def day_aligned(d: dict) -> dict:
        first = int(np.argmax((d["ts"] % (DAY * 60_000)) == 0))
        return {k: v[first:] for k, v in d.items()}

    og, btc = day_aligned(og), day_aligned(btc)
    report["period"] = [str(np.datetime64(int(og["ts"][0]), "ms")), str(np.datetime64(int(og["ts"][-1]), "ms"))]

    # 1. Корреляция с BTC по разным ТФ и по годам.
    corr = {}
    for name, minutes in (("1h", 60), ("4h", 240), ("1d", DAY)):
        a = resample(og["ts"], og["ohlc"], og["qvol"], minutes)
        b = resample(btc["ts"], btc["ohlc"], btc["qvol"], minutes)
        t, ra, rb = aligned_returns(a, b)
        ok = np.isfinite(ra) & np.isfinite(rb) & (ra != 0)
        corr[name] = round(float(np.corrcoef(ra[ok], rb[ok])[0, 1]), 3)
        if name == "1d":
            years = np.datetime64("1970") + (t // 1000).astype("timedelta64[s]")
            yr = years.astype("datetime64[Y]").astype(int) + 1970
            corr["1d_by_year"] = {
                int(y): round(float(np.corrcoef(ra[ok & (yr == y)], rb[ok & (yr == y)])[0, 1]), 3)
                for y in np.unique(yr)
                if (ok & (yr == y)).sum() > 30
            }
            beta = float(np.polyfit(rb[ok], ra[ok], 1)[0])
            corr["beta_1d"] = round(beta, 2)
    report["corr_btc"] = corr

    d = resample(og["ts"], og["ohlc"], og["qvol"], DAY)
    r = np.diff(np.log(d["close"]))
    report["daily"] = {
        "vol_annual_pct": round(float(r.std() * np.sqrt(365) * 100), 1),
        "btc_vol_annual_pct": round(
            float(np.diff(np.log(resample(btc["ts"], btc["ohlc"], btc["qvol"], DAY)["close"])).std() * np.sqrt(365) * 100), 1
        ),
        "share_days_abs_gt_10pct": round(float((np.abs(r) > np.log(1.1)).mean() * 100), 1),
        "max_day_up_pct": round(float((np.exp(r.max()) - 1) * 100), 1),
        "max_day_down_pct": round(float((np.exp(r.min()) - 1) * 100), 1),
        "median_range_pct": round(float(np.median((d["high"] - d["low"]) / d["open"]) * 100), 1),
    }

    # 2. Возврат к среднему или тренд: автокорреляция доходностей на разных ТФ.
    ac = {}
    for name, minutes in (("15m", 15), ("1h", 60), ("4h", 240), ("1d", DAY)):
        a = resample(og["ts"], og["ohlc"], og["qvol"], minutes)
        rr = np.diff(np.log(a["close"]))
        rr = rr[np.isfinite(rr)]
        ac[name] = {f"lag{l}": round(autocorr(rr, l), 3) for l in (1, 2, 3)}
    report["autocorr"] = ac

    # Variance ratio: <1 — возврат к среднему, >1 — тренд.
    h = resample(og["ts"], og["ohlc"], og["qvol"], 60)
    rh = np.diff(np.log(h["close"]))
    vr = {}
    for k in (4, 24, 24 * 7):
        agg = np.convolve(rh, np.ones(k), "valid")[::k]
        vr[f"{k}h"] = round(float(agg.var() / (k * rh.var())), 3)
    report["variance_ratio"] = vr

    # 3. Пампы: часы с ростом > X% и что было дальше (24 ч / 72 ч).
    pumps = {}
    for thr in (0.05, 0.10, 0.20):
        c = h["close"]
        ret1 = c[1:] / c[:-1] - 1
        idx = np.where(ret1 > thr)[0] + 1
        # не считать соседние часы одного пампа
        idx = idx[np.concatenate([[True], np.diff(idx) > 24])] if len(idx) else idx
        nxt = {}
        for hz in (4, 24, 72):
            v = [c[i + hz] / c[i] - 1 for i in idx if i + hz < len(c)]
            nxt[f"next_{hz}h_median_pct"] = round(float(np.median(v) * 100), 2) if v else None
            nxt[f"next_{hz}h_share_down"] = round(float(np.mean(np.array(v) < 0)), 2) if v else None
        pumps[f"hour_up_{int(thr * 100)}pct"] = {"count": int(len(idx)), **nxt}
    # Дампы (часовое падение) и отскок
    dumps = {}
    for thr in (0.05, 0.10):
        c = h["close"]
        ret1 = c[1:] / c[:-1] - 1
        idx = np.where(ret1 < -thr)[0] + 1
        idx = idx[np.concatenate([[True], np.diff(idx) > 24])] if len(idx) else idx
        nxt = {}
        for hz in (4, 24, 72):
            v = [c[i + hz] / c[i] - 1 for i in idx if i + hz < len(c)]
            nxt[f"next_{hz}h_median_pct"] = round(float(np.median(v) * 100), 2) if v else None
            nxt[f"next_{hz}h_share_up"] = round(float(np.mean(np.array(v) > 0)), 2) if v else None
        dumps[f"hour_down_{int(thr * 100)}pct"] = {"count": int(len(idx)), **nxt}
    report["pumps"] = pumps
    report["dumps"] = dumps

    # 4. Объём: всплески объёма и направление.
    qv = h["qvol"]
    med = np.array([np.median(qv[max(0, i - 168) : i]) if i > 24 else np.nan for i in range(len(qv))])
    spike = np.where(qv > 8 * med)[0]
    spike = spike[np.concatenate([[True], np.diff(spike) > 24])] if len(spike) else spike
    c = h["close"]
    v24 = [c[i + 24] / c[i] - 1 for i in spike if i + 24 < len(c)]
    report["volume_spikes_8x"] = {
        "count": int(len(spike)),
        "next_24h_median_pct": round(float(np.median(v24) * 100), 2) if v24 else None,
        "next_24h_share_down": round(float(np.mean(np.array(v24) < 0)), 2) if v24 else None,
    }

    # 5. По годам: доходность buy&hold, просадка, средний дневной диапазон.
    years = {}
    t_days = np.datetime64("1970") + (d["ts"] // 1000).astype("timedelta64[s]")
    yr = t_days.astype("datetime64[Y]").astype(int) + 1970
    for y in np.unique(yr):
        m = yr == y
        c = d["close"][m]
        peak = np.maximum.accumulate(c)
        years[int(y)] = {
            "return_pct": round(float((c[-1] / d["open"][m][0] - 1) * 100), 1),
            "max_dd_pct": round(float((1 - (c / peak).min()) * 100), 1),
            "median_day_range_pct": round(float(np.median((d["high"][m] - d["low"][m]) / d["open"][m]) * 100), 1),
            "avg_daily_volume_musd": round(float(d["qvol"][m].mean() / 1e6), 2),
        }
    report["years"] = years

    # 6. Часы суток (UTC): средняя доходность и волатильность.
    hours = ((h["ts"] // 3_600_000) % 24)[1:]
    report["hour_of_day"] = {
        int(k): {"mean_bp": round(float(rh[hours == k].mean() * 1e4), 1), "vol_bp": round(float(rh[hours == k].std() * 1e4), 0)}
        for k in range(24)
    }
    out = OUT / f"analysis_{args.market}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "hour_of_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
