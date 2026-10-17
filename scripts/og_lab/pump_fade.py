"""
«Шорт после большого пампа» в форме бота: вход на закрытии часа, выросшего больше чем на X%,
выход по тейку −TP% или стопу +SL% (по минутным максимумам/минимумам, стоп проверяется первым —
худший случай), без выхода по времени. Капитал не реинвестируется: каждая сделка — номинал 1×.
Комиссии и проскальзывание 0.25% на круг.

    PYTHONPATH=. python3 -m scripts.og_lab.pump_fade
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.dca_lab.data import load_symbol

from .analyze import resample
from .events import COST, PERIODS

OUT = Path(__file__).resolve().parents[2] / "output" / "og_lab"


def simulate(d: dict, thr: float, tp: float, sl: float, max_hours: int = 24 * 30) -> list[dict]:
    """Минутная симуляция: сигнал по часовым барам, исполнение по минутам."""
    h = resample(d["ts"], d["ohlc"], d["qvol"], 60)
    hi_m, lo_m, ts_m = d["ohlc"][:, 1], d["ohlc"][:, 2], d["ts"]
    c = h["close"]
    out = []
    busy_until = -1
    for i in range(1, len(c)):
        if c[i] / c[i - 1] - 1 <= thr or i <= busy_until:
            continue
        entry = c[i]
        m0 = (i + 1) * 60  # первая минута после закрытия часа
        m1 = min(len(ts_m), m0 + max_hours * 60)
        up = np.nonzero(hi_m[m0:m1] >= entry * (1 + sl))[0]
        down = np.nonzero(lo_m[m0:m1] <= entry * (1 - tp))[0]
        t_up = up[0] if len(up) else None
        t_dn = down[0] if len(down) else None
        if t_up is None and t_dn is None:
            ret, k, how = 1 - d["ohlc"][m1 - 1, 3] / entry, m1 - m0, "time"
        elif t_dn is None or (t_up is not None and t_up <= t_dn):
            ret, k, how = -sl, t_up, "stop"
        else:
            ret, k, how = tp, t_dn, "take"
        out.append({"ts": int(h["ts"][i]), "ret": ret - COST, "hours": round(k / 60, 1), "exit": how,
                    "pump_pct": round((c[i] / c[i - 1] - 1) * 100, 1)})  # fmt: skip
        busy_until = i + int(np.ceil(k / 60))
    return out


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"n": 0}
    r = np.array([t["ret"] for t in tr])
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    return {
        "n": len(r),
        "mean_pct": round(float(r.mean() * 100), 2),
        "win": round(float((r > 0).mean()), 2),
        "sum_pct": round(float(r.sum() * 100), 1),
        "worst_pct": round(float(r.min() * 100), 1),
        "max_dd_pct": round(dd * 100, 1),
        "median_hours": float(np.median([t["hours"] for t in tr])),
        "stops": sum(t["exit"] == "stop" for t in tr),
    }


def by_period(tr: list[dict]) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        lo = np.datetime64(a).astype("datetime64[ms]").astype(np.int64)
        hi = np.datetime64(b).astype("datetime64[ms]").astype(np.int64)
        out[name] = stats([t for t in tr if lo <= t["ts"] < hi])
    return out


def main() -> None:
    spot = load_symbol("OGUSDT", "spot")
    fut = load_symbol("OGUSDT", "futures")
    grid = []
    for thr in (0.12, 0.15, 0.2, 0.25):
        for tp in (0.06, 0.1, 0.15, 0.2, 0.3):
            for sl in (0.15, 0.25, 0.4, 0.6):
                tr = simulate(spot, thr, tp, sl)
                grid.append({"thr": thr, "tp": tp, "sl": sl, "all": stats(tr), "periods": by_period(tr)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pump_fade_grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=1))

    def robust(g):  # худший средний результат сделки по периодам, где были сделки
        vals = [p["mean_pct"] for p in g["periods"].values() if p.get("n", 0) >= 2]
        return min(vals) if vals else -99

    print("Лучшие по худшему периоду (порог роста / тейк / стоп):")
    for g in sorted(grid, key=lambda g: -robust(g))[:10]:
        a = g["all"]
        per = " | ".join(f"{k}: {v.get('mean_pct')}% n{v.get('n')}" for k, v in g["periods"].items())
        print(
            f"  >{int(g['thr']*100)}% тейк {int(g['tp']*100)}% стоп {int(g['sl']*100)}%: всего n{a['n']} ср {a['mean_pct']}% "
            f"win {a['win']} сумма {a['sum_pct']}% худшая {a['worst_pct']}% просадка {a['max_dd_pct']}% стопов {a['stops']} "
            f"медиана {a['median_hours']}ч\n      {per}"
        )
    # Тот же сигнал на фьючерсе (где бот и будет торговать)
    best = sorted(grid, key=lambda g: -robust(g))[0]
    tr_f = simulate(fut, best["thr"], best["tp"], best["sl"])
    print("\nФьючерс OGUSDT (2025-05…2026-09), лучший вариант:", stats(tr_f))
    for t in simulate(spot, best["thr"], best["tp"], best["sl"]):
        print("   ", str(np.datetime64(t["ts"], "ms"))[:16], f"памп +{t['pump_pct']}%", f"{t['exit']} {t['ret']*100:+.1f}% за {t['hours']}ч")


if __name__ == "__main__":
    main()
