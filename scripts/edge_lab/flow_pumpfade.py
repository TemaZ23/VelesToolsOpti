"""
flow (c): торговля после часовых пампов/дампов с всплеском объёма — портфель равновзвешенных позиций.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_pumpfade            # полный грид -> part_c.json
    PYTHONPATH=. python3 -m scripts.edge_lab.flow_pumpfade show       # таблица лучших по IS

Гипотезы (одинаковый грид, разные знаки):
  fade_pump   — шорт после пампа r1 = close[t]/close[t-1]-1 > X с объёмом vr >= V;
  bounce_dump — лонг после дампа r1 < -X с vr >= V;
  follow_pump — лонг после пампа (продолжение);
  follow_dump — шорт после дампа (продолжение).
Сигнал по close[t] (или по первому «развороту» — бару с close против движения — в течение 6 ч после события,
confirm=1), вход open следующего бара тейкером. Выходы: стоп / тейк (лимит, maker, только «сквозь» уровень) /
по времени; стоп раньше тейка в одном баре; ликвидация шорта на +90% (изолированная маржа 1x).
Позиция = капитал/K, не более K одновременно (валовая экспозиция ≤ 1x, без плеча).
Выборка: все крипто-перпы (включая делистинговые и мелкие) с возрастом ≥ 168 ч; min_adv — фильтр по ADV.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
from numba import njit

from scripts.edge_lab.common import IS, OOS, OUT, metrics, ms
from scripts.edge_lab.flow_common import age_hours, cost_matrix, daily_from_hourly, load_all, past_median_at, summarize
from scripts.edge_lab.flow_sim import simulate

ODIR = OUT / "flow"


@njit(cache=True)
def confirm_idx(C, rows, cols, sign, maxlag):
    """Первый бар t' in (t, t+maxlag], где close[t'] идёт против движения (для пампа: close[t'] < close[t'-1])."""
    out = np.full(len(rows), -1, np.int64)
    T = C.shape[0]
    for k in range(len(rows)):
        t, j = rows[k], cols[k]
        for s in range(t + 1, min(T, t + maxlag + 1)):
            a, b = C[s, j], C[s - 1, j]
            if not (a == a) or not (b == b):
                break
            if sign * (a - b) < 0:
                out[k] = s
                break
    return out


def prep(p: dict, xmin: float = 0.05) -> dict:
    c = p["close"]
    age = age_hours(c)
    r1 = np.full(c.shape, np.nan, dtype=np.float32)
    r1[1:] = c[1:] / c[:-1] - 1
    ok = (age >= 169) & p["crypto"][None, :] & np.isfinite(r1) & (np.abs(r1) > xmin)
    rows, cols = np.nonzero(ok)
    vr = past_median_at(p["qvol"], rows, cols, 168, 100, 1)
    vr = np.nan_to_num(p["qvol"][rows, cols] / vr, nan=0.0)
    r = r1[rows, cols].astype(float)
    conf_up = confirm_idx(c, rows, cols, 1.0, 6)
    conf_dn = confirm_idx(c, rows, cols, -1.0, 6)
    del r1, age
    return {"rows": rows, "cols": cols, "r": r, "vr": vr, "adv": np.nan_to_num(p["adv"][rows, cols].astype(float)),
            "conf_up": conf_up, "conf_dn": conf_dn}  # fmt: skip


def signals(ev: dict, side: int, X: float, V: float, min_adv: float, confirm: int):
    """side=+1 события-пампы, −1 дампы. Возвращает отсортированные (sig_t, sig_j) и приоритет по |r|."""
    m = (side * ev["r"] > X) & (ev["vr"] >= V) & (ev["adv"] >= min_adv)
    t = ev["rows"][m]
    if confirm:
        t = (ev["conf_up"] if side > 0 else ev["conf_dn"])[m]
    j = ev["cols"][m]
    a = np.abs(ev["r"][m])
    keep = t >= 0
    t, j, a = t[keep], j[keep], a[keep]
    order = np.lexsort((-a, t))
    return t[order].astype(np.int64), j[order].astype(np.int64)


def run_one(p, COST, ev, side, direction, X, V, H, stop, tp, K, confirm, min_adv, delist_pen=0.0,
            stop_slip_mult=2.0, full=False, sig=None):  # fmt: skip
    st, sj = signals(ev, side, X, V, min_adv, confirm) if sig is None else sig
    T = len(p["ts"])
    out = simulate(p["open"], p["high"], p["low"], p["close"], p["funding"], COST, st, sj, direction, H, stop, tp,
                   True, 1.0 / K, K, delist_pen, stop_slip_mult, 0, 0, T)  # fmt: skip
    eq, tj, t0, t1, tret, tfund, tcost, treason, avg_gross = out
    dts, dret = daily_from_hourly(p["ts"], eq)
    tts = p["ts"][t0] if len(t0) else np.empty(0, np.int64)
    res = {}
    for name, (a, b) in (("IS", IS), ("OOS", OOS)):
        m = (dts >= ms(a)) & (dts < ms(b))
        res[name] = metrics(dts[m], dret[m])
        tm = (tts >= ms(a)) & (tts < ms(b))
        res[name]["trades"] = int(tm.sum())
        if tm.sum():
            res[name]["avg_trade_net_pct"] = round(float(np.mean(tret[tm] + tfund[tm] - tcost[tm])) * 100, 3)
    res["avg_gross"] = round(float(avg_gross), 3)
    if full:
        return res, {"eq": eq, "dts": dts, "dret": dret, "tj": tj, "t0": t0, "t1": t1, "tret": tret, "tfund": tfund,
                     "tcost": tcost, "treason": treason}  # fmt: skip
    return res


HYP = {
    "fade_pump": (1, -1),
    "bounce_dump": (-1, 1),
    "follow_pump": (1, 1),
    "follow_dump": (-1, -1),
}
GRID = {
    "fade_pump": dict(X=[0.10, 0.15, 0.20, 0.30, 0.40], V=[3, 10], H=[4, 12, 24, 48, 96], stop=[0.0, 0.10, 0.20, 0.40],
                      tp=[0.0, 0.10, 0.20], K=[5, 10], confirm=[0, 1], min_adv=[0.0, 1e7]),
    "bounce_dump": dict(X=[0.05, 0.10, 0.15, 0.20, 0.30], V=[3, 10], H=[4, 12, 24, 48, 96], stop=[0.0, 0.10, 0.20, 0.40],
                        tp=[0.0, 0.10, 0.20], K=[5, 10], confirm=[0, 1], min_adv=[0.0, 1e7]),
    "follow_pump": dict(X=[0.05, 0.10, 0.20, 0.30], V=[3, 10], H=[1, 4, 12, 24, 48], stop=[0.0, 0.10],
                        tp=[0.0, 0.20], K=[5, 10], confirm=[0], min_adv=[0.0, 1e7]),
    "follow_dump": dict(X=[0.05, 0.10, 0.20, 0.30], V=[3, 10], H=[1, 4, 12, 24, 48], stop=[0.0, 0.10],
                        tp=[0.0, 0.20], K=[5, 10], confirm=[0], min_adv=[0.0, 1e7]),
}  # fmt: skip
KEYS = ["X", "V", "H", "stop", "tp", "K", "confirm", "min_adv"]


def run(hyps: list[str]) -> None:
    t0 = time.time()
    p = load_all()
    COST = cost_matrix(p["adv"])
    ev = prep(p)
    print("events", len(ev["r"]), round(time.time() - t0, 1), flush=True)
    ODIR.mkdir(parents=True, exist_ok=True)
    path = ODIR / "part_c.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    for hyp in hyps:
        side, direction = HYP[hyp]
        g = GRID[hyp]
        rows = []
        sig_cache = {}
        for vals in itertools.product(*[g[k] for k in KEYS]):
            cfg = dict(zip(KEYS, vals))
            sk = (cfg["X"], cfg["V"], cfg["min_adv"], cfg["confirm"])
            if sk not in sig_cache:
                sig_cache[sk] = signals(ev, side, cfg["X"], cfg["V"], cfg["min_adv"], cfg["confirm"])
            r = run_one(p, COST, ev, side, direction, cfg["X"], cfg["V"], cfg["H"], cfg["stop"], cfg["tp"], cfg["K"],
                        cfg["confirm"], cfg["min_adv"], sig=sig_cache[sk])  # fmt: skip
            rows.append({"cfg": cfg, **r})
        allres[hyp] = rows
        print(hyp, len(rows), round(time.time() - t0, 1), flush=True)
        path.write_text(json.dumps(allres, default=float))
    show(allres)


def pick_best(rows: list[dict], min_is_trades: int = 30) -> dict | None:
    ok = [r for r in rows if r["IS"].get("trades", 0) >= min_is_trades]
    if not ok:
        return None
    return max(ok, key=lambda r: r["IS"]["sharpe"])


def show(allres: dict | None = None) -> None:
    if allres is None:
        allres = json.loads((ODIR / "part_c.json").read_text())
    for hyp, rows in allres.items():
        isr = np.array([r["IS"]["sharpe"] for r in rows])
        oos = np.array([r["OOS"]["sharpe"] for r in rows])
        print(f"\n== {hyp}: {len(rows)} configs; IS SR median {np.median(isr):.2f}; OOS SR median {np.median(oos):.2f}, "
              f"share OOS>0 {np.mean(oos > 0):.2f}; corr(IS,OOS) {np.corrcoef(isr, oos)[0, 1]:.2f}")  # fmt: skip
        top = sorted([r for r in rows if r["IS"].get("trades", 0) >= 30], key=lambda r: -r["IS"]["sharpe"])[:12]
        for r in top:
            print(" ", r["cfg"], "| IS", {k: r["IS"].get(k) for k in ("sharpe", "cagr_pct", "max_dd_pct", "trades")},
                  "| OOS", {k: r["OOS"].get(k) for k in ("sharpe", "cagr_pct", "max_dd_pct", "trades")})  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        run(sys.argv[1:] or list(HYP))
