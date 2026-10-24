"""
flow (b): всплески объёма + сильные часовые движения — продолжение или откат на 1–48 ч.

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_events     # -> output/edge_lab/flow/part_b.json

Событие (монета j, бар t): r_w = close[t]/close[t-w] - 1 (w = 1 или 4 часа) выше +X (памп) или ниже −X (дамп);
объём окна относительно медианы часового объёма за 168 ч ДО окна (vr = mean(qvol[t-w+1..t]) / median).
Возраст монеты ≥ 168 ч, только крипто-перпы (TradFi исключены), делистинговые монеты включены.
Дедупликация: первое событие того же знака по монете за 24 ч.
Вход open[t+1], выход close[t+h], h ∈ {1,2,4,8,12,24,48}; если монета пропала — последний close.
Статистика: средняя/медианная доходность ЛОНГА (знак «продолжения» для пампа), рыночно-скорректированная
(минус BTC за то же окно), t-статистика с кластеризацией по суткам, доля разворотов, нетто для сделки «против
движения» (fade) и «по движению» (follow): −2·cost_per_side(ADV) ± фандинг.
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.edge_lab.common import DAY, OUT, ms, IS, OOS
from scripts.edge_lab.flow_common import (
    age_hours, cost_matrix, day_clustered_t, fwd_event, load_all, past_median_at,
)  # fmt: skip

HS = np.array([1, 2, 4, 8, 12, 24, 48], dtype=np.int64)
ODIR = OUT / "flow"


def dedupe(rows: np.ndarray, cols: np.ndarray, gap: int) -> np.ndarray:
    """Маска: событие оставляется, если предыдущее событие той же монеты было > gap баров назад."""
    order = np.lexsort((rows, cols))
    r, c = rows[order], cols[order]
    keep_sorted = np.ones(len(r), bool)
    same = np.r_[False, c[1:] == c[:-1]]
    # событие подавляется, если предыдущее (любое, даже подавленное) было ближе gap: «кластер» = цепочка
    d = np.r_[np.inf, np.diff(r).astype(float)]
    keep_sorted = ~(same & (d <= gap))
    keep = np.empty(len(r), bool)
    keep[order] = keep_sorted
    return keep


def build_events(p: dict, w: int, thr: float) -> dict:
    c, o = p["close"], p["open"]
    T, N = c.shape
    age = age_hours(c)
    rw = np.full(c.shape, np.nan, dtype=np.float32)
    rw[w:] = c[w:] / c[:-w] - 1
    ok = (age >= 168 + w) & p["crypto"][None, :] & np.isfinite(rw)
    cand = ok & (np.abs(rw) > thr)
    rows, cols = np.nonzero(cand)
    qv = p["qvol"]
    med = past_median_at(qv, rows, cols, 168, 100, w)
    vwin = np.zeros(len(rows))
    for s in range(w):
        vwin += np.nan_to_num(qv[rows - s, cols])
    vwin /= w
    vr = vwin / med
    ret, fund, dead = fwd_event(o, c, p["funding"], rows, cols, HS)
    bi = int(np.nonzero(p["symbols"] == "BTCUSDT")[0][0])
    bret, _, _ = fwd_event(o, c, p["funding"], rows, np.full(len(rows), bi), HS)
    adv = p["adv"][rows, cols].astype(float)
    cost = cost_matrix(p["adv"][rows, cols][None, :])[0].astype(float)
    tb = p["taker_buy_qvol"][rows, cols] / np.where(qv[rows, cols] > 0, qv[rows, cols], np.nan)
    # продолжение/экстремум внутри бара: где close относительно high-low бара
    hi, lo = p["high"][rows, cols], p["low"][rows, cols]
    pos = (c[rows, cols] - lo) / np.where(hi > lo, hi - lo, np.nan)
    return {"rows": rows, "cols": cols, "r": rw[rows, cols].astype(float), "vr": vr, "ret": ret, "fund": fund,
            "dead": dead, "bret": bret, "adv": adv, "cost": cost, "tbr": tb.astype(float), "pos": pos.astype(float),
            "ts": p["ts"][rows]}  # fmt: skip


def stats_block(ev: dict, m: np.ndarray, sign: int) -> dict:
    """sign=+1 для пампа, −1 для дампа. «cont» — доходность сделки ПО движению, «fade» — ПРОТИВ."""
    out = {"n": int(m.sum())}
    if m.sum() < 5:
        return out
    day = ev["ts"][m] // DAY
    for ih, h in enumerate(HS):
        r = ev["ret"][m, ih]
        ra = r - ev["bret"][m, ih]
        f = ev["fund"][m, ih]
        cost = 2 * ev["cost"][m]
        cont = sign * r - sign * f  # лонг на пампе платит фандинг (+rate), шорт на дампе получает
        fade = -sign * r + sign * f
        mc, tc, _ = day_clustered_t(sign * r, day)
        ma, ta, _ = day_clustered_t(sign * ra, day)
        out[f"h{h}"] = {
            "cont_mean_pct": round(mc * 100, 3), "t": round(tc, 2),
            "cont_adj_mean_pct": round(ma * 100, 3), "t_adj": round(ta, 2),
            "cont_median_pct": round(float(np.nanmedian(sign * r)) * 100, 3),
            "share_reversal": round(float(np.nanmean(sign * r < 0)), 3),
            "fade_net_pct": round(float(np.nanmean(fade - cost)) * 100, 3),
            "follow_net_pct": round(float(np.nanmean(cont - cost)) * 100, 3),
            "fund_mean_pct": round(float(np.nanmean(f)) * 100, 3),
            "dead_share": round(float(np.mean(ev["dead"][m, ih])), 4),
        }  # fmt: skip
    return out


def run() -> dict:
    t0 = time.time()
    p = load_all()
    ODIR.mkdir(parents=True, exist_ok=True)
    res = {"desc": __doc__, "hs": HS.tolist(), "tables": []}
    is_lo, is_hi, oos_lo, oos_hi = ms(IS[0]), ms(IS[1]), ms(OOS[0]), ms(OOS[1])
    for w in (1, 4):
        ev = build_events(p, w, 0.05 if w == 1 else 0.08)
        print("w", w, "candidates", len(ev["rows"]), round(time.time() - t0, 1), flush=True)
        for sign, name in ((1, "pump"), (-1, "dump")):
            xs = (0.05, 0.10, 0.20, 0.30, 0.40) if sign > 0 else (0.05, 0.10, 0.15, 0.20, 0.30)
            if w == 4:
                xs = (0.10, 0.20, 0.30, 0.50) if sign > 0 else (0.10, 0.15, 0.20, 0.30)
            for X in xs:
                for V in (0, 3, 5, 10):
                    base = (sign * ev["r"] > X) & ((ev["vr"] >= V) if V > 0 else np.isfinite(ev["r"]))
                    idx = np.nonzero(base)[0]
                    keep = np.zeros(len(base), bool)
                    keep[idx[dedupe(ev["rows"][idx], ev["cols"][idx], 24)]] = True
                    for liq_name, lm in (
                        ("all", np.ones(len(base), bool)),
                        ("adv<2M", ev["adv"] < 2e6),
                        ("adv2-10M", (ev["adv"] >= 2e6) & (ev["adv"] < 1e7)),
                        ("adv10-50M", (ev["adv"] >= 1e7) & (ev["adv"] < 5e7)),
                        ("adv>50M", ev["adv"] >= 5e7),
                    ):
                        for per, pm in (
                            ("IS", (ev["ts"] >= is_lo) & (ev["ts"] < is_hi)),
                            ("OOS", (ev["ts"] >= oos_lo) & (ev["ts"] < oos_hi)),
                        ):
                            m = keep & lm & pm
                            row = {"w": w, "side": name, "X": X, "V": V, "liq": liq_name, "period": per}
                            row.update(stats_block(ev, m, sign))
                            res["tables"].append(row)
        # профиль по годам для базовых определений
        if w == 1:
            yrs = ev["ts"].astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
            res["by_year"] = []
            for sign, name, X in ((1, "pump", 0.10), (1, "pump", 0.20), (-1, "dump", 0.10), (-1, "dump", 0.15)):
                base = (sign * ev["r"] > X) & (ev["vr"] >= 3)
                idx = np.nonzero(base)[0]
                keep = np.zeros(len(base), bool)
                keep[idx[dedupe(ev["rows"][idx], ev["cols"][idx], 24)]] = True
                for y in np.unique(yrs):
                    row = {"side": name, "X": X, "V": 3, "year": int(y)}
                    row.update(stats_block(ev, keep & (yrs == y), sign))
                    res["by_year"].append(row)
            # без дедупликации и с позицией close внутри бара (close у хая -> продолжение?)
            res["by_pos"] = []
            for sign, name, X in ((1, "pump", 0.10), (-1, "dump", 0.10)):
                base = (sign * ev["r"] > X) & (ev["vr"] >= 3)
                idx = np.nonzero(base)[0]
                keep = np.zeros(len(base), bool)
                keep[idx[dedupe(ev["rows"][idx], ev["cols"][idx], 24)]] = True
                posx = ev["pos"] if sign > 0 else 1 - ev["pos"]  # 1 = закрытие на экстремуме движения
                for lo_, hi_ in ((0, 0.5), (0.5, 0.8), (0.8, 1.01)):
                    for per, pm in (("IS", ev["ts"] < is_hi), ("OOS", ev["ts"] >= oos_lo)):
                        m = keep & (posx >= lo_) & (posx < hi_) & pm
                        row = {"side": name, "X": X, "close_pos": [lo_, hi_], "period": per}
                        row.update(stats_block(ev, m, sign))
                        res["by_pos"].append(row)
        del ev
    (ODIR / "part_b.json").write_text(json.dumps(res, indent=1, default=float))
    print("done", round(time.time() - t0, 1))
    return res


def show(res: dict | None = None) -> None:
    if res is None:
        res = json.loads((ODIR / "part_b.json").read_text())
    print("w side  X    V  liq        per   n   | cont% by h (1,4,12,24,48) [t]           | fade_net% h4/h24/h48")
    for r in res["tables"]:
        if r["liq"] != "all" or r["n"] < 5:
            continue
        cs = " ".join(f"{r[f'h{h}']['cont_mean_pct']:6.2f}[{r[f'h{h}']['t']:5.1f}]" for h in (1, 4, 12, 24, 48))
        fn = "/".join(f"{r[f'h{h}']['fade_net_pct']:.2f}" for h in (4, 24, 48))
        print(f"{r['w']} {r['side']} {r['X']:.2f} {r['V']:2d} {r['liq']:9s} {r['period']:3s} {r['n']:5d} | {cs} | {fn}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show()
    else:
        show(run())
