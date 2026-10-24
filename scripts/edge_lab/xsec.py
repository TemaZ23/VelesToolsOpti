"""
Семейство xsec — кросс-секционный моментум/реверсал, «лотерейные» и прочие факторы, стат-арбитраж.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_data        # один раз: подготовка панелей
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec a           # (a) моментум/реверсал, суточная кросс-секция
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec c           # (c) MAX / низкая волатильность / размер / неликвидность / возраст
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_intraday    # (b) остаточный реверсал 1h / 4h / 1d
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_pairs       # (d) пары и относительная стоимость
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec report      # выбор по IS, устойчивость, итоговые файлы

Все конфигурации и их IS/OOS метрики пишутся в output/edge_lab/xsec/part_<x>.json; выбор параметров —
только по IS (2020–2023), OOS (2024-01 .. 2026-09) только отчитывается.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from scripts.edge_lab.common import OUT
from scripts.edge_lab.xsec_ctx import Ctx, roll_max, roll_mean
from scripts.edge_lab.xsec_engine import backtest, beta_hedge, ls_weights, overlap, summarize

XOUT = OUT / "xsec"
KEEP = ("sharpe", "cagr_pct", "max_dd_pct", "avg_month_pct", "worst_month_pct", "pos_months_pct", "trades",
        "gross_sharpe", "tstat", "ann_ret_pct", "ann_gross_pct", "ann_price_pct", "ann_fund_pct", "ann_cost_pct",
        "turnover_per_day", "avg_exposure", "active_share", "avg_hold_days", "avg_positions", "forced_exits")  # fmt: skip

UNIS = {"adv10": dict(min_adv=1e7, min_age=60), "top50": dict(min_adv=5e6, min_age=60, top=50),
        "adv5": dict(min_adv=5e6, min_age=60), "adv20": dict(min_adv=2e7, min_age=60)}  # fmt: skip


def uni_of(ctx: Ctx, cfg: dict):
    if "min_adv" in cfg:  # явное задание (используется в проверке устойчивости ±25%)
        return ctx.universe(min_adv=cfg["min_adv"], min_age=cfg.get("min_age", 60), top=cfg.get("top", 0))
    if cfg["uni"] in UNIS:
        return ctx.universe(**UNIS[cfg["uni"]])
    # age-factor universes: "age<min_age>_adv<M>"
    a, m = cfg["uni"][3:].split("_adv")
    return ctx.universe(min_adv=float(m) * 1e6, min_age=int(a))


def score(ctx: Ctx, cfg: dict) -> np.ndarray:
    f = cfg["fac"]
    key = ("score", f, cfg.get("L"), cfg.get("skip"), cfg.get("n"))
    if key in ctx._cache:
        return ctx._cache[key]
    if f == "mom":
        S = ctx.mom(cfg["L"], cfg["skip"])
    elif f == "max":
        S = roll_max(ctx.r1, cfg["n"])
    elif f == "vol":
        S = ctx.vol(cfg["n"])
    elif f == "size":
        with np.errstate(all="ignore"):
            S = np.log(ctx.adv)
    elif f == "illiq":
        with np.errstate(all="ignore"):
            S = roll_mean(np.abs(ctx.r1) / np.where(ctx.QV > 0, ctx.QV, np.nan), cfg.get("n") or 30) * 1e6
    elif f == "age":
        S = ctx.age.astype(float)
    else:
        raise ValueError(f)
    ctx._cache[key] = S
    return S


def build_W(ctx: Ctx, cfg: dict) -> np.ndarray:
    S = score(ctx, cfg) * cfg.get("sign", 1)
    u = uni_of(ctx, cfg)
    side = 1.0 / ctx.vol(30) if cfg.get("wt", "eq") == "iv" else np.ones_like(S)
    W = ls_weights(S, u, side, cfg["q"], cfg.get("min_n", 20), cfg.get("mode", 0), cfg.get("kmax", 0))
    W = overlap(W, cfg["H"])
    if cfg.get("rebal", 0) > 1:  # ребаланс раз в rebal дней (держим портфель последней даты ребаланса)
        R = cfg["rebal"]
        idx = (np.arange(len(W)) // R) * R
        W = W[idx]
    if cfg.get("bn", 0):
        W = beta_hedge(W, ctx.beta(60), ctx.j_btc)
    return W


def run_cfg(ctx: Ctx, cfg: dict, pess: float = 0.0, full: bool = False):
    W = build_W(ctx, cfg)
    bt = backtest(W, ctx.ex, pess=pess)
    s = summarize(ctx.ts, bt)
    rec = {"cfg": cfg, "IS": {k: s["IS"].get(k) for k in KEEP}, "OOS": {k: s["OOS"].get(k) for k in KEEP},
           "per_year": s["per_year"], "oos_half_years": s["oos_half_years"]}  # fmt: skip
    if full:
        return rec, W, bt
    return rec


def grid_a():
    for L, skip, H, q, wt, bn, uni, sign in itertools.product(
        (1, 3, 7, 14, 28, 56), (0, 1), (1, 3, 7, 14), (0.1, 0.2, 0.3), ("eq", "iv"), (0, 1), ("adv10", "top50"), (1, -1)
    ):
        yield {"hyp": "a_mom" if sign > 0 else "a_rev", "fac": "mom", "L": L, "skip": skip, "H": H, "q": q, "wt": wt,
               "bn": bn, "uni": uni, "sign": sign}  # fmt: skip


def grid_c():
    for n, (mode, sign), q, H, bn, uni in itertools.product(
        (7, 14, 28), ((0, -1), (1, 1)), (0.1, 0.2, 0.3), (1, 7, 14), (0, 1), ("adv10", "top50")
    ):
        yield {"hyp": "c_max", "fac": "max", "n": n, "mode": mode, "sign": sign, "q": q, "H": H, "bn": bn, "uni": uni}
    for n, q, H, bn, uni in itertools.product((14, 28, 56), (0.1, 0.2, 0.3), (1, 7, 14), (0, 1), ("adv10", "top50")):
        yield {"hyp": "c_lowvol", "fac": "vol", "n": n, "sign": -1, "q": q, "H": H, "bn": bn, "uni": uni}
    for q, H, bn, uni in itertools.product((0.1, 0.2, 0.3), (1, 7, 14), (0, 1), ("adv5", "adv10", "adv20")):
        yield {"hyp": "c_size", "fac": "size", "sign": -1, "q": q, "H": H, "bn": bn, "uni": uni}
        yield {"hyp": "c_illiq", "fac": "illiq", "sign": 1, "q": q, "H": H, "bn": bn, "uni": uni}
    for (mode, bn), q, H, a, m in itertools.product(
        ((0, 0), (0, 1), (2, 1)), (0.1, 0.2, 0.3), (1, 7, 14), (14, 30, 60), (5, 10)
    ):
        yield {"hyp": "c_age", "fac": "age", "sign": 1 if mode == 0 else -1, "mode": mode, "q": q, "H": H, "bn": bn,
               "uni": f"age{a}_adv{m}"}  # fmt: skip


def run_part(part: str) -> None:
    XOUT.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(0)
    grid = {"a": grid_a, "c": grid_c}[part]
    recs, t0 = [], time.time()
    for k, cfg in enumerate(grid()):
        recs.append(run_cfg(ctx, cfg))
        if k % 200 == 0:
            print(k, round(time.time() - t0, 1), cfg, recs[-1]["IS"]["sharpe"], recs[-1]["OOS"]["sharpe"], flush=True)
    (XOUT / f"part_{part}.json").write_text(json.dumps(recs))
    print("saved", len(recs), "configs in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "a"
    if arg in ("a", "c"):
        run_part(arg)
    elif arg == "report":
        from scripts.edge_lab.xsec_report import main as report_main

        report_main()
