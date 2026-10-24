"""
xsec: диагностика кандидатов суточной кросс-секции (проверки на артефакты).

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_diag '<json cfg>'

Для конфигурации: разложение по ногам (лонг/шорт), вклад топ-монет, пессимистичный делистинг (−50%),
издержки ×2, исполнение с задержкой 1 час (сигнал 00:00, сделка 01:00), другие границы суток
(08:00 / 16:00 UTC), случайные сигналы с той же структурой.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from scripts.edge_lab.common import IS, OOS, ms
from scripts.edge_lab.xsec import build_W
from scripts.edge_lab.xsec_ctx import Ctx
from scripts.edge_lab.xsec_engine import backtest, summarize, window_stats

_CTX: dict = {}


def ctx(off: int) -> Ctx:
    if off not in _CTX:
        _CTX[off] = Ctx(off)
    return _CTX[off]


def short(s: dict) -> dict:
    keys = ("sharpe", "cagr_pct", "max_dd_pct", "gross_sharpe", "ann_ret_pct", "ann_price_pct", "ann_fund_pct",
            "ann_cost_pct", "turnover_per_day", "avg_hold_days", "avg_positions", "forced_exits", "tstat")  # fmt: skip
    return {k: s.get(k) for k in keys}


def both(ts, bt) -> dict:
    return {"IS": short(window_stats(ts, bt, IS)), "OOS": short(window_stats(ts, bt, OOS))}


def contributions(c: Ctx, W: np.ndarray, window) -> dict:
    """Вклад монет в ценовой P&L (сумма w·r за окно) — проверка концентрации."""
    i0, i1 = int(np.searchsorted(c.ts, ms(window[0]))), int(np.searchsorted(c.ts, ms(window[1])))
    i0 = max(i0, 1)
    O = c.O
    r = np.full_like(O, np.nan)
    r[:-1] = O[1:] / O[:-1] - 1
    r = np.where(np.isfinite(r), r, np.nan_to_num(c.C / O - 1))
    pnl = np.zeros(c.N)
    pnl += np.nansum(W[i0 - 1 : i1 - 1] * np.nan_to_num(r[i0:i1]), axis=0)
    tot = pnl.sum()
    o = np.argsort(-pnl)
    top = [(str(c.symbols[j]), round(float(pnl[j]) * 100, 2)) for j in o[:8]]
    bot = [(str(c.symbols[j]), round(float(pnl[j]) * 100, 2)) for j in o[-5:]]
    return {"total_pct": round(float(tot) * 100, 1), "top10_share": round(float(pnl[o[:10]].sum() / tot), 2) if tot else None,
            "top": top, "bottom": bot}  # fmt: skip


def diag(cfg: dict, n_random: int = 5) -> dict:
    c0 = ctx(0)
    W = build_W(c0, cfg)
    out = {"cfg": cfg}
    bt = backtest(W, c0.ex)
    out["base"] = both(c0.ts, bt)
    out["pessimistic_delist"] = both(c0.ts, backtest(W, c0.ex, pess=0.5))
    ex2 = dict(c0.ex)
    ex2["cps"] = c0.ex["cps"] * 2
    out["cost_x2"] = both(c0.ts, backtest(W, ex2))
    WL, WS = np.where(W > 0, W, 0), np.where(W < 0, W, 0)
    out["long_leg"] = both(c0.ts, backtest(WL, c0.ex))
    out["short_leg"] = both(c0.ts, backtest(WS, c0.ex))
    out["contrib_IS"] = contributions(c0, W, IS)
    out["contrib_OOS"] = contributions(c0, W, OOS)
    # 1h execution delay: signal at 00:00 close, trade at 01:00 open (panel off1)
    c1 = ctx(1)
    out["exec_delay_1h"] = both(c1.ts, backtest(W[: c1.T], c1.ex))
    for off in (8, 16):
        cc = ctx(off)
        out[f"day_boundary_{off}h"] = both(cc.ts, backtest(build_W(cc, cfg), cc.ex))
    s = summarize(c0.ts, bt)
    out["per_year"] = s["per_year"]
    out["oos_half_years"] = s["oos_half_years"]
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    print(json.dumps(diag(cfg), indent=1, ensure_ascii=False))
