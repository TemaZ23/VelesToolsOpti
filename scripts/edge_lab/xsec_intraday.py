"""
xsec (b): краткосрочный остаточный реверсал (residual reversal) на 1h / 4h / 1d.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_intraday 1h     # часовая панель грузится один раз
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_intraday 4h
    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_intraday 1d

Скор: остаток e_i = r_i(k) − β_i·r_BTC(k), r(k) — лог-доходность за k последних баров, β — скользящая
регрессия баровых доходностей монеты на BTC по прошлому окну. Реверсал: лонг монеты с наименьшим
остатком, шорт — с наибольшим (квантиль q), удержание H баров перекрывающимися портфелями.
Для сравнения — «сырой» реверсал (без вычитания беты). Исполнение по open[t+1], издержки — taker+слиппедж.
Дополнительно для IS-лучших: верхняя граница «maker без слиппеджа» (НЕ доказательство торгуемости)
и задержка исполнения на 1 бар.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import cost_per_side, panel
from scripts.edge_lab.xsec import KEEP, XOUT
from numba import njit
from scripts.edge_lab.xsec_data import crypto_mask, load_daily, load_h4
from scripts.edge_lab.xsec_engine import backtest, ls_weights, overlap, summarize

BARS_PER_DAY = {"1h": 24, "4h": 6, "1d": 1}


@njit(cache=True)
def roll_mean_nb(x, n):
    """Скользящее среднее за n баров включая текущий (NaN пропускаются; NaN если валидных нет)."""
    T, N = x.shape
    out = np.full((T, N), np.nan, np.float32)
    for i in range(N):
        s = 0.0
        k = 0
        for t in range(T):
            v = x[t, i]
            if np.isfinite(v):
                s += v
                k += 1
            if t >= n:
                v0 = x[t - n, i]
                if np.isfinite(v0):
                    s -= v0
                    k -= 1
            if k > 0:
                out[t, i] = s / k
    return out


@njit(cache=True)
def roll_beta_nb(lc, j, n, min_cnt):
    """β доходностей монеты (разности lc) к доходностям колонки j за n прошлых баров (включая текущий)."""
    T, N = lc.shape
    out = np.full((T, N), np.nan, np.float32)
    y = np.full(T, np.nan)
    for t in range(1, T):
        y[t] = lc[t, j] - lc[t - 1, j]
    for i in range(N):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        syy = 0.0
        k = 0
        x = np.full(T, np.nan)
        for t in range(1, T):
            x[t] = lc[t, i] - lc[t - 1, i]
        for t in range(1, T):
            if np.isfinite(x[t]) and np.isfinite(y[t]):
                sx += x[t]
                sy += y[t]
                sxy += x[t] * y[t]
                syy += y[t] * y[t]
                k += 1
            if t - n >= 1:
                t0 = t - n
                if np.isfinite(x[t0]) and np.isfinite(y[t0]):
                    sx -= x[t0]
                    sy -= y[t0]
                    sxy -= x[t0] * y[t0]
                    syy -= y[t0] * y[t0]
                    k -= 1
            if k >= min_cnt:
                vy = syy / k - (sy / k) ** 2
                if vy > 0:
                    out[t, i] = (sxy / k - sx / k * sy / k) / vy
    return out


def load(freq: str) -> dict:
    if freq == "1h":
        p = panel("1h")
        f = np.nan_to_num(p["funding"])
        fh = np.vstack([f[1:], np.zeros((1, f.shape[1]), f.dtype)])  # выплата в начале часа h+1 → период h
        d = {"ts": p["ts"], "symbols": p["symbols"], "open": p["open"], "close": p["close"], "qvol": p["qvol"],
             "fund_hold": fh}  # fmt: skip
        del p
    elif freq == "4h":
        d = load_h4()
    else:
        d = load_daily(0)
    return d


class ICtx:
    def __init__(self, freq: str):
        self.freq = freq
        d = load(freq)
        self.bpd = BARS_PER_DAY[freq]
        self.ts, self.symbols = d["ts"], d["symbols"]
        self.O, self.C = d["open"], d["close"]
        self.T, self.N = self.C.shape
        self.j_btc = int(np.where(self.symbols == "BTCUSDT")[0][0])
        alive = ~np.isnan(self.C)
        self.alive = alive
        self.age_days = np.cumsum(alive, axis=0, dtype=np.int32) / self.bpd
        self.crypto = crypto_mask(self.symbols)
        self.adv = roll_mean_nb(d["qvol"], 30 * self.bpd) * np.float32(self.bpd)
        dt = np.float32 if freq == "1h" else np.float64
        self.ex = {"ts": self.ts, "open": self.O.astype(dt, copy=False), "close": self.C.astype(dt, copy=False),
                   "fund_hold": d["fund_hold"].astype(dt, copy=False),
                   "cps": cost_per_side(np.nan_to_num(self.adv)).astype(dt)}  # fmt: skip
        with np.errstate(all="ignore"):
            self.lc = np.log(self.C).astype(np.float32 if freq == "1h" else np.float64)
        nb = {"1h": 168, "4h": 180, "1d": 60}[freq]  # 7 дней / 30 дней / 60 дней
        self.beta = roll_beta_nb(self.lc, self.j_btc, nb, int(0.7 * nb))
        self._u: dict = {}

    def universe(self, uni: str):
        if uni not in self._u:
            if uni == "top50":
                u = self.alive & self.crypto & (self.age_days >= 60) & (np.nan_to_num(self.adv) >= 5e6)
                a = np.where(u, self.adv, -np.inf)
                rank = np.argsort(np.argsort(-a, axis=1), axis=1)
                u &= rank < 50
            else:
                m = {"adv20": 2e7, "adv50": 5e7}[uni]
                u = self.alive & self.crypto & (self.age_days >= 60) & (np.nan_to_num(self.adv) >= m)
            self._u[uni] = u
        return self._u[uni]

    def score(self, k: int, resid: bool):
        lc = self.lc
        rk = np.full_like(lc, np.nan)
        rk[k:] = lc[k:] - lc[:-k]
        if resid:
            b = np.where(np.isfinite(self.beta), np.clip(self.beta, 0, 3), 1.0)
            rk = rk - b * rk[:, [self.j_btc]]
        return -rk  # fade: high score = most negative residual → long


def grid(freq: str):
    ks = {"1h": (1, 4, 24), "4h": (1, 3, 6), "1d": (1, 2, 3)}[freq]
    Hs = {"1h": (1, 4, 12, 24), "4h": (1, 3, 6), "1d": (1, 2, 3, 5)}[freq]
    for k, H, q, uni, resid in itertools.product(ks, Hs, (0.1, 0.2), ("adv20", "top50"), (1, 0)):
        yield {"hyp": f"b_resrev_{freq}", "freq": freq, "k": k, "H": H, "q": q, "uni": uni, "resid": resid}


def run_cfg(c: ICtx, cfg: dict, cache: dict, full=False, delay=0, maker=False):
    key = (cfg["k"], cfg["resid"])
    if key not in cache:
        cache.clear()
        cache[key] = c.score(cfg["k"], bool(cfg["resid"]))
    S = cache[key]
    u = c.universe(cfg["uni"])
    W = ls_weights(S, u, np.ones(S.shape), cfg["q"], 20, 0, 0)
    W = overlap(W, cfg["H"])
    if delay:
        W = np.vstack([np.zeros((delay, c.N)), W[:-delay]])
    ex = c.ex
    if maker:
        ex = dict(ex)
        ex["cps"] = np.full_like(ex["cps"], 0.0002)
    bt = backtest(W, ex)
    s = summarize(c.ts, bt, c.bpd)
    rec = {"cfg": cfg, "IS": {k: s["IS"].get(k) for k in KEEP}, "OOS": {k: s["OOS"].get(k) for k in KEEP},
           "per_year": s["per_year"], "oos_half_years": s["oos_half_years"]}  # fmt: skip
    return (rec, W, bt) if full else rec


def main(freq: str) -> None:
    t0 = time.time()
    c = ICtx(freq)
    print("ctx", freq, round(time.time() - t0, 1), flush=True)
    recs, cache = [], {}
    for i, cfg in enumerate(grid(freq)):
        recs.append(run_cfg(c, cfg, cache))
        r = recs[-1]
        print(i, round(time.time() - t0), cfg["k"], cfg["H"], cfg["q"], cfg["uni"], cfg["resid"], "IS", r["IS"]["sharpe"],
              r["IS"]["gross_sharpe"], r["IS"]["ann_gross_pct"], r["IS"]["ann_cost_pct"], "OOS", r["OOS"]["sharpe"],
              r["OOS"]["gross_sharpe"], flush=True)  # fmt: skip
    # extras for the IS-best config (by net Sharpe) and by gross Sharpe
    best = max(recs, key=lambda r: r["IS"]["sharpe"])
    bestg = max(recs, key=lambda r: r["IS"]["gross_sharpe"])
    extra = {}
    for tag, r in (("best_net", best), ("best_gross", bestg)):
        cfg = r["cfg"]
        extra[tag] = {"cfg": cfg, "base": {"IS": r["IS"], "OOS": r["OOS"]}}
        for nm, kw in (("maker_upper_bound", dict(maker=True)), ("delay1", dict(delay=1))):
            rr = run_cfg(c, cfg, cache, **kw)
            extra[tag][nm] = {"IS": rr["IS"], "OOS": rr["OOS"]}
    (XOUT / f"part_b_{freq}.json").write_text(json.dumps({"configs": recs, "extra": extra}))
    print("saved", len(recs), round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1d")
