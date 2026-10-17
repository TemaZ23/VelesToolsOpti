"""
xsec (d): пары и относительная стоимость.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_pairs

1) ETH/BTC: спред s = ln ETH − β ln BTC (β = 1 или скользящая OLS-бета доходностей), z-скор по
   скользящему окну; режимы «возврат к среднему» и «пробой» (тренд). 1d / 4h / 1h.
2) Коррелированные пары (метод формирования/торговли): раз в 30 дней в топ-K монет по ADV выбираются P пар
   с максимальной корреляцией доходностей за окно формирования; спред с β из формирования, z по
   среднему/σ окна формирования; торговля следующие 30 дней (вход |z|>z_in, выход z→z_out, всё
   закрывается в конце периода). 1d / 4h.
3) Пиринговые корзины: остаток доходности монеты к среднему её m самых коррелированных «соседей»
   (пересчёт раз в 7 дней по прошлым 60 дням) → реверсал или моментум остатка.
4) Секторные корзины (ручная разметка): остаток к сектору и секторный моментум.
Всё — сигнал по close[t], исполнение open[t+1], издержки и фандинг как в движке.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
from numba import njit

from scripts.edge_lab.common import HOUR, cost_per_side, load_funding, load_hourly
from scripts.edge_lab.xsec import KEEP, XOUT
from scripts.edge_lab.xsec_ctx import Ctx, roll_mean
from scripts.edge_lab.xsec_data import crypto_mask, load_h4
from scripts.edge_lab.xsec_engine import backtest, ls_weights, make_exec, overlap, summarize

SECTORS = {
    "L1": "ETH SOL ADA AVAX DOT NEAR ATOM APT SUI SEI TRX ALGO EGLD FTM ICP TON XTZ EOS NEO KAVA INJ TIA HBAR XLM "
          "XRP ETC BCH LTC ZIL ONE WAVES QTUM ICX VET IOTA FLOW MINA KAS CELO ROSE KSP S BERA HYPE",
    "MEME": "DOGE 1000SHIB 1000PEPE 1000FLOKI 1000BONK WIF BOME MEME PEOPLE TURBO POPCAT BRETT NEIRO 1000SATS "
            "MEW PNUT GOAT ACT MOODENG 1000RATS ORDI TRUMP FARTCOIN PENGU SPX 1000CHEEMS 1000LUNC",
    "DEFI": "UNI AAVE SUSHI CRV COMP MKR SNX YFI 1INCH DYDX LDO PENDLE GMX CAKE BAL RUNE JUP ENA ETHFI CVX FXS "
            "RDNT JTO RAY ONDO MORPHO PERP LQTY UMA",
    "L2": "MATIC POL ARB OP STRK ZK IMX MANTA METIS LRC SKL CELR BLAST ZRO ALT DYM SCR",
    "AI": "FET AGIX OCEAN RNDR RENDER TAO WLD ARKM AI16Z VIRTUAL GRT IO NFP AIXBT PHB ATH",
    "GAME": "SAND MANA AXS GALA ENJ APE ILV ALICE YGG MAGIC GMT PIXEL PORTAL BEAMX XAI CHZ FLOW SUPER",
    "INFRA": "FIL AR STORJ LINK BAND API3 PYTH TRB ANKR THETA HNT LPT AKT RLC IOTX",
    "PRIV": "XMR ZEC DASH ZEN",
}
SECTOR_OF = {}
for _s, _v in SECTORS.items():
    for _c in _v.split():
        SECTOR_OF.setdefault(_c + "USDT", _s)


def _rec(cfg, ts, bt, bpd=1.0):
    s = summarize(ts, bt, bpd)
    return {"cfg": cfg, "IS": {k: s["IS"].get(k) for k in KEEP}, "OOS": {k: s["OOS"].get(k) for k in KEEP},
            "per_year": s["per_year"], "oos_half_years": s["oos_half_years"]}  # fmt: skip


# ----------------------------------------------------------------------------- spread state machine
@njit(cache=True)
def spread_pos(z, active, z_in, z_out, max_hold, trend):
    """Позиция по спреду (+1 — лонг спреда) по z-скору; trend=1 — торговля пробоя вместо возврата."""
    T = len(z)
    pos = np.zeros(T)
    p = 0.0
    held = 0
    for t in range(T):
        if not active[t] or not np.isfinite(z[t]):
            p = 0.0
            held = 0
            pos[t] = 0.0
            continue
        zt = z[t]
        if p == 0.0:
            if zt > z_in:
                p = 1.0 if trend else -1.0
                held = 0
            elif zt < -z_in:
                p = -1.0 if trend else 1.0
                held = 0
        else:
            held += 1
            side = p if trend else -p  # +1: entered on z>z_in
            if side > 0 and zt < z_out:
                p = 0.0
            elif side < 0 and zt > -z_out:
                p = 0.0
            elif held >= max_hold:
                p = 0.0
        pos[t] = p
    return pos


def _hourly_pair(syms=("BTCUSDT", "ETHUSDT")) -> dict:
    hs = [load_hourly(s) for s in syms]
    t0 = max(h["ts"][0] for h in hs)
    t1 = min(h["ts"][-1] for h in hs)
    ts = np.arange(t0, t1 + HOUR, HOUR, dtype=np.int64)
    T = len(ts)
    O = np.full((T, 2), np.nan)
    C = np.full((T, 2), np.nan)
    Q = np.full((T, 2), np.nan)
    F = np.zeros((T, 2))
    for j, (s, h) in enumerate(zip(syms, hs)):
        m = (h["ts"] >= t0) & (h["ts"] <= t1)
        ix = ((h["ts"][m] - t0) // HOUR).astype(np.int64)
        O[ix, j] = h["ohlc"][m, 0]
        C[ix, j] = h["ohlc"][m, 3]
        Q[ix, j] = h["qvol"][m]
        f = load_funding(s)
        b = (f["ts"] - t0) // HOUR - 1  # выплата в начале часа h → период удержания h−1
        ok = (b >= 0) & (b < T)
        np.add.at(F[:, j], b[ok].astype(np.int64), f["rate"][ok])
    adv = roll_mean(Q, 720, 1) * 24
    return {"ts": ts, "open": O, "close": C, "fund_hold": F, "cps": cost_per_side(np.nan_to_num(adv)), "adv": adv}


def ethbtc(ctx: Ctx) -> list:
    recs = []
    h4 = load_h4()
    jb, je = ctx.j_btc, ctx.j_eth
    panels = {}
    d1 = {"ts": ctx.ts, "open": ctx.O[:, [jb, je]], "close": ctx.C[:, [jb, je]], "fund_hold": ctx.d["fund_hold"][:, [jb, je]],
          "cps": ctx.ex["cps"][:, [jb, je]]}  # fmt: skip
    panels["1d"] = (d1, 1)
    adv4 = roll_mean(h4["qvol"][:, [jb, je]], 180, 1) * 6
    panels["4h"] = ({"ts": h4["ts"], "open": h4["open"][:, [jb, je]], "close": h4["close"][:, [jb, je]],
                     "fund_hold": h4["fund_hold"][:, [jb, je]], "cps": cost_per_side(np.nan_to_num(adv4))}, 6)  # fmt: skip
    panels["1h"] = (_hourly_pair(), 24)
    for freq, (ex, bpd) in panels.items():
        lc = np.log(ex["close"])
        lb, le = lc[:, 0], lc[:, 1]
        rb = np.r_[np.nan, np.diff(lb)]
        re_ = np.r_[np.nan, np.diff(le)]
        for wd, hedge in itertools.product((10, 30, 90), ("ratio", "ols")):
            n = wd * bpd
            mb, me = roll_mean(lb[:, None], n)[:, 0], roll_mean(le[:, None], n)[:, 0]
            vbb = roll_mean((lb * lb)[:, None], n)[:, 0] - mb * mb
            vee = roll_mean((le * le)[:, None], n)[:, 0] - me * me
            veb = roll_mean((le * lb)[:, None], n)[:, 0] - me * mb
            if hedge == "ratio":
                beta = np.ones(len(lb))
            else:
                mrb = roll_mean(rb[:, None], n)[:, 0]
                mre = roll_mean(re_[:, None], n)[:, 0]
                cov = roll_mean((rb * re_)[:, None], n)[:, 0] - mrb * mre
                var = roll_mean((rb * rb)[:, None], n)[:, 0] - mrb * mrb
                beta = np.clip(cov / var, 0.2, 3.0)
            s = le - beta * lb
            ms_ = me - beta * mb
            vs = vee + beta * beta * vbb - 2 * beta * veb
            with np.errstate(all="ignore"):
                z = (s - ms_) / np.sqrt(np.maximum(vs, 1e-12))
            active = np.isfinite(z)
            for z_in, z_out, trend in itertools.product((1.5, 2.0, 2.5), (0.0, 0.5), (0, 1)):
                pos = spread_pos(z, active, z_in, z_out, n, trend)
                b = np.nan_to_num(beta, nan=1.0)
                W = np.zeros((len(pos), 2))
                W[:, 1] = pos / (1 + b)
                W[:, 0] = -pos * b / (1 + b)
                bt = backtest(W, ex)
                cfg = {"hyp": f"d_ethbtc_{freq}", "freq": freq, "win_days": wd, "hedge": hedge, "z_in": z_in,
                       "z_out": z_out, "mode": "trend" if trend else "revert"}  # fmt: skip
                recs.append(_rec(cfg, ex["ts"], bt, bpd))
        print("ethbtc", freq, "done", flush=True)
    return recs


# ----------------------------------------------------------------------------- correlated pairs
def corr_pairs(ctx_d: dict, freq: str, bpd: int) -> list:
    """ctx_d: ts, open, close, fund_hold, cps, adv (по барам), symbols, crypto, age_days."""
    recs = []
    lc = np.log(ctx_d["close"])
    T, N = lc.shape
    r = np.vstack([np.full((1, N), np.nan), np.diff(lc, axis=0)])
    adv = ctx_d["adv"]
    alive = ~np.isnan(ctx_d["close"])
    age = np.cumsum(alive, axis=0) / bpd
    crypto = ctx_d["crypto"]
    step = 30 * bpd
    for K, wf in itertools.product((30, 50), (60, 120)):
        nf = wf * bpd
        forms = []  # (t_form, list of (a, b, beta, mu, sd, corr))
        for tf in range(nf, T - 1, step):
            u = alive[tf] & crypto & (age[tf] >= wf + 10) & (np.nan_to_num(adv[tf]) >= 5e6)
            cand = np.where(u)[0]
            if len(cand) < 10:
                forms.append((tf, []))
                continue
            cand = cand[np.argsort(-adv[tf, cand])][:K]
            R = r[tf - nf + 1 : tf + 1, cand]
            ok = np.isfinite(R).mean(axis=0) >= 0.95
            cand, R = cand[ok], R[:, ok]
            R = np.nan_to_num(R)
            Rz = (R - R.mean(0)) / (R.std(0) + 1e-12)
            Cm = Rz.T @ Rz / len(R)
            iu = np.triu_indices(len(cand), 1)
            order = np.argsort(-Cm[iu])
            pairs = []
            for o in order[:20]:
                a, b = cand[iu[0][o]], cand[iu[1][o]]
                ra, rb = np.nan_to_num(r[tf - nf + 1 : tf + 1, a]), np.nan_to_num(r[tf - nf + 1 : tf + 1, b])
                beta = float(np.cov(ra, rb)[0, 1] / np.var(rb, ddof=1))
                if beta <= 0.2 or beta > 3:
                    continue
                sp = lc[tf - nf + 1 : tf + 1, a] - beta * lc[tf - nf + 1 : tf + 1, b]
                pairs.append((a, b, beta, float(np.nanmean(sp)), float(np.nanstd(sp)), float(Cm[iu][o])))
            forms.append((tf, pairs))
        for P, z_in, z_out in itertools.product((5, 10, 20), (1.5, 2.0, 2.5), (0.0, 0.5)):
            W = np.zeros((T, N))
            for tf, pairs in forms:
                t_end = min(tf + step, T - 1)
                for a, b, beta, mu, sd, _ in pairs[:P]:
                    seg = slice(tf + 1, t_end + 1)  # сигнал с бара tf+1 (закрытие), закрытие позиции на t_end
                    sp = lc[seg, a] - beta * lc[seg, b]
                    z = (sp - mu) / sd
                    act = np.isfinite(z)
                    act[-1] = False  # конец торгового периода — позиция закрывается
                    pos = spread_pos(z, act, z_in, z_out, 10**9, 0)
                    W[seg, a] += pos / (1 + beta) / P
                    W[seg, b] -= pos * beta / (1 + beta) / P
            bt = backtest(W, ctx_d)
            cfg = {"hyp": f"d_pairs_{freq}", "freq": freq, "K": K, "form_days": wf, "P": P, "z_in": z_in, "z_out": z_out}
            recs.append(_rec(cfg, ctx_d["ts"], bt, bpd))
        print("pairs", freq, K, wf, "done", flush=True)
    return recs


# ----------------------------------------------------------------------------- peer / sector baskets
def peer_resid(ctx: Ctx) -> list:
    recs = []
    T, N = ctx.T, ctx.N
    lc = ctx.lc
    r = ctx.r1
    for uni in ("adv10", "top50"):
        u = ctx.universe(**({"min_adv": 1e7, "min_age": 60} if uni == "adv10" else {"min_adv": 5e6, "min_age": 60, "top": 50}))
        peers = {m: np.full((T, N, m), -1, np.int64) for m in (3, 5, 10)}
        cur = {m: np.full((N, m), -1, np.int64) for m in (3, 5, 10)}
        for t in range(T):
            if t % 7 == 0 and t >= 60:
                cand = np.where(u[t])[0]
                for m in cur:
                    cur[m][:] = -1
                if len(cand) > 12:
                    R = r[t - 59 : t + 1, cand]
                    ok = np.isfinite(R).mean(0) >= 0.9
                    cand, R = cand[ok], np.nan_to_num(R[:, ok])
                    Rz = (R - R.mean(0)) / (R.std(0) + 1e-12)
                    Cm = Rz.T @ Rz / len(R)
                    np.fill_diagonal(Cm, -np.inf)
                    o = np.argsort(-Cm, axis=1)
                    for m in cur:
                        cur[m][cand] = cand[o[:, :m]]
            for m in cur:
                peers[m][t] = cur[m]
        for m in (3, 5, 10):
            pm = peers[m]
            for k in (1, 3, 7):
                rk = np.full_like(lc, np.nan)
                rk[k:] = lc[k:] - lc[:-k]
                pr = np.where(pm >= 0, np.take_along_axis(rk[:, None, :].repeat(1, axis=1), np.maximum(pm, 0).reshape(T, 1, -1), axis=2).reshape(T, N, m), np.nan)
                with np.errstate(all="ignore"):
                    e = rk - np.nanmean(pr, axis=2)
                for sign, H, q in itertools.product((-1, 1), (1, 3, 7), (0.1, 0.2)):
                    W = overlap(ls_weights(sign * e, u, np.ones((T, N)), q, 20, 0, 0), H)
                    bt = backtest(W, ctx.ex)
                    cfg = {"hyp": "d_peer_rev" if sign < 0 else "d_peer_mom", "m": m, "k": k, "H": H, "q": q, "uni": uni}
                    recs.append(_rec(cfg, ctx.ts, bt))
        print("peer", uni, "done", flush=True)
    return recs


def sector(ctx: Ctx) -> list:
    recs = []
    T, N = ctx.T, ctx.N
    lc = ctx.lc
    secs = sorted(SECTORS)
    sec_idx = np.array([secs.index(SECTOR_OF[s]) if s in SECTOR_OF else -1 for s in ctx.symbols])
    u = ctx.universe(min_adv=1e7, min_age=60) & (sec_idx >= 0)[None, :]
    for k in (1, 3, 7, 14, 28):
        rk = np.full_like(lc, np.nan)
        rk[k:] = lc[k:] - lc[:-k]
        x = np.where(u, rk, np.nan)
        smean = np.full((T, len(secs)), np.nan)
        scnt = np.zeros((T, len(secs)))
        for si in range(len(secs)):
            cols = sec_idx == si
            with np.errstate(all="ignore"):
                smean[:, si] = np.nanmean(x[:, cols], axis=1)
            scnt[:, si] = np.isfinite(x[:, cols]).sum(axis=1)
        # остаток к сектору (среднее сектора без самой монеты)
        sm = smean[:, np.maximum(sec_idx, 0)]
        sc = scnt[:, np.maximum(sec_idx, 0)]
        with np.errstate(all="ignore"):
            loo = (sm * sc - np.nan_to_num(x)) / (sc - 1)
        e = np.where((sc >= 3) & u, x - loo, np.nan)
        if k <= 7:
            for sign, H in itertools.product((-1, 1), (1, 3, 7)):
                W = overlap(ls_weights(sign * e, u, np.ones((T, N)), 0.2, 20, 0, 0), H)
                bt = backtest(W, ctx.ex)
                cfg = {"hyp": "d_sector_resid_rev" if sign < 0 else "d_sector_resid_mom", "k": k, "H": H, "q": 0.2}
                recs.append(_rec(cfg, ctx.ts, bt))
        # секторный моментум: лонг монет лучших nsec секторов, шорт худших (сектора с ≥3 монетами)
        if k >= 7:
            for nsec, H in itertools.product((1, 2), (1, 7)):
                Ws = np.zeros((T, N))
                for t in range(T):
                    okS = np.where((scnt[t] >= 3) & np.isfinite(smean[t]))[0]
                    if len(okS) < 2 * nsec + 1:
                        continue
                    o = okS[np.argsort(smean[t, okS])]
                    for side, group in ((-1, o[:nsec]), (1, o[-nsec:])):
                        cols = np.where(u[t] & np.isin(sec_idx, group) & np.isfinite(x[t]))[0]
                        if len(cols):
                            Ws[t, cols] = side * 0.5 / len(cols)
                W = overlap(Ws, H)
                bt = backtest(W, ctx.ex)
                cfg = {"hyp": "d_sector_mom", "L": k, "nsec": nsec, "H": H}
                recs.append(_rec(cfg, ctx.ts, bt))
    print("sector done", flush=True)
    return recs


def main() -> None:
    t0 = time.time()
    ctx = Ctx(0)
    out = {}
    out["ethbtc"] = ethbtc(ctx)
    print(round(time.time() - t0), flush=True)
    d1 = dict(ctx.ex)
    d1.update(adv=ctx.adv, crypto=ctx.crypto, symbols=ctx.symbols)
    out["pairs_1d"] = corr_pairs(d1, "1d", 1)
    print(round(time.time() - t0), flush=True)
    h4 = load_h4()
    adv4 = roll_mean(h4["qvol"], 180, 1) * 6
    d4 = make_exec(h4, adv4)
    d4.update(adv=adv4, crypto=crypto_mask(h4["symbols"]), symbols=h4["symbols"])
    out["pairs_4h"] = corr_pairs(d4, "4h", 6)
    del h4, d4
    print(round(time.time() - t0), flush=True)
    out["peer"] = peer_resid(ctx)
    out["sector"] = sector(ctx)
    recs = [r for v in out.values() for r in v]
    (XOUT / "part_d.json").write_text(json.dumps(recs))
    print("saved", len(recs), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
