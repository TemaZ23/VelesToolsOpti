"""
Семейство «trend»: диверсифицированный тайм-сериес моментум с таргетированием волатильности,
фильтр режима BTC (часть c). Часть (b) — лид-лаг BTC → альты — в trend_leadlag.py.

    PYTHONPATH=. python3 -m scripts.edge_lab.trend grid     # (a) полный перебор, part_a.json
    PYTHONPATH=. python3 -m scripts.edge_lab.trend regime   # (c) фильтр режима, part_c.json
    PYTHONPATH=. python3 -m scripts.edge_lab.trend robust   # проверки выбранного, part_robust.json
    PYTHONPATH=. python3 -m scripts.edge_lab.trend final    # results.json, best_daily.npz

Выбор параметров — только по IS Sharpe (2020-01-01 … 2023-12-31). OOS (2024-01 … 2026-09) только
отчётный. Веса — доля капитала (номинал перпа / капитал); валовая экспозиция = Σ|w|.
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

from scripts.edge_lab.common import DAY, IS, OOS, OUT, metrics, ms, per_year, window_idx
from scripts.edge_lab.trend_engine import (
    TCACHE, daily_data, donchian_state, ema_cols, ewm_vol, hist_port_vol, run, topk_mask, weekday,
)  # fmt: skip

ODIR = OUT / "trend"
ODIR.mkdir(parents=True, exist_ok=True)

SIG_FLOOR = 0.30  # нижняя граница годовой волатильности монеты при сайзинге
TV_COIN = 0.40  # целевая волатильность одной монеты в режиме "iv" (её доля капитала = 1/K)
GCAP_IV = 1.5
TV_PORT = 0.30  # целевая волатильность портфеля в режиме "pv"
GCAP_PV = 2.0
PV_WIN = 60


# ----------------------------------------------------------------------------------------------
# сигналы: S[t] в [-1, 1] по данным до close[t] включительно; NaN → 0 (нет сигнала)


def _sign(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.sign(x))


def mom(C: np.ndarray, L: int) -> np.ndarray:
    out = np.full_like(C, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[L:] = C[L:] / C[:-L] - 1
    return out


def build_signals(d: dict, hl: float = 20.0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    C = d["C"]
    sig_ann = ewm_vol(d["rcc"], hl, 20)
    sd = sig_ann / np.sqrt(365.0)
    S: dict[str, np.ndarray] = {}
    for L in (10, 20, 30, 60, 90, 120):
        S[f"mom{L}"] = _sign(mom(C, L))
    for L in (20, 60, 120):
        with np.errstate(invalid="ignore", divide="ignore"):
            z = mom(C, L) / (sd * np.sqrt(L))
        S[f"tz{L}"] = np.clip(np.nan_to_num(z), -1, 1)
    emas = {n: ema_cols(C, n) for n in (5, 10, 20, 30, 40, 80, 120)}
    for f, s in ((5, 20), (10, 40), (20, 80), (30, 120)):
        S[f"ema{f}_{s}"] = _sign(emas[f] - emas[s])
    S["multi_mom"] = (S["mom20"] + S["mom60"] + S["mom120"]) / 3
    S["multi_ema"] = (S["ema10_40"] + S["ema20_80"] + S["ema30_120"]) / 3
    for n in (20, 55):
        S[f"donch{n}"] = np.nan_to_num(donchian_state(C, n, n // 2, True))
    return S, sig_ann


def sig_variant(d: dict, name: str, sd: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Сигнал с изменённым окном (для проверок ±25%). name — базовое имя."""
    C = d["C"]
    if name.startswith("mom"):
        return _sign(mom(C, max(2, round(int(name[3:]) * scale))))
    if name.startswith("tz"):
        L = max(2, round(int(name[2:]) * scale))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(np.nan_to_num(mom(C, L) / (sd * np.sqrt(L))), -1, 1)
    if name.startswith("ema"):
        f, s = (int(x) for x in name[3:].split("_"))
        f, s = max(2, round(f * scale)), max(3, round(s * scale))
        return _sign(ema_cols(C, f) - ema_cols(C, s))
    if name == "multi_mom":
        return sum(_sign(mom(C, round(L * scale))) for L in (20, 60, 120)) / 3
    if name == "multi_ema":
        return sum(_sign(ema_cols(C, round(f * scale)) - ema_cols(C, round(s * scale)))
                   for f, s in ((10, 40), (20, 80), (30, 120))) / 3  # fmt: skip
    if name.startswith("donch"):
        n = max(3, round(int(name[5:]) * scale))
        return np.nan_to_num(donchian_state(C, n, max(2, n // 2), True))
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------
# сайзинг


def weights(d: dict, S: np.ndarray, uni: np.ndarray, sig_ann: np.ndarray, K: int, sizing: str,
            long_only: bool, tv_coin: float = TV_COIN, tv_port: float = TV_PORT,
            gcap_iv: float = GCAP_IV, gcap_pv: float = GCAP_PV, pv_win: int = PV_WIN) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    s = np.where(uni, S, 0.0)
    if long_only:
        s = np.maximum(s, 0.0)
    sig = np.maximum(np.nan_to_num(sig_ann, nan=np.inf), SIG_FLOOR)
    s = np.where(np.isfinite(sig_ann), s, 0.0)  # нет оценки волатильности — нет позиции
    if sizing == "eq":
        W = s / K
        scale = np.ones(len(W))
    elif sizing == "iv":
        W = s * (tv_coin / sig) / K
        g = np.abs(W).sum(1)
        scale = np.where(g > gcap_iv, gcap_iv / np.maximum(g, 1e-12), 1.0)
        W = W * scale[:, None]
    elif sizing == "pv":
        W0 = s * (1.0 / sig) / K
        pv = hist_port_vol(np.ascontiguousarray(W0), d["rcc"], pv_win)
        g = np.abs(W0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sc = np.where(pv > 0, tv_port / pv, 0.0)
            sc = np.minimum(sc, np.where(g > 0, gcap_pv / g, 0.0))
        scale = np.nan_to_num(sc)
        W = W0 * scale[:, None]
    else:
        raise ValueError(sizing)
    return W, scale


REBS = {"D0": ("D", 0.0), "D25": ("D", 0.25), "W0": ("W", 0.0), "W25": ("W", 0.25)}


def reb_flags(ts: np.ndarray, kind: str, wd: int = 6) -> np.ndarray:
    if kind == "D":
        return np.ones(len(ts), bool)
    return weekday(ts) == wd  # сигнал по close воскресенья → сделка в понедельник 00:00 UTC


# ----------------------------------------------------------------------------------------------
# статистика


def daily_stats(ts: np.ndarray, res: dict, full: bool = False) -> dict:
    ret = res["ret"]
    out = {}
    for name, win in (("is", IS), ("oos", OOS)):
        a, b = window_idx(ts, win)
        m = metrics(ts[a:b], ret[a:b], None)
        gross_ret = ret[a:b] + (-res["cost"][a:b])
        m["sharpe_gross"] = metrics(ts[a:b], gross_ret)["sharpe"]
        m["avg_gross_lev"] = round(float(res["gross"][a:b].mean()), 3)
        m["turnover_day"] = round(float(res["turn"][a:b].mean()), 4)
        yrs = (b - a) / 365.25
        m["cost_pct_yr"] = round(float(-res["cost"][a:b].sum()) / yrs * 100, 2)
        m["funding_pct_yr"] = round(float(res["funding"][a:b].sum()) / yrs * 100, 2)
        m["price_pct_yr"] = round(float(res["price"][a:b].sum()) / yrs * 100, 2)
        out[name] = m
    if full:
        out["full"] = metrics(ts, ret)
        out["per_year"] = per_year(ts, ret)
        out["half_years"] = half_years(ts, ret)
        out["entries"], out["fills"], out["delists"] = res["entries"], res["fills"], res["delists"]
        g, tv = res["gross"], res["turn"]
        a, b = window_idx(ts, OOS)
        out["avg_hold_days_oos"] = round(float(2 * g[a:b].mean() / max(tv[a:b].mean(), 1e-12)), 1)
        out["max_gross"] = round(float(g.max()), 3)
        out["avg_net_exposure_oos"] = round(float(res["net"][a:b].mean()), 3)
    return out


def half_years(ts: np.ndarray, ret: np.ndarray) -> dict[str, float]:
    dt = ts.astype("datetime64[ms]")
    y = dt.astype("datetime64[Y]").astype(int) + 1970
    mth = dt.astype("datetime64[M]").astype(int) % 12
    key = np.array([f"{a}H{1 if m < 6 else 2}" for a, m in zip(y, mth)])
    out = {}
    for k in dict.fromkeys(key):
        r = ret[key == k]
        out[k] = round((float(np.prod(1 + r)) - 1) * 100, 1)
    return out


def summarize(rows: list[dict], key: str = "oos") -> dict:
    sh = np.array([r[key]["sharpe"] for r in rows])
    return {"count": len(rows), "median_sharpe": round(float(np.median(sh)), 2),
            "share_positive": round(float(np.mean(sh > 0)), 3),
            "p10": round(float(np.percentile(sh, 10)), 2), "p90": round(float(np.percentile(sh, 90)), 2)}  # fmt: skip


def save(name: str, obj) -> None:
    (ODIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))


# ----------------------------------------------------------------------------------------------
# (a) перебор


def baselines(d: dict, sig_ann: np.ndarray) -> dict:
    ts = d["ts"]
    out = {}
    j = list(d["symbols"]).index("BTCUSDT")
    TW = np.zeros_like(d["R"])
    TW[:, j] = np.where(d["alive"][:, j], 1.0, 0.0)
    r = run(d, TW, reb_flags(ts, "D"), buf=0.02)
    out["btc_perp_1x"] = daily_stats(ts, r, True)
    spot = {**r, "ret": r["ret"] - r["funding"], "funding": np.zeros_like(r["ret"])}
    out["btc_spot_like_1x"] = daily_stats(ts, spot, True)
    for K in (20, 30, 50):
        uni = topk_mask(d, K)
        ones = np.ones_like(d["R"])
        W, _ = weights(d, ones, uni, sig_ann, K, "eq", True)
        out[f"ew_top{K}_long_1x_W"] = daily_stats(ts, run(d, W, reb_flags(ts, "W")), True)
        out[f"ew_top{K}_long_1x_D25"] = daily_stats(ts, run(d, W, reb_flags(ts, "D"), buf=0.25), True)
        W, _ = weights(d, ones, uni, sig_ann, K, "pv", True)
        out[f"pv_top{K}_long_W"] = daily_stats(ts, run(d, W, reb_flags(ts, "W")), True)
    return out


def config_iter():
    sigs = ["mom10", "mom20", "mom30", "mom60", "mom90", "mom120", "tz20", "tz60", "tz120",
            "ema5_20", "ema10_40", "ema20_80", "ema30_120", "multi_mom", "multi_ema", "donch20", "donch55"]  # fmt: skip
    for sg, side, K, A, sizing, rb in itertools.product(sigs, ("LO", "LS"), (20, 30, 50), (60, 365), ("eq", "iv", "pv"), REBS):
        yield {"signal": sg, "side": side, "K": K, "A": A, "sizing": sizing, "reb": rb}


CFG_KEYS = ("signal", "side", "K", "A", "sizing", "reb")


def cfg_name(c: dict) -> str:
    return f"{c['signal']}|{c['side']}|K{c['K']}|A{c['A']}|{c['sizing']}|{c['reb']}"


def make_unis(d: dict) -> dict:
    return {(K, A): topk_mask(d, K, A) for K in (20, 30, 50) for A in (60, 365)}


def run_cfg(d, S, sig_ann, unis, c, **kw):
    W, _ = weights(d, S[c["signal"]], unis[(c["K"], c["A"])], sig_ann, c["K"], c["sizing"], c["side"] == "LO", **kw)
    kind, buf = REBS[c["reb"]]
    return run(d, W, reb_flags(d["ts"], kind), buf=buf), W


def cmd_grid() -> None:
    t0 = time.time()
    d = daily_data()
    ts = d["ts"]
    S, sig_ann = build_signals(d)
    unis = make_unis(d)
    print(f"data+signals {time.time() - t0:.0f}s", flush=True)
    base = baselines(d, sig_ann)
    for k, v in base.items():
        print(f"{k:28s} IS {v['is']['sharpe']:5.2f} {v['is']['cagr_pct']:7.1f}%  OOS {v['oos']['sharpe']:5.2f} "
              f"{v['oos']['cagr_pct']:7.1f}% dd {v['oos']['max_dd_pct']:.0f}%", flush=True)  # fmt: skip
    rows, rets = [], []
    for i, c in enumerate(config_iter()):
        res, _ = run_cfg(d, S, sig_ann, unis, c)
        st = daily_stats(ts, res)
        rows.append({**c, "name": cfg_name(c), **st})
        rets.append(res["ret"].astype(np.float32))
        if i % 200 == 0:
            print(i, f"{time.time() - t0:.0f}s", cfg_name(c), st["is"]["sharpe"], st["oos"]["sharpe"], flush=True)
    save("part_a.json", {"baselines": base, "configs": rows})
    np.savez_compressed(TCACHE / "grid_rets.npz", ts=ts, ret=np.array(rets), names=np.array([r["name"] for r in rows]))
    print(f"done {len(rows)} configs {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------
# выбор лучшего по IS


def pick_best(rows: list[dict], flt=None) -> dict:
    cand = [r for r in rows if (flt is None or flt(r))]
    return max(cand, key=lambda r: r["is"]["sharpe"])


def load_part(name: str):
    return json.loads((ODIR / name).read_text())


# ----------------------------------------------------------------------------------------------
# (c) фильтр режима BTC


def btc_regimes(d: dict) -> dict[str, np.ndarray]:
    """risk-on[t] по close BTC до t включительно. True — держим портфель."""
    j = list(d["symbols"]).index("BTCUSDT")
    c = d["C"][:, j : j + 1]
    out = {}
    for n in (50, 100, 150, 200):
        sma = np.full_like(c, np.nan)
        cs = np.cumsum(np.nan_to_num(c), axis=0)
        sma[n - 1 :] = (cs[n - 1 :] - np.r_[np.zeros((1, 1)), cs[:-n]]) / n
        out[f"btc_sma{n}"] = np.nan_to_num(c > sma).astype(bool)[:, 0]
    for f, s in ((10, 40), (20, 80), (50, 200)):
        out[f"btc_ema{f}_{s}"] = np.nan_to_num(ema_cols(c, f) > ema_cols(c, s)).astype(bool)[:, 0]
    for L in (30, 60, 90):
        out[f"btc_mom{L}"] = (np.nan_to_num(mom(c, L)) > 0)[:, 0]
    v = ewm_vol(d["rcc"][:, j : j + 1], 10, 10)[:, 0]
    med = np.full_like(v, np.nan)
    for t in range(180, len(v)):
        med[t] = np.nanmedian(v[t - 365 + 1 if t >= 365 else 0 : t + 1])
    lowvol = np.nan_to_num(v <= med).astype(bool)
    out["btc_lowvol"] = lowvol
    out["btc_sma100_or_lowvol"] = out["btc_sma100"] | lowvol
    out["btc_sma100_and_lowvol"] = out["btc_sma100"] & lowvol
    out["btc_ema20_80_and_mom30"] = out["btc_ema20_80"] & out["btc_mom30"]
    return out


def apply_regime(W: np.ndarray, on: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flat":
        return W * on[:, None]
    if mode == "cut_longs":  # в risk-off лонги убираются, шорты остаются
        return np.where(on[:, None], W, np.minimum(W, 0.0))
    raise ValueError(mode)


def cmd_regime() -> None:
    d = daily_data()
    ts = d["ts"]
    S, sig_ann = build_signals(d)
    unis = make_unis(d)
    rows = load_part("part_a.json")["configs"]
    reg = btc_regimes(d)
    bases = {
        "best_all": pick_best(rows),
        "best_LO": pick_best(rows, lambda r: r["side"] == "LO"),
        "best_LS": pick_best(rows, lambda r: r["side"] == "LS"),
    }
    out = {"bases": {k: v["name"] for k, v in bases.items()}, "rows": [], "ew": []}
    for bk, b in bases.items():
        c = {k: b[k] for k in CFG_KEYS}
        _, W = run_cfg(d, S, sig_ann, unis, c)
        kind, buf = REBS[c["reb"]]
        rf = reb_flags(ts, kind)
        for rn, on in reg.items():
            for mode in ("flat",) + (("cut_longs",) if c["side"] == "LS" else ()):
                W2 = apply_regime(W, on, mode)
                # смена режима исполняется на следующем открытии даже между недельными ребалансировками
                rf2 = rf | np.r_[False, on[1:] != on[:-1]]
                st = daily_stats(ts, run(d, W2, rf2, buf=buf))
                out["rows"].append({"base": bk, "base_name": b["name"], "regime": rn, "mode": mode, **st})
                print(bk, rn, mode, st["is"]["sharpe"], st["oos"]["sharpe"], flush=True)
    # фильтр режима для простого равновзвешенного лонга (то, что можно сделать ботом)
    for K in (20, 30):
        ones = np.ones_like(d["R"])
        W, _ = weights(d, ones, unis[(K, 60)], sig_ann, K, "eq", True)
        for rn, on in reg.items():
            W2 = apply_regime(W, on, "flat")
            rf2 = reb_flags(ts, "W") | np.r_[False, on[1:] != on[:-1]]
            st = daily_stats(ts, run(d, W2, rf2))
            out["ew"].append({"K": K, "regime": rn, **st})
            print("ew", K, rn, st["is"]["sharpe"], st["oos"]["sharpe"], flush=True)
        # BTC отдельно, 1×
    j = list(d["symbols"]).index("BTCUSDT")
    for rn, on in reg.items():
        TW = np.zeros_like(d["R"])
        TW[:, j] = on * d["alive"][:, j]
        st = daily_stats(ts, run(d, TW, reb_flags(ts, "D"), buf=0.02))
        out["ew"].append({"K": "BTC", "regime": rn, **st})
    save("part_c.json", out)


# ----------------------------------------------------------------------------------------------
# выбранный кандидат и проверки


def chosen() -> tuple[dict, dict | None]:
    """Лучший по IS Sharpe среди всех конфигураций (a) и всех вариантов фильтра режима (c)."""
    rows = load_part("part_a.json")["configs"]
    best_a = pick_best(rows)
    best = {"src": "a", **{k: best_a[k] for k in CFG_KEYS}, "regime": None, "mode": None, "is_sharpe": best_a["is"]["sharpe"]}
    pc = ODIR / "part_c.json"
    if pc.exists():
        rc = [r for r in load_part("part_c.json")["rows"]]
        if rc:
            bc = max(rc, key=lambda r: r["is"]["sharpe"])
            if bc["is"]["sharpe"] > best["is_sharpe"]:
                base = next(r for r in rows if r["name"] == bc["base_name"])
                best = {"src": "c", **{k: base[k] for k in CFG_KEYS}, "regime": bc["regime"], "mode": bc["mode"],
                        "is_sharpe": bc["is"]["sharpe"]}  # fmt: skip
    return best, best_a


def build_W(d: dict, cfg: dict, scale: float = 1.0, K: int | None = None, A: int | None = None, hl: float = 20.0,
            tv_port: float = TV_PORT, tv_coin: float = TV_COIN, pv_win: int = PV_WIN, gcap_pv: float = GCAP_PV,
            gcap_iv: float = GCAP_IV, sig_override: np.ndarray | None = None, regimes: dict | None = None) -> np.ndarray:  # fmt: skip
    K = K or cfg["K"]
    A = A or cfg["A"]
    sig_ann = ewm_vol(d["rcc"], hl, 20)
    if sig_override is not None:
        S = sig_override
    else:
        S = sig_variant(d, cfg["signal"], sig_ann / np.sqrt(365.0), scale)
    uni = topk_mask(d, K, A)
    W, _ = weights(d, S, uni, sig_ann, K, cfg["sizing"], cfg["side"] == "LO", tv_coin=tv_coin, tv_port=tv_port,
                   gcap_iv=gcap_iv, gcap_pv=gcap_pv, pv_win=pv_win)  # fmt: skip
    if cfg.get("regime"):
        on = (regimes or btc_regimes(d))[cfg["regime"]]
        W = apply_regime(W, on, cfg["mode"])
    return W


def reb_for(d: dict, cfg: dict, wd: int = 6) -> tuple[np.ndarray, float]:
    kind, buf = REBS[cfg["reb"]]
    rf = reb_flags(d["ts"], kind, wd)
    if cfg.get("regime"):
        on = btc_regimes(d)[cfg["regime"]]
        rf = rf | np.r_[False, on[1:] != on[:-1]]
    return rf, buf


def shuffled_signal(S: np.ndarray, alive: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Контроль: сигнал каждой монеты циклически сдвинут во времени внутри её периода торговли —
    сохраняются распределение, автокорреляция и оборот, но теряется связь с будущей доходностью."""
    out = np.zeros_like(S)
    for j in range(S.shape[1]):
        idx = np.where(alive[:, j])[0]
        n = len(idx)
        if n < 200:
            continue
        k = int(rng.integers(60, n - 60))
        out[idx, j] = np.roll(S[idx, j], k)
    return out


def capacity(d: dict, res_W: np.ndarray, win=OOS) -> dict:
    """Ёмкость: номинал сделки ≤ 1 % среднечасового объёма монеты (ADV/24 за 30 дней).
    Размер счёта E, при котором позиция |w|·E укладывается в лимит, — по позициям-дням OOS."""
    a, b = window_idx(d["ts"], win)
    W = res_W[a:b]
    adv_h = d["adv"][a:b] / 24.0
    m = np.abs(W) > 1e-9
    lim = 0.01 * adv_h[m]
    acct = lim / np.abs(W[m])
    return {"per_trade_cap_usd_p5": round(float(np.percentile(lim, 5))), "per_trade_cap_usd_median": round(float(np.median(lim))),
            "account_cap_usd_p1": round(float(np.percentile(acct, 1))), "account_cap_usd_p5": round(float(np.percentile(acct, 5))),
            "account_cap_usd_median": round(float(np.median(acct)))}  # fmt: skip


def concentration(d: dict, W: np.ndarray, res: dict, win=OOS) -> dict:
    """Вклад монет в P&L (приближённо: целевые веса t держатся в t+1) за окно."""
    a, b = window_idx(d["ts"], win)
    a = max(a, 1)
    F = d["F0"] + d["F1"]
    pr = W[a - 1 : b - 1] * d["R"][a:b]
    fu = -W[a - 1 : b - 1] * F[a:b]
    tot = (pr + fu).sum(0)
    o = np.argsort(-tot)
    syms = d["symbols"]
    total = float(tot.sum())
    return {"total_sum": round(total, 4), "top10": [(str(syms[j]), round(float(tot[j]), 4), round(float(fu.sum(0)[j]), 4)) for j in o[:10]],
            "bottom5": [(str(syms[j]), round(float(tot[j]), 4)) for j in o[::-1][:5]],
            "share_top5": round(float(tot[o[:5]].sum()) / total, 3) if total else None,
            "share_top10": round(float(tot[o[:10]].sum()) / total, 3) if total else None}  # fmt: skip


def spot_presample(cfg: dict) -> dict:
    """Досэмпловая проверка 2018-01 … 2019-12 на дневных СПОТ-свечах 9 монет (без фандинга, шорт —
    синтетический). Вселенная = все 9 монет (K не используется). Только как доп. свидетельство."""
    from scripts.cycle_lab import data as cl

    syms = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "LINK", "SOL", "AVAX"]
    raw = {}
    for s in syms:
        z = np.load(cl.CACHE / f"{s}USDT.npz")
        raw[s] = (z["ts"], z["ohlc"], z["qvol"])
    t0 = min(int(v[0][0]) for v in raw.values()) // DAY * DAY
    t1 = ms("2020-01-01")
    ts = np.arange(t0, t1, DAY, dtype=np.int64)
    T, N = len(ts), len(syms)
    O = np.full((T, N), np.nan)
    C = np.full((T, N), np.nan)
    Q = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        tt, ohlc, qv = raw[s]
        k = (tt - t0) // DAY
        ok = (k >= 0) & (k < T)
        O[k[ok], j], C[k[ok], j], Q[k[ok], j] = ohlc[ok, 0], ohlc[ok, 3], qv[ok]
    from scripts.edge_lab.common import adv_usd, cost_per_side

    alive = ~np.isnan(C)
    nxt = np.vstack([O[1:], np.full((1, N), np.nan)])
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.nan_to_num(np.where(alive & ~np.isnan(nxt), nxt / O - 1, np.where(alive, C / O - 1, 0.0)))
        rcc = np.vstack([np.full((1, N), np.nan), C[1:] / C[:-1] - 1])
    adv = np.nan_to_num(adv_usd({"ts": ts, "qvol": Q}))
    dd = {"ts": ts, "symbols": np.array([s + "USDT" for s in syms]), "O": O, "C": C, "R": R, "rcc": rcc, "alive": alive,
          "tradable_next": np.vstack([~np.isnan(O[1:]), np.zeros((1, N), bool)]), "adv": adv,
          "cost": cost_per_side(np.maximum(adv, 1.0)), "age": np.cumsum(alive, axis=0),
          "F0": np.zeros((T, N)), "F1": np.zeros((T, N))}  # fmt: skip
    out = {}
    c2 = {**cfg, "regime": None}
    W = build_W(dd, c2, K=N, A=60)  # вселенная = все 9 монет, доля слота 1/9
    r = run(dd, W, reb_flags(ts, REBS[cfg["reb"]][0]), buf=REBS[cfg["reb"]][1])
    for wn, (a0, a1) in (("2018", ("2018-01-01", "2019-01-01")), ("2019", ("2019-01-01", "2020-01-01")),
                         ("2018-2019", ("2018-01-01", "2020-01-01"))):  # fmt: skip
        a, b = window_idx(ts, (a0, a1))
        out[wn] = metrics(ts[a:b], r["ret"][a:b])
    ones = np.ones_like(R)
    sig_ann = ewm_vol(rcc, 20, 20)
    Wb, _ = weights(dd, ones, topk_mask(dd, N, 60), sig_ann, N, "eq", True)
    rb = run(dd, Wb, reb_flags(ts, "W"))
    a, b = window_idx(ts, ("2018-01-01", "2020-01-01"))
    out["ew_long_2018-2019"] = metrics(ts[a:b], rb["ret"][a:b])
    out["note"] = "9 монет спот, без фандинга, шорт синтетический; IS/OOS основного теста не затрагивается"
    return out


def cmd_robust() -> None:
    t0 = time.time()
    d = daily_data()
    ts = d["ts"]
    best, best_a = chosen()
    cfgs = {"chosen": best}
    if best["src"] == "c":
        cfgs["chosen_a"] = {"src": "a", **{k: best_a[k] for k in CFG_KEYS}, "regime": None, "mode": None}
    out = {}
    for tag, cfg in cfgs.items():
        print(tag, cfg, flush=True)
        rf, buf = reb_for(d, cfg)
        W = build_W(d, cfg)
        res = run(d, W, rf, buf=buf)
        o = {"cfg": cfg, "base": daily_stats(ts, res, True)}
        var = {}
        L = lambda **kw: daily_stats(ts, run(d, build_W(d, cfg, **{k: v for k, v in kw.items() if k != "_reb"}), rf, buf=buf))  # noqa: E731
        for sc in (0.75, 1.25):
            var[f"window_x{sc}"] = L(scale=sc)
        for k in sorted({max(5, round(cfg["K"] * 0.75)), round(cfg["K"] * 1.25)}):
            var[f"K_{k}"] = L(K=k)
        for A in sorted({max(30, round(cfg["A"] * 0.75)), round(cfg["A"] * 1.25)}):
            var[f"minage_{A}"] = L(A=A)
        for hl in (15.0, 25.0):
            var[f"vol_hl_{hl:.0f}"] = L(hl=hl)
        if cfg["sizing"] == "pv":
            for tv in (TV_PORT * 0.75, TV_PORT * 1.25):
                var[f"tv_port_{tv:.3f}"] = L(tv_port=tv)
            for pw in (45, 75):
                var[f"pv_win_{pw}"] = L(pv_win=pw)
        if cfg["sizing"] == "iv":
            for tv in (TV_COIN * 0.75, TV_COIN * 1.25):
                var[f"tv_coin_{tv:.2f}"] = L(tv_coin=tv)
        kind, _ = REBS[cfg["reb"]]
        for b2 in (0.0, 0.25, 0.5):
            if b2 != buf:
                var[f"buffer_{b2}"] = daily_stats(ts, run(d, W, rf, buf=b2))
        if kind == "W":
            for wd in range(7):
                rf2, _ = reb_for(d, cfg, wd)
                var[f"weekday_{wd}"] = daily_stats(ts, run(d, W, rf2, buf=buf))
        else:
            for wd in (0, 3, 6):
                c2 = {**cfg, "reb": "W0" if buf == 0 else "W25"}
                rf2, _ = reb_for(d, c2, wd)
                var[f"weekly_wd{wd}"] = daily_stats(ts, run(d, W, rf2, buf=buf))
        d2 = {**d, "cost": d["cost"] * 2}
        var["costs_x2"] = daily_stats(ts, run(d2, W, rf, buf=buf))
        var["delist_minus50"] = daily_stats(ts, run(d, W, rf, buf=buf, delist_pen=0.5), True)
        d3 = {**d, "F0": d["F0"] * 0, "F1": d["F1"] * 0}
        var["no_funding"] = daily_stats(ts, run(d3, W, rf, buf=buf))
        # фандинг с ограничением ±0.3 %/сутки на монету (защита от экстремальных сквизов в данных)
        d4 = {**d, "F0": np.clip(d["F0"], -0.003, 0.003), "F1": np.clip(d["F1"], -0.003, 0.003)}
        var["funding_clip_0.3pct_day"] = daily_stats(ts, run(d4, W, rf, buf=buf))
        for cap0 in (1000.0, 5000.0, 20000.0):
            var[f"account_{int(cap0)}_min10usd"] = daily_stats(ts, run(d, W, rf, buf=buf, cap0=cap0, min_usd=10.0))
        o["variants"] = var
        o["robust_oos_sharpe_min"] = min(v["oos"]["sharpe"] for k, v in var.items() if not k.startswith(("costs", "no_funding", "funding_clip")))
        # плечо и целевая волатильность
        lev = {}
        if cfg["sizing"] == "pv":
            for tv in (0.10, 0.20, 0.30, 0.40, 0.60):
                for gc in (2.0, 3.0, 5.0):
                    r2 = run(d, build_W(d, cfg, tv_port=tv, gcap_pv=gc), rf, buf=buf)
                    st = daily_stats(ts, r2, True)
                    lev[f"tv{tv:.2f}_cap{gc:.0f}"] = {"is": st["is"], "oos": st["oos"], "max_gross": st["max_gross"]}
        else:
            for tv in (0.2, 0.4, 0.6, 0.9):
                for gc in (1.0, 1.5, 3.0):
                    r2 = run(d, build_W(d, cfg, tv_coin=tv, gcap_iv=gc), rf, buf=buf)
                    st = daily_stats(ts, r2, True)
                    lev[f"tvc{tv:.2f}_cap{gc:.1f}"] = {"is": st["is"], "oos": st["oos"], "max_gross": st["max_gross"]}
        o["leverage"] = lev
        # случайный контроль: циклический сдвиг сигнала
        rng = np.random.default_rng(12345)
        sig_ann = ewm_vol(d["rcc"], 20, 20)
        S0 = sig_variant(d, cfg["signal"], sig_ann / np.sqrt(365.0), 1.0)
        rnd = []
        for k in range(60):
            Sr = shuffled_signal(S0, d["alive"], rng)
            st = daily_stats(ts, run(d, build_W(d, cfg, sig_override=Sr), rf, buf=buf))
            rnd.append((st["is"]["sharpe"], st["oos"]["sharpe"], st["oos"]["turnover_day"]))
        rnd = np.array(rnd)
        o["random_shift_control"] = {
            "n": len(rnd), "is_sharpe_median": float(np.median(rnd[:, 0])), "is_sharpe_p95": float(np.percentile(rnd[:, 0], 95)),
            "oos_sharpe_median": float(np.median(rnd[:, 1])), "oos_sharpe_p95": float(np.percentile(rnd[:, 1], 95)),
            "share_random_oos_ge_actual": float(np.mean(rnd[:, 1] >= o["base"]["oos"]["sharpe"])),
            "share_random_is_ge_actual": float(np.mean(rnd[:, 0] >= o["base"]["is"]["sharpe"])),
            "turnover_median": float(np.median(rnd[:, 2])),
        }  # fmt: skip
        o["capacity"] = capacity(d, W)
        o["concentration_oos"] = concentration(d, W, res)
        o["concentration_is"] = concentration(d, W, res, IS)
        try:
            o["spot_presample"] = spot_presample(cfg)
        except Exception as exc:  # noqa: BLE001
            o["spot_presample"] = {"error": repr(exc)}
        out[tag] = o
        print(tag, "done", f"{time.time() - t0:.0f}s", flush=True)
    save("part_robust.json", out)


def _sharpe(r: np.ndarray) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def walk_forward() -> dict:
    """Скользящий выбор: в начале каждого года Y берём конфигурацию (a) с лучшим Sharpe на всей истории
    до Y (расширяющееся окно) и торгуем её в году Y. Это оценка самой ПРОЦЕДУРЫ выбора вне выборки."""
    z = np.load(TCACHE / "grid_rets.npz")
    ts, R, names = z["ts"], z["ret"].astype(np.float64), z["names"]
    yrs = ts.astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    out_r = np.zeros(len(ts))
    picks = {}
    for Y in range(2021, 2027):
        tr = yrs < Y
        te = yrs == Y
        sh = R[:, tr].mean(1) / np.maximum(R[:, tr].std(1), 1e-12)
        k = int(np.argmax(sh))
        out_r[te] = R[k, te]
        picks[str(Y)] = {"pick": str(names[k]), "train_sharpe": round(float(sh[k] * np.sqrt(365)), 2),
                         "year_return_pct": round((float(np.prod(1 + R[k, te])) - 1) * 100, 1),
                         "year_sharpe": round(_sharpe(R[k, te]), 2)}  # fmt: skip
    a, b = window_idx(ts, OOS)
    a21 = int(np.searchsorted(ts, ms("2021-01-01")))
    # то же, но выбор по медиане: равновесная смесь top-10 IS-конфигураций (меньше шума выбора)
    return {"picks": picks, "wf_2021_2026": metrics(ts[a21:], out_r[a21:]), "wf_oos_2024_2026": metrics(ts[a:b], out_r[a:b]),
            "wf_per_year": per_year(ts[a21:], out_r[a21:])}  # fmt: skip


def cmd_final() -> None:
    d = daily_data()
    ts = d["ts"]
    pa, pb, pc, prb = (load_part(n) for n in ("part_a.json", "part_b.json", "part_c.json", "part_robust.json"))
    best, best_a = chosen()
    rf, buf = reb_for(d, best)
    W = build_W(d, best)
    res = run(d, W, rf, buf=buf)
    np.savez(ODIR / "best_daily.npz", ts=ts, ret=res["ret"], gross=res["gross"], net=res["net"],
             note=np.array("ret = дневная чистая доходность капитала; веса — доля капитала (номинал/капитал), "
                           "валовая экспозиция Σ|w| в массиве gross (средняя ≈0.3–0.4, потолок 2.0)"))  # fmt: skip
    rows = pa["configs"]
    allrows = rows + pc["rows"]
    wf = walk_forward()
    results = {
        "family": "trend",
        "conventions": {"is": IS, "oos": OOS, "sharpe": "дневные доходности, ×sqrt(365)", "weights": "доля капитала",
                        "execution": "сигнал по close[t] (сутки UTC), сделка по open[t+1]", "costs": "cost_per_side(ADV30) на каждую сделку",
                        "funding": "выплата 00:00 — старая позиция, остальные — новая; + ставка = лонг платит",
                        "delisting": "закрытие по последнему close; вариант −50% в robust"},  # fmt: skip
        "best": {"cfg": best, "stats": daily_stats(ts, res, True)},
        "best_a_only": best_a["name"],
        "part_a": {"summary_all": summarize(rows), "summary_is": summarize(rows, "is"), "baselines": pa["baselines"],
                   "configs": rows},  # fmt: skip
        "part_b": pb,
        "part_c": pc,
        "all_configs_summary": {"a_plus_c": summarize(allrows), "b": {"count": pb["summary"]["count"],
                                "median_sharpe": round(pb["summary"]["median_oos_sharpe"], 2),
                                "share_positive": round(pb["summary"]["share_oos_pos"], 3)}},  # fmt: skip
        "robust": prb,
        "walk_forward": wf,
    }
    save("results.json", results)
    print(json.dumps({"best": best, "stats": results["best"]["stats"], "wf": wf}, indent=1, default=float, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    {"grid": cmd_grid, "regime": cmd_regime, "robust": cmd_robust, "final": cmd_final}[cmd]()
