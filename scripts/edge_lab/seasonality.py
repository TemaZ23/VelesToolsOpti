"""
seasonality — календарные и внутридневные эффекты (Binance USDT-M перпы, 2020-01 … 2026-09).

    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality tests     # A: статтесты ячеек (IS), поправки
    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality cme       # B: гэп CME (выходные) BTC
    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality strats    # C: стратегии из ячеек, IS→OOS
    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality robust    # D: устойчивость лучшего кандидата
    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality report    # E: results.json + best_daily.npz

Данные — cache/edge_lab/seasonality/base_1h.npz (scripts/edge_lab/seasonality_data.py).

Методика тестов (раздел A):
- «ячейка» — булева маска на сетке (день × час UTC) для часовых эффектов или (неделя × день недели)
  для дневных; статистика — t разности средних «в ячейке» и «вне ячейки» (сезонная компонента,
  не зависящая от общего тренда), SE по Уэлчу;
- поправка на множественность: (1) перестановочный тест внутри строки (перемешиваем 24 часа внутри
  каждых суток / 7 дней внутри каждой недели — сохраняются суточная/недельная волатильность и тренд)
  со статистикой max|t| по семье → FWER-p; (2) Холм–Бонферрони по ВСЕМ ячейкам всех семей и активов;
- стабильность: знак эффекта в 2020, 2021, 2022, 2023 по отдельности;
- OOS (2024-01 … 2026-09) в выборе ячеек не участвует, приводится справочно.
"""

from __future__ import annotations

import json
import math
import sys
import time
import zlib

import numpy as np
import pandas as pd

from scripts.edge_lab.common import DAY, HOUR, IS, MAKER_FEE, OOS, OUT, metrics, ms, per_year
from scripts.edge_lab.seasonality_data import load_base

ODIR = OUT / "seasonality"
ASSETS = ["BTC", "ETH", "ALT", "ALT30"]
SINGLES = ["BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "DOT", "BCH", "TRX"]
IS_YEARS = [2020, 2021, 2022, 2023]
OOS_YEARS = [2024, 2025, 2026]
B_PERM = 2000

# NYSE: полные выходные дни (праздники) 2020–2026
NYSE_HOLIDAYS = """
2020-01-01 2020-01-20 2020-02-17 2020-04-10 2020-05-25 2020-07-03 2020-09-07 2020-11-26 2020-12-25
2021-01-01 2021-01-18 2021-02-15 2021-04-02 2021-05-31 2021-07-05 2021-09-06 2021-11-25 2021-12-24
2022-01-17 2022-02-21 2022-04-15 2022-05-30 2022-06-20 2022-07-04 2022-09-05 2022-11-24 2022-12-26
2023-01-02 2023-01-16 2023-02-20 2023-04-07 2023-05-29 2023-06-19 2023-07-04 2023-09-04 2023-11-23 2023-12-25
2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19 2024-07-04 2024-09-02 2024-11-28 2024-12-25
2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26 2025-06-19 2025-07-04 2025-09-01 2025-11-27
2025-12-25 2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03 2026-09-07
""".split()


# ----------------------------------------------------------------------------- общие утилиты


def norm_sf(z: np.ndarray | float) -> np.ndarray | float:
    """P(Z > z) для стандартного нормального."""
    z = np.asarray(z, dtype=float)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def p_two(t: np.ndarray | float) -> np.ndarray | float:
    return 2.0 * norm_sf(np.abs(t))


def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Холм–Бонферрони: булев массив «отвергаем H0» и скорректированные p."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def year_of(ts: np.ndarray) -> np.ndarray:
    return ts.astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970


def save_json(obj, name: str) -> None:
    ODIR.mkdir(parents=True, exist_ok=True)
    (ODIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jdefault))


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def load_json(name: str):
    return json.loads((ODIR / name).read_text())


# ----------------------------------------------------------------------------- данные и календарь


def calendar(ts: np.ndarray) -> dict[str, np.ndarray]:
    t = pd.to_datetime(ts, unit="ms", utc=True)
    et = t.tz_convert("America/New_York")
    ct = t.tz_convert("America/Chicago")
    lon = t.tz_convert("Europe/London")
    et_date = np.array(et.strftime("%Y-%m-%d"))
    hol = np.isin(et_date, np.array(NYSE_HOLIDAYS))
    dom = np.asarray(t.day)
    dim = np.asarray(t.days_in_month)
    return {
        "hour": np.asarray(t.hour),
        "wday": np.asarray(t.dayofweek),  # 0 = понедельник
        "dom": dom,
        "dim": dim,
        "dfe": dim - dom,  # дней до конца месяца (0 = последний день)
        "month": np.asarray(t.month),
        "year": np.asarray(t.year),
        "et_hour": np.asarray(et.hour),
        "et_wday": np.asarray(et.dayofweek),
        "nyse": (np.asarray(et.dayofweek) < 5) & ~hol,
        "ct_hour": np.asarray(ct.hour),
        "ct_wday": np.asarray(ct.dayofweek),
        "lon_hour": np.asarray(lon.hour),
    }


def assets(b: dict) -> dict[str, dict[str, np.ndarray]]:
    """Часовые ряды по активам: r[t] (доходность часа t), cost[t] (taker, на сторону, по ADV до t-1),
    fund[t] (ставка, рассчитанная в момент ts[t]), valid[t]; для отдельных перпов — OHLC (для мейкера)."""
    from scripts.edge_lab.common import cost_per_side

    T = len(b["ts"])
    out = {}
    for s in ["BTC", "ETH"] + SINGLES:
        c = b[f"{s}_close"]
        r = np.full(T, np.nan)
        r[1:] = c[1:] / c[:-1] - 1
        adv_prev = np.r_[np.nan, b[f"{s}_adv"][:-1]]
        out[s] = {
            "r": r,
            "cost": cost_per_side(np.nan_to_num(adv_prev)),
            "fund": b[f"{s}_funding"],
            "valid": np.isfinite(r) & np.isfinite(adv_prev),  # для статтестов (полные часы)
            # для торговли на t — только прошлое: close[t-1] известен и есть ADV до t-1; если close[t]
            # не появился (остановка торгов) — r = 0 (выход по последнему close)
            "tradable": np.r_[False, np.isfinite(c[:-1])] & np.isfinite(adv_prev),
            "o": b[f"{s}_open"], "h": b[f"{s}_high"], "l": b[f"{s}_low"], "c": c,
            "adv_prev": adv_prev, "qvol": b[f"{s}_qvol"],
        }  # fmt: skip
    for s in ["ALT", "ALT30"]:
        nmin = 10 if s == "ALT" else 30
        out[s] = {
            "r": b[f"{s}_ret"],
            "r_pess": b[f"{s}_ret_pess"],
            "cost": np.nan_to_num(b[f"{s}_cost"], nan=0.004),
            "fund": b[f"{s}_fund"],
            "valid": b[f"{s}_n"] >= nmin,
            "adv_prev": b[f"{s}_adv_med"],
            "n": b[f"{s}_n"],
        }
    return out


# ----------------------------------------------------------------------------- тест ячеек


def _cell_t(Xf: np.ndarray, M: np.ndarray, n_in: np.ndarray, n_out: np.ndarray):
    """Xf (b, P) значения (NaN недопустимы — заменены 0 и исключены через маски), M (P, C) маски.
    Возвращает t (b, C), mean_in, mean_out."""
    s_in = Xf @ M
    q_in = (Xf * Xf) @ M
    s_all = Xf.sum(1, keepdims=True)
    q_all = (Xf * Xf).sum(1, keepdims=True)
    mi = s_in / n_in
    mo = (s_all - s_in) / n_out
    vi = (q_in / n_in - mi**2) * n_in / np.maximum(n_in - 1, 1)
    vo = ((q_all - q_in) / n_out - mo**2) * n_out / np.maximum(n_out - 1, 1)
    se = np.sqrt(np.maximum(vi / n_in + vo / n_out, 1e-30))
    return (mi - mo) / se, mi, mo


def family_test(X: np.ndarray, masks: dict[str, np.ndarray], B: int = B_PERM, seed: int = 0):
    """X — сетка (D, K) доходностей (строки — сутки/недели, без NaN), masks — {имя: (D, K) bool}.
    Перестановочный тест внутри строк, статистика max|t| по семье. Возвращает dict по ячейкам."""
    D, K = X.shape
    names = list(masks)
    M = np.stack([masks[k].ravel() for k in names], 1).astype(np.float64)
    n_in = M.sum(0)
    n_out = D * K - n_in
    ok = (n_in >= 10) & (n_out >= 10)
    Xf = X.ravel()[None, :]
    t_obs, mi, mo = _cell_t(Xf, M, n_in, n_out)
    t_obs, mi, mo = t_obs[0], mi[0], mo[0]
    # t против нуля (направленная торговля)
    s_in = (X.ravel()[:, None] * M).sum(0)
    q_in = ((X.ravel() ** 2)[:, None] * M).sum(0)
    m_in = s_in / n_in
    v_in = (q_in / n_in - m_in**2) * n_in / np.maximum(n_in - 1, 1)
    t_abs = m_in / np.sqrt(np.maximum(v_in / n_in, 1e-30))
    rng = np.random.default_rng(seed)
    maxes = np.empty(B)
    bs = 100
    for i0 in range(0, B, bs):
        b = min(bs, B - i0)
        idx = np.argsort(rng.random((b, D, K)), axis=2)
        Xp = np.take_along_axis(np.broadcast_to(X, (b, D, K)), idx, axis=2).reshape(b, D * K)
        tp, _, _ = _cell_t(Xp, M, n_in, n_out)
        tp = np.where(ok[None, :], tp, 0.0)
        maxes[i0 : i0 + b] = np.abs(tp).max(1)
    res = {}
    for c, k in enumerate(names):
        res[k] = {
            "n": int(n_in[c]),
            "mean_bp": float(mi[c] * 1e4),
            "mean_out_bp": float(mo[c] * 1e4),
            "diff_bp": float((mi[c] - mo[c]) * 1e4),
            "t": float(t_obs[c]),
            "p": float(p_two(t_obs[c])),
            "t_abs": float(t_abs[c]),
            "p_fwer_perm": float((1 + np.sum(maxes >= abs(t_obs[c]))) / (B + 1)) if ok[c] else 1.0,
        }
    return res


def simple_stats(X: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """Те же t-статистики без перестановок (для отдельных лет / OOS)."""
    D, K = X.shape
    out = {}
    for k, m in masks.items():
        a, o = X[m], X[~m]
        if len(a) < 5 or len(o) < 5:
            out[k] = {"n": int(len(a)), "diff_bp": float("nan"), "t": float("nan"), "mean_bp": float("nan")}
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + o.var(ddof=1) / len(o))
        out[k] = {"n": int(len(a)), "mean_bp": float(a.mean() * 1e4), "diff_bp": float((a.mean() - o.mean()) * 1e4),
                  "t": float((a.mean() - o.mean()) / se) if se > 0 else 0.0,
                  "t_abs": float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))) if a.std() > 0 else 0.0}  # fmt: skip
    return out


def hourly_grid(a: dict, ts: np.ndarray):
    """(D, 24) сетка часовых доходностей и булев вектор «сутки полностью валидны»."""
    T = len(ts)
    D = T // 24
    r = a["r"][: D * 24].reshape(D, 24)
    v = a["valid"][: D * 24].reshape(D, 24) & np.isfinite(r)
    return np.nan_to_num(r), v.all(1), ts[: D * 24 : 24]


def daily_returns(a: dict, ts: np.ndarray, key: str = "r"):
    """Доходность суток UTC (00:00 → 24:00) компаундингом часов; valid — все 24 часа валидны."""
    T = len(ts)
    D = T // 24
    r = np.nan_to_num(a[key][: D * 24].reshape(D, 24))
    v = (a["valid"][: D * 24].reshape(D, 24) & np.isfinite(a["r"][: D * 24].reshape(D, 24))).all(1)
    return np.prod(1 + r, axis=1) - 1, v, ts[: D * 24 : 24]


def weekly_grid(dr: np.ndarray, dv: np.ndarray, dts: np.ndarray):
    """(W, 7) сетка дневных доходностей пн..вс, только полные валидные недели."""
    wd = np.asarray(pd.to_datetime(dts, unit="ms").dayofweek)
    first = int(np.argmax(wd == 0))
    W = (len(dr) - first) // 7
    sl = slice(first, first + W * 7)
    X = dr[sl].reshape(W, 7)
    V = dv[sl].reshape(W, 7).all(1)
    return X, V, dts[sl][::7]


def hourly_masks(cal: dict, fund_last: np.ndarray, D: int) -> dict[str, dict[str, np.ndarray]]:
    """Семьи часовых ячеек на сетке (D, 24)."""
    g = lambda k: cal[k][: D * 24].reshape(D, 24)  # noqa: E731
    hour, wday, et_h, nyse, lon_h = g("hour"), g("wday"), g("et_hour"), g("nyse"), g("lon_hour")
    wk = wday < 5
    fl = fund_last[: D * 24].reshape(D, 24)
    fam = {}
    fam["hod"] = {f"h{h:02d}": hour == h for h in range(24)}
    fam["block4h"] = {f"b{4 * k:02d}-{4 * k + 4:02d}": (hour // 4) == k for k in range(6)}
    pre = np.isin(hour, [23, 7, 15])
    post = np.isin(hour, [0, 8, 16])
    pre2 = np.isin(hour, [22, 23, 6, 7, 14, 15])
    post2 = np.isin(hour, [0, 1, 8, 9, 16, 17])
    hi = fl > 1.0001e-4
    neg = fl < 0
    fam["funding"] = {
        "pre1": pre, "post1": post, "pre2": pre2, "post2": post2,
        "pre1_fhi": pre & hi, "pre1_fneg": pre & neg, "post1_fhi": post & hi, "post1_fneg": post & neg,
    }  # fmt: skip
    fam["sessions"] = {
        "us_open_et9": (et_h == 9) & nyse,
        "us_et10": (et_h == 10) & nyse,
        "us_close_et15": (et_h == 15) & nyse,
        "us_after_et16": (et_h == 16) & nyse,
        "us_cash_et9_15": (et_h >= 9) & (et_h <= 15) & nyse,
        "us_et9_nontrading": (et_h == 9) & ~nyse,
        "asia_open_utc00_wk": (hour == 0) & wk,
        "asia_utc01_wk": (hour == 1) & wk,
        "asia_utc00_08_wk": (hour < 8) & wk,
        "london_open_lon8_wk": (lon_h == 8) & wk,
        "weekend_hours": ~wk,
    }
    return fam


def daily_masks(dts_weeks: np.ndarray, W: int) -> dict[str, dict[str, np.ndarray]]:
    """Семьи дневных ячеек на сетке (W, 7)."""
    days = (dts_weeks[:, None] + np.arange(7)[None, :] * DAY).ravel()
    t = pd.to_datetime(days, unit="ms")
    dom = np.asarray(t.day).reshape(W, 7)
    dfe = (np.asarray(t.days_in_month) - np.asarray(t.day)).reshape(W, 7)
    wd = np.tile(np.arange(7), (W, 1))
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    fam = {}
    fam["dow"] = {names[k]: wd == k for k in range(7)}
    fam["weekend"] = {"sat_sun": wd >= 5, "fri_sat_sun": wd >= 4, "mon": wd == 0}
    fam["tom"] = {
        "tom_m1_p3": (dfe == 0) | (dom <= 3),
        "tom_m2_p2": (dfe <= 1) | (dom <= 2),
        "tom_p1_p3": dom <= 3,
        "tom_d1": dom == 1,
        "tom_last": dfe == 0,
        "mid_month_10_20": (dom >= 10) & (dom <= 20),
    }
    return fam


def fund_last_known(a: dict) -> np.ndarray:
    """Последняя УЖЕ рассчитанная ставка фандинга на момент начала часа t (по расчётам ≤ ts[t]-1ч).
    Для BTC/ETH/ALT — 8-часовые расчёты (00/08/16); у ALT берём среднюю ставку только по 00/08/16."""
    f = a["fund"].copy()
    T = len(f)
    hour = (np.arange(T)) % 24  # ts[0] = 2020-01-01 00:00 UTC
    is_set = np.isin(hour, [0, 8, 16])
    last = np.full(T, np.nan)
    cur = np.nan
    for t in range(T):
        # на начало часа t известны расчёты в моменты ≤ ts[t]; используем строго прошлые (< ts[t]) —
        # ставка в момент ts[t] относится к расчёту на границе и на «пред-часе» неизвестна
        last[t] = cur
        if is_set[t]:
            cur = f[t]
    return last


def run_tests() -> None:
    t0 = time.time()
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    cal = calendar(ts)
    T = len(ts)
    D = T // 24
    dts = ts[: D * 24 : 24]
    yr_d = year_of(dts)
    i0, i1 = np.searchsorted(dts, ms(IS[0])), np.searchsorted(dts, ms(IS[1]))
    o0, o1 = np.searchsorted(dts, ms(OOS[0])), np.searchsorted(dts, ms(OOS[1]))
    cells = []
    for s in ASSETS:
        a = A[s]
        X, V, _ = hourly_grid(a, ts)
        fl = fund_last_known(a)
        fams = hourly_masks(cal, fl, D)
        rows_is = np.zeros(D, bool)
        rows_is[i0:i1] = True
        rows_is &= V
        rows_oos = np.zeros(D, bool)
        rows_oos[o0:o1] = True
        rows_oos &= V
        for fname, masks in fams.items():
            res = family_test(X[rows_is], {k: m[rows_is] for k, m in masks.items()}, seed=zlib.crc32(f'{s}/{fname}'.encode()))
            oos = simple_stats(X[rows_oos], {k: m[rows_oos] for k, m in masks.items()})
            per = {}
            for y in IS_YEARS + OOS_YEARS:
                ry = (yr_d == y) & V
                per[y] = simple_stats(X[ry], {k: m[ry] for k, m in masks.items()})
            for k, r in res.items():
                r.update({"asset": s, "family": fname, "cell": k, "grid": "hour"})
                r["oos"] = oos[k]
                r["year_diff_bp"] = {str(y): per[y][k].get("diff_bp") for y in per}
                r["year_t"] = {str(y): per[y][k].get("t") for y in per}
                ys = np.sign([per[y][k]["diff_bp"] for y in IS_YEARS])
                r["is_years_same_sign"] = int(np.sum(ys == np.sign(r["diff_bp"])))
                cells.append(r)
        print(f"{s} hourly families done {time.time() - t0:.0f}s", flush=True)
        # дневные семьи
        dr, dv, ddts = daily_returns(a, ts)
        Xw, Vw, wts = weekly_grid(dr, dv, ddts)
        W = len(Xw)
        fams = daily_masks(wts, W)
        wyr = year_of(wts + 3 * DAY)
        w_is = (wts >= ms(IS[0])) & (wts + 7 * DAY <= ms(IS[1])) & Vw
        w_oos = (wts >= ms(OOS[0])) & (wts + 7 * DAY <= ms(OOS[1])) & Vw
        for fname, masks in fams.items():
            res = family_test(Xw[w_is], {k: m[w_is] for k, m in masks.items()}, seed=zlib.crc32(f'{s}/{fname}'.encode()))
            oos = simple_stats(Xw[w_oos], {k: m[w_oos] for k, m in masks.items()})
            per = {}
            for y in IS_YEARS + OOS_YEARS:
                ry = (wyr == y) & Vw
                per[y] = simple_stats(Xw[ry], {k: m[ry] for k, m in masks.items()})
            for k, r in res.items():
                r.update({"asset": s, "family": fname, "cell": k, "grid": "day"})
                r["oos"] = oos[k]
                r["year_diff_bp"] = {str(y): per[y][k].get("diff_bp") for y in per}
                r["year_t"] = {str(y): per[y][k].get("t") for y in per}
                ys = np.sign([per[y][k]["diff_bp"] for y in IS_YEARS])
                r["is_years_same_sign"] = int(np.sum(ys == np.sign(r["diff_bp"])))
                cells.append(r)
        print(f"{s} daily families done {time.time() - t0:.0f}s", flush=True)
    p = np.array([c["p"] for c in cells])
    adj = holm(p)
    for c, pa in zip(cells, adj):
        c["p_holm_global"] = float(pa)
        c["p_bonf_global"] = float(min(1.0, c["p"] * len(cells)))
        c["significant"] = bool(pa < 0.05)
        c["significant_stable"] = bool(pa < 0.05 and c["is_years_same_sign"] >= 3)
    save_json({"n_cells": len(cells), "cells": cells}, "part_tests.json")
    print(f"cells tested: {len(cells)}; Holm-significant: {sum(c['significant'] for c in cells)}; "
          f"stable: {sum(c['significant_stable'] for c in cells)}; perm-FWER<0.05: "
          f"{sum(c['p_fwer_perm'] < 0.05 for c in cells)}")  # fmt: skip
    top = sorted(cells, key=lambda c: c["p"])[:40]
    for c in top:
        print(f"{c['asset']:6s} {c['family']:9s} {c['cell']:22s} n={c['n']:5d} in={c['mean_bp']:7.2f} "
              f"d={c['diff_bp']:7.2f}bp t={c['t']:6.2f} tabs={c['t_abs']:5.2f} pHolm={c['p_holm_global']:.3f} "
              f"pFW={c['p_fwer_perm']:.3f} yrs={c['is_years_same_sign']} | OOS d={c['oos'].get('diff_bp', float('nan')):7.2f} "
              f"t={c['oos'].get('t', float('nan')):5.2f} | y " + " ".join(f"{(v if v is not None else float('nan')):6.1f}" for v in c['year_diff_bp'].values()))  # fmt: skip


# ----------------------------------------------------------------------------- B: гэп CME


def cme_weeks(ts: np.ndarray) -> dict[str, np.ndarray]:
    """Моменты закрытия CME в пятницу (16:00 America/Chicago) и открытия в воскресенье (17:00 CT)
    в мс UTC, а также следующее пятничное закрытие; плацебо — вт 16:00 CT → чт 17:00 CT (те же 49 ч)."""
    fr = pd.date_range("2020-01-03", "2026-09-25", freq="W-FRI")
    def at(days, hh, off):  # noqa: E306
        loc = pd.DatetimeIndex([pd.Timestamp(d.date()) + pd.Timedelta(days=off, hours=hh) for d in days])
        return np.asarray(loc.tz_localize("America/Chicago").tz_convert("UTC").as_unit("ms").asi8, dtype=np.int64)
    out = {
        "fri_close": at(fr, 16, 0), "sun_open": at(fr, 17, 2), "next_fri_close": at(fr, 16, 7),
        "tue_close": at(fr, 16, -3), "thu_open": at(fr, 17, -1), "thu_next": at(fr, 16, 4),
    }  # fmt: skip
    return out


def ols_robust(x: np.ndarray, y: np.ndarray) -> dict:
    """y = a + b x, SE по Уайту (HC1)."""
    n = len(x)
    X = np.c_[np.ones(n), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None] ** 2).T @ X
    V = XtX_inv @ S @ XtX_inv * n / max(n - 2, 1)
    se = np.sqrt(np.diag(V))
    return {"n": int(n), "a": float(beta[0]), "b": float(beta[1]), "t_b": float(beta[1] / se[1]) if se[1] > 0 else 0.0,
            "corr": float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")}  # fmt: skip


def gap_events(a: dict, ts: np.ndarray, t_close: np.ndarray, t_open: np.ndarray, t_end: np.ndarray) -> dict:
    """Для каждого окна: гэп = P(open)/P(close)-1 (P(x) = open бара, начинающегося в x), доходности после
    открытия и факт «закрытия гэпа» (цена вернулась к P(close)) до t_end по high/low часов."""
    idx = lambda x: np.searchsorted(ts, x)  # noqa: E731
    ic, io, ie = idx(t_close), idx(t_open), idx(t_end)
    ok = (ic < len(ts)) & (ie < len(ts)) & (ts[np.minimum(ic, len(ts) - 1)] == t_close) & (ts[np.minimum(io, len(ts) - 1)] == t_open)
    ic, io, ie = ic[ok], io[ok], ie[ok]
    # цена в момент x = close бара, закончившегося в x (бар x-1ч); вход — по open бара x (следующий бар)
    C = a["c"]
    P = a["o"]
    pc, po = C[ic - 1], C[io - 1]
    gap = po / pc - 1
    hz = {"1h": 1, "6h": 6, "24h": 24, "48h": 48}
    fwd = {k: P[np.minimum(io + h, len(ts) - 1)] / po - 1 for k, h in hz.items()}
    fwd["to_fri"] = P[ie] / po - 1
    filled = np.zeros(len(io), bool)
    t_fill = np.full(len(io), np.nan)
    for k in range(len(io)):
        hh, ll = a["h"][io[k] : ie[k]], a["l"][io[k] : ie[k]]
        hit = (ll <= pc[k]) if gap[k] > 0 else (hh >= pc[k])
        if hit.any():
            filled[k] = True
            t_fill[k] = int(np.argmax(hit)) + 1
    return {"ts_close": t_close[ok], "i_open": io, "i_end": ie, "gap": gap, "fwd": fwd, "filled": filled, "t_fill": t_fill, "p_close": pc}


def run_cme() -> None:
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    a = A["BTC"]
    W = cme_weeks(ts)
    real = gap_events(a, ts, W["fri_close"], W["sun_open"], W["next_fri_close"])
    plac = gap_events(a, ts, W["tue_close"], W["thu_open"], W["thu_next"])
    res = {"real": {}, "placebo": {}}
    for name, ev in (("real", real), ("placebo", plac)):
        y = year_of(ev["ts_close"])
        for per, m in (("IS", y <= 2023), ("OOS", y >= 2024), *((str(yy), y == yy) for yy in IS_YEARS + OOS_YEARS)):
            g = ev["gap"][m]
            d = {"n": int(m.sum()), "mean_abs_gap_pct": float(np.mean(np.abs(g)) * 100), "fill_rate": float(ev["filled"][m].mean()),
                 "median_hours_to_fill": float(np.nanmedian(ev["t_fill"][m])) if ev["filled"][m].any() else None}  # fmt: skip
            for k, f in ev["fwd"].items():
                d[f"reg_{k}"] = ols_robust(g, f[m])
                # «фейд»: доходность позиции −sign(gap)
                fade = -np.sign(g) * f[m]
                d[f"fade_{k}_mean_bp"] = float(fade.mean() * 1e4)
                d[f"fade_{k}_t"] = float(fade.mean() / (fade.std(ddof=1) / math.sqrt(len(fade)))) if len(fade) > 2 else 0.0
            # заполнение по корзинам |gap|
            bins = [0, 0.0025, 0.005, 0.01, 0.02, 1.0]
            fb = {}
            for lo, hi in zip(bins[:-1], bins[1:]):
                mm = (np.abs(g) >= lo) & (np.abs(g) < hi)
                fb[f"{lo * 100:.2f}-{hi * 100:.2f}%"] = {"n": int(mm.sum()), "fill": float(ev["filled"][m][mm].mean()) if mm.any() else None}
            d["fill_by_gap"] = fb
            res[name][per] = d
    # «заполнение» против случайного блуждания: |gap| в единицах σ следующих 119 ч (σ — по 720 прошлым
    # часам до момента закрытия), теоретическая вероятность касания 2·(1−Φ(z)) без дрейфа
    r = np.nan_to_num(a["r"])
    cs = np.r_[0, np.cumsum(r)]
    cs2 = np.r_[0, np.cumsum(r * r)]
    for name, ev in (("real", real), ("placebo", plac)):
        ic = np.searchsorted(ts, ev["ts_close"])
        n = 720
        mu = (cs[ic] - cs[ic - n]) / n
        sd = np.sqrt(np.maximum((cs2[ic] - cs2[ic - n]) / n - mu**2, 1e-12))
        z = np.abs(ev["gap"]) / (sd * math.sqrt(119))
        theo = 2 * norm_sf(z)
        y = year_of(ev["ts_close"])
        for per, m in (("IS", y <= 2023), ("OOS", y >= 2024)):
            zb = {}
            for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 9.0)):
                mm = m & (z >= lo) & (z < hi)
                zb[f"z{lo}-{hi}"] = {"n": int(mm.sum()), "fill": float(ev["filled"][mm].mean()) if mm.any() else None,
                                     "theory_rw": float(theo[mm].mean()) if mm.any() else None}  # fmt: skip
            res[name][per]["fill_by_z"] = zb
            res[name][per]["fill_minus_theory"] = float(ev["filled"][m].mean() - theo[m].mean())
    save_json(res, "part_cme.json")
    for name in res:
        for per in ("IS", "OOS", "2020", "2021", "2022", "2023", "2024", "2025", "2026"):
            d = res[name][per]
            print(f"{name:7s} {per:4s} n={d['n']:3d} |gap|={d['mean_abs_gap_pct']:.2f}% fill={d['fill_rate']:.2f} "
                  f"b24={d['reg_24h']['b']:6.3f}(t{d['reg_24h']['t_b']:5.2f}) bFri={d['reg_to_fri']['b']:6.3f}(t{d['reg_to_fri']['t_b']:5.2f}) "
                  f"fade24={d['fade_24h_mean_bp']:6.1f}bp(t{d['fade_24h_t']:5.2f}) fadeFri={d['fade_to_fri_mean_bp']:6.1f}bp(t{d['fade_to_fri_t']:5.2f})")  # fmt: skip
        print(name, "fill by |gap| IS:", res[name]["IS"]["fill_by_gap"])
        print(name, "fill by |gap| OOS:", res[name]["OOS"]["fill_by_gap"])
        for per in ("IS", "OOS"):
            print(name, per, "fill-theory", round(res[name][per]["fill_minus_theory"], 3), res[name][per]["fill_by_z"])


# ----------------------------------------------------------------------------- C: симуляция стратегий

from numba import njit  # noqa: E402


def sim_taker(w: np.ndarray, a: dict, pess: bool = False) -> dict[str, np.ndarray]:
    """Позиция w[t] держится на баре t (от open[t] ≈ close[t-1] до close[t]). Смена позиции на границе
    t-1|t — рыночной заявкой, издержки |Δw|·cost_per_side(ADV до t-1). Фандинг, рассчитанный в момент
    ts[t] (граница), платится по позиции на границе; если позиция на границе меняется — берётся худший
    из двух вариантов (консервативно)."""
    w = np.where(a.get("tradable", a["valid"]), w, 0.0)
    r = np.nan_to_num(a["r_pess"] if pess and "r_pess" in a else a["r"])
    wp = np.r_[0.0, w[:-1]]
    f = a["fund"]
    fund = np.where(w == wp, -w * f, np.minimum(-wp * f, -w * f))
    gross = w * r
    cost = np.abs(w - wp) * a["cost"]
    return {"net": gross - cost + fund, "gross": gross, "cost": cost, "fund": fund, "w": w}


@njit(cache=True)
def _maker_loop(w, c, h, l, r, ctaker, f, fee, thr):  # noqa: E741
    T = len(w)
    gross = np.zeros(T)
    cost = np.zeros(T)
    fund = np.zeros(T)
    held = np.zeros(T)
    pos = 0.0
    nfill = 0
    nmiss = 0
    for t in range(1, T):
        fund[t] = -pos * f[t]  # позиция на границе — до исполнения лимитки
        tgt = w[t]
        if tgt == pos:
            gross[t] = pos * r[t]
            held[t] = pos
            continue
        P = c[t - 1]
        d = tgt - pos
        if d > 0:
            filled = l[t] < P * (1.0 - thr)
        else:
            filled = h[t] > P * (1.0 + thr)
        if filled:
            gross[t] = tgt * r[t]
            cost[t] = abs(d) * fee
            nfill += 1
        else:  # не исполнилась за час → рыночная заявка по close[t]
            gross[t] = pos * r[t]
            cost[t] = abs(d) * ctaker[t]
            nmiss += 1
        pos = tgt
        held[t] = tgt
    return gross, cost, fund, held, nfill, nmiss


def sim_maker(w: np.ndarray, a: dict, thr: float = 1e-4, fee: float = MAKER_FEE) -> dict[str, np.ndarray]:
    """Консервативный мейкер: лимитка по close[t-1] (= цена на границе); исполнена, только если за час
    t цена прошла СКВОЗЬ лимит на thr; иначе — рыночная заявка по close[t] с taker-издержками.
    Комиссия мейкера 0.02%."""
    w = np.where(a.get("tradable", a["valid"]), w, 0.0).astype(np.float64)
    r = np.nan_to_num(a["r"])
    g, c, f, held, nf, nm = _maker_loop(w, np.nan_to_num(a["c"]), np.nan_to_num(a["h"]), np.nan_to_num(a["l"]), r,
                                        a["cost"].astype(np.float64), a["fund"].astype(np.float64), fee, thr)  # fmt: skip
    return {"net": g - c + f, "gross": g, "cost": c, "fund": f, "w": w, "fill_rate": nf / max(nf + nm, 1)}


def strat_stats(ts: np.ndarray, res: dict) -> dict:
    """IS/OOS метрики, по годам, полугодия OOS, сделки, оборот, экспозиция, удержание, доля фандинга."""
    net, w = res["net"], res["w"]
    out = {}
    for name, win in (("is", IS), ("oos", OOS)):
        i0, i1 = np.searchsorted(ts, ms(win[0])), np.searchsorted(ts, ms(win[1]))
        ww = w[i0:i1]
        wp = np.r_[0.0, ww[:-1]]
        entries = int(np.sum((ww != 0) & (ww != wp)))
        m = metrics(ts[i0:i1], net[i0:i1], trades=entries)
        yrs = (i1 - i0) / 24 / 365.25
        m["turnover_x_per_year"] = round(float(np.abs(ww - wp).sum() / yrs), 1)
        m["exposure"] = round(float(np.mean(np.abs(ww))), 3)
        m["avg_hold_h"] = round(float(np.sum(ww != 0) / max(entries, 1)), 1)
        g, c, f = res["gross"][i0:i1].sum(), res["cost"][i0:i1].sum(), res["fund"][i0:i1].sum()
        m["sum_gross_pct"] = round(float(g) * 100, 2)
        m["sum_cost_pct"] = round(float(c) * 100, 2)
        m["sum_fund_pct"] = round(float(f) * 100, 2)
        out[name] = m
    out["per_year"] = per_year(ts, net)
    # полугодия OOS
    i0 = np.searchsorted(ts, ms(OOS[0]))
    hy = {}
    for y in OOS_YEARS:
        for h, (a0, a1) in enumerate(((f"{y}-01-01", f"{y}-07-01"), (f"{y}-07-01", f"{y + 1}-01-01"))):
            j0, j1 = np.searchsorted(ts, ms(a0)), np.searchsorted(ts, ms(a1))
            if j1 > j0 and j0 >= i0:
                hy[f"{y}H{h + 1}"] = round((float(np.prod(1 + net[j0:j1])) - 1) * 100, 2)
    out["oos_halfyears"] = hy
    out["oos_pos_halfyears"] = int(sum(v > 0 for v in hy.values()))
    out["oos_n_halfyears"] = len(hy)
    if "fill_rate" in res:
        out["maker_fill_rate"] = round(float(res["fill_rate"]), 3)
    return out


def full_masks(cal: dict, fund_last: np.ndarray, T: int) -> dict[str, dict[str, np.ndarray]]:
    """Те же ячейки, что и в тестах, но как маски на всей часовой шкале [T]."""
    D = T // 24
    fam = {k: {c: m.ravel() for c, m in v.items()} for k, v in hourly_masks(cal, fund_last, D).items()}
    wd, dom, dfe = cal["wday"], cal["dom"], cal["dfe"]
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    fam["dow"] = {names[k]: wd == k for k in range(7)}
    fam["weekend"] = {"sat_sun": wd >= 5, "fri_sat_sun": wd >= 4, "mon": wd == 0}
    fam["tom"] = {
        "tom_m1_p3": (dfe == 0) | (dom <= 3), "tom_m2_p2": (dfe <= 1) | (dom <= 2), "tom_p1_p3": dom <= 3,
        "tom_d1": dom == 1, "tom_last": dfe == 0, "mid_month_10_20": (dom >= 10) & (dom <= 20),
    }  # fmt: skip
    return fam


def cme_positions(a: dict, ts: np.ndarray, W: dict, thr: float, H: int, tp: bool, follow: bool) -> np.ndarray:
    """Позиции стратегии по гэпу CME: на открытии CME в воскресенье (бар, начинающийся в sun_open) при
    |gap| ≥ thr — позиция −sign(gap) (фейд) или +sign(gap) (follow), держим H часов. tp=True: выход,
    когда цена дошла до пятничного закрытия (выход моделируется по close того часа, где касание
    произошло — приближение лимитки на уровне пятницы; close может быть как лучше, так и хуже уровня),
    не дольше H часов и не дальше закрытия CME в следующую пятницу."""
    T = len(ts)
    w = np.zeros(T)
    C = a["c"]
    for fc, so, nf in zip(W["fri_close"], W["sun_open"], W["next_fri_close"]):
        ic, io, ie = np.searchsorted(ts, fc), np.searchsorted(ts, so), np.searchsorted(ts, nf)
        if io >= T or ts[min(ic, T - 1)] != fc or ts[io] != so:
            continue
        # сигнал — по close бара, закончившегося в момент открытия CME; позиция с бара io (вход по open[io])
        pc, po = C[ic - 1], C[io - 1]
        if not (np.isfinite(pc) and np.isfinite(po)):
            continue
        gap = po / pc - 1
        if abs(gap) < thr or gap == 0:
            continue
        d = np.sign(gap) * (1 if follow else -1)
        end = min(io + H, ie, T)
        if tp and not follow:
            for t in range(io, end):
                w[t] = d
                hit = (a["l"][t] <= pc) if gap > 0 else (a["h"][t] >= pc)
                if hit:
                    break
        else:
            w[io:end] = d
    return w


def build_configs(b: dict, A: dict, cal: dict, tests: list[dict]) -> list[dict]:
    """Список конфигураций: (имя, актив, исполнение, позиции w). Направление и выбор ячеек — ТОЛЬКО по
    IS-статистикам part_tests.json (поля t, diff_bp, mean_bp по IS)."""
    ts = b["ts"]
    T = len(ts)
    cfgs = []
    by = {}
    for c in tests:
        by.setdefault((c["asset"], c["family"]), {})[c["cell"]] = c
    for s in ASSETS:
        a = A[s]
        fam = full_masks(cal, fund_last_known(a), T)
        execs = ["taker", "maker"] if s in ("BTC", "ETH") else ["taker"]
        hod = by[(s, "hod")]
        order = sorted(hod, key=lambda k: -hod[k]["t"])  # по убыванию IS t (сезонная компонента)
        hours = lambda ks: np.isin(cal["hour"], [int(k[1:]) for k in ks])  # noqa: E731
        sets = []
        for k in (1, 2, 3, 4, 6, 8, 12):
            top = [c for c in order[:k] if hod[c]["t"] > 0]
            if len(top) == k:
                sets.append((f"hod_long_top{k}", hours(top).astype(float), {"family": "hod", "k": k, "hours": top}))
        for k in (1, 2, 3, 4, 6, 8):
            bot = [c for c in order[::-1][:k] if hod[c]["t"] < 0]
            if len(bot) == k:
                sets.append((f"hod_short_bot{k}", -hours(bot).astype(float), {"family": "hod", "k": k, "hours": bot}))
                sets.append((f"hod_avoid_bot{k}", (~hours(bot)).astype(float), {"family": "hod", "k": k, "hours": bot}))
        for k in (1, 2, 3, 4, 6):
            top, bot = order[:k], order[::-1][:k]
            sets.append((f"hod_ls{k}", hours(top).astype(float) - hours(bot).astype(float), {"family": "hod", "k": k, "long": top, "short": bot}))
        blk = by[(s, "block4h")]
        bo = sorted(blk, key=lambda k: -blk[k]["t"])
        bm = fam["block4h"]
        for k in (1, 2):
            sets.append((f"block_long_top{k}", np.any([bm[c] for c in bo[:k]], 0).astype(float), {"family": "block4h", "k": k, "cells": bo[:k]}))
            sets.append((f"block_short_bot{k}", -np.any([bm[c] for c in bo[::-1][:k]], 0).astype(float), {"family": "block4h", "k": k, "cells": bo[::-1][:k]}))
            sets.append((f"block_ls{k}", np.any([bm[c] for c in bo[:k]], 0).astype(float) - np.any([bm[c] for c in bo[::-1][:k]], 0).astype(float),
                         {"family": "block4h", "k": k}))  # fmt: skip
        for fname in ("funding", "sessions", "dow", "weekend", "tom"):
            for cell, st in by[(s, fname)].items():
                m = fam[fname][cell].astype(float)
                d = 1.0 if st["t"] > 0 else -1.0
                sets.append((f"{fname}_{cell}_{'long' if d > 0 else 'short'}", d * m, {"family": fname, "cell": cell, "dir": d}))
                if d < 0:  # «избегать плохой ячейки»: лонг всё остальное время
                    sets.append((f"{fname}_{cell}_avoid", 1.0 - m, {"family": fname, "cell": cell, "dir": "avoid"}))
        dw = by[(s, "dow")]
        do = sorted(dw, key=lambda k: -dw[k]["t"])
        for k in (2, 3):
            sets.append((f"dow_long_top{k}", np.any([fam["dow"][c] for c in do[:k]], 0).astype(float), {"family": "dow", "k": k, "days": do[:k]}))
            sets.append((f"dow_avoid_bot{k}", 1.0 - np.any([fam["dow"][c] for c in do[::-1][:k]], 0).astype(float), {"family": "dow", "k": k, "days": do[::-1][:k]}))
        sets.append(("buy_and_hold", np.ones(T), {"family": "baseline"}))
        for name, w, meta in sets:
            for ex in execs:
                cfgs.append({"name": f"{s}:{name}:{ex}", "asset": s, "exec": ex, "w": w, "meta": meta})
    # гэп CME (BTC)
    W = cme_weeks(ts)
    for follow in (False, True):
        for thr in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
            for H in (6, 24, 48, 72, 119):
                for tp in ((False, True) if not follow else (False,)):
                    w = cme_positions(A["BTC"], ts, W, thr, H, tp, follow)
                    nm = f"BTC:cme_{'follow' if follow else 'fade'}_thr{thr * 100:.1f}_H{H}{'_tp' if tp else ''}"
                    for ex in ("taker", "maker"):
                        cfgs.append({"name": f"{nm}:{ex}", "asset": "BTC", "exec": ex, "w": w,
                                     "meta": {"family": "cme", "thr": thr, "H": H, "tp": tp, "follow": follow}})  # fmt: skip
    return cfgs


def run_config(cfg: dict, A: dict) -> dict:
    a = A[cfg["asset"]]
    return sim_maker(cfg["w"], a) if cfg["exec"] == "maker" else sim_taker(cfg["w"], a)


def run_strats() -> None:
    t0 = time.time()
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    cal = calendar(ts)
    tests = load_json("part_tests.json")["cells"]
    cfgs = build_configs(b, A, cal, tests)
    print(f"{len(cfgs)} configs built in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for cfg in cfgs:
        res = run_config(cfg, A)
        st = strat_stats(ts, res)
        rows.append({"name": cfg["name"], "asset": cfg["asset"], "exec": cfg["exec"], "meta": cfg["meta"], **st})
    print(f"simulated in {time.time() - t0:.0f}s", flush=True)
    # случайные наборы часов (тот же k, тот же оборот) — нулевое распределение для hod_long_topk
    rng = np.random.default_rng(7)
    rand = {}
    for s in ("BTC", "ETH", "ALT"):
        for k in (1, 2, 4, 8):
            sh = []
            for _ in range(200):
                hs = rng.choice(24, k, replace=False)
                w = np.isin(cal["hour"], hs).astype(float)
                st = strat_stats(ts, sim_taker(w, A[s]))
                sh.append((st["is"]["sharpe"], st["oos"]["sharpe"]))
            sh = np.array(sh)
            rand[f"{s}_k{k}"] = {"is_sharpe_median": float(np.median(sh[:, 0])), "oos_sharpe_median": float(np.median(sh[:, 1])),
                                 "oos_sharpe_p90": float(np.quantile(sh[:, 1], 0.9)), "is_sharpe_p95": float(np.quantile(sh[:, 0], 0.95)),
                                 "corr_is_oos": float(np.corrcoef(sh[:, 0], sh[:, 1])[0, 1])}  # fmt: skip
    print(f"random baselines in {time.time() - t0:.0f}s", flush=True)
    save_json({"configs": rows, "random_hours": rand}, "part_strats.json")
    ok = [r for r in rows if r["is"].get("trades", 0) >= 30 and r["meta"].get("family") != "baseline"]
    is_sh = np.array([r["is"]["sharpe"] for r in ok])
    oos_sh = np.array([r["oos"]["sharpe"] for r in ok])
    print(f"configs (>=30 IS trades): {len(ok)}; median IS Sharpe {np.median(is_sh):.2f}; median OOS Sharpe {np.median(oos_sh):.2f}; "
          f"share OOS>0 {np.mean(oos_sh > 0):.2f}; corr IS/OOS {np.corrcoef(is_sh, oos_sh)[0, 1]:.2f}")  # fmt: skip
    for r in sorted(ok, key=lambda r: -r["is"]["sharpe"])[:30]:
        print(f"{r['name']:45s} IS sh={r['is']['sharpe']:5.2f} cagr={r['is']['cagr_pct']:7.1f} dd={r['is']['max_dd_pct']:5.1f} tr={r['is']['trades']:5d} "
              f"| OOS sh={r['oos']['sharpe']:5.2f} cagr={r['oos']['cagr_pct']:7.1f} dd={r['oos']['max_dd_pct']:5.1f} hy+={r['oos_pos_halfyears']}/{r['oos_n_halfyears']} "
              f"cost={r['is']['sum_cost_pct']:.0f}%")  # fmt: skip
    for r in rows:
        if r["meta"].get("family") == "baseline":
            print(f"{r['name']:45s} IS sh={r['is']['sharpe']:5.2f} cagr={r['is']['cagr_pct']:7.1f} | OOS sh={r['oos']['sharpe']:5.2f} cagr={r['oos']['cagr_pct']:7.1f}")
    for k, v in rand.items():
        print("random", k, v)


# ----------------------------------------------------------------------------- диагностика: до-IS и монеты


def _cells_simple(a: dict, ts: np.ndarray, rows_sel: np.ndarray | None = None) -> dict[str, dict]:
    """t-статистики всех часовых и дневных ячеек (кроме фандинга) для ряда a на всей длине ts."""
    cal = calendar(ts)
    T = len(ts)
    D = T // 24
    X, V, _ = hourly_grid(a, ts)
    if rows_sel is not None:
        V = V & rows_sel
    fams = hourly_masks(cal, np.zeros(T), D)
    out = {}
    for fname, masks in fams.items():
        if fname == "funding":
            continue
        st = simple_stats(X[V], {k: m[V] for k, m in masks.items()})
        for k, v in st.items():
            out[f"{fname}/{k}"] = v
    dr, dv, ddts = daily_returns(a, ts)
    Xw, Vw, wts = weekly_grid(dr, dv, ddts)
    if rows_sel is not None:
        wsel = np.asarray(rows_sel[: len(dr)])
        first = int(np.argmax(np.asarray(pd.to_datetime(ddts, unit="ms").dayofweek) == 0))
        Vw = Vw & wsel[first : first + 7 * len(Xw) : 7]
    fams = daily_masks(wts, len(Xw))
    for fname, masks in fams.items():
        st = simple_stats(Xw[Vw], {k: m[Vw] for k, m in masks.items()})
        for k, v in st.items():
            out[f"{fname}/{k}"] = v
    return out


def run_presample() -> None:
    """Спот BTC/ETH 2017-08 … 2019-12: знак ячеек против IS (только диагностика)."""
    from scripts.edge_lab.seasonality_spot import load as load_spot

    tests = load_json("part_tests.json")["cells"]
    res = {}
    for s in ("BTC", "ETH"):
        d = load_spot(f"{s}USDT")
        t0 = int(ms("2017-08-18"))
        t1 = int(ms("2020-01-01"))
        ts = np.arange(t0, t1, HOUR, dtype=np.int64)
        idx = np.searchsorted(ts, d["ts"])
        ok = (idx < len(ts)) & (ts[np.minimum(idx, len(ts) - 1)] == d["ts"])
        c = np.full(len(ts), np.nan)
        c[idx[ok]] = d["ohlc"][ok, 3]
        r = np.full(len(ts), np.nan)
        r[1:] = c[1:] / c[:-1] - 1
        a = {"r": r, "valid": np.isfinite(r)}
        pre = _cells_simple(a, ts)
        agree, n = 0, 0
        rows = []
        for cobj in tests:
            if cobj["asset"] != s or cobj["family"] == "funding":
                continue
            k = f"{cobj['family']}/{cobj['cell']}"
            if k not in pre or not np.isfinite(pre[k].get("diff_bp", np.nan)):
                continue
            same = np.sign(pre[k]["diff_bp"]) == np.sign(cobj["diff_bp"])
            agree += int(same)
            n += 1
            rows.append({"cell": k, "is_t": cobj["t"], "is_diff_bp": cobj["diff_bp"], "pre_t": pre[k]["t"], "pre_diff_bp": pre[k]["diff_bp"],
                         "oos_t": cobj["oos"].get("t"), "same_sign": bool(same)})  # fmt: skip
        is_t = np.array([r_["is_t"] for r_ in rows])
        pre_t = np.array([r_["pre_t"] for r_ in rows])
        oos_t = np.array([r_["oos_t"] for r_ in rows], dtype=float)
        res[s] = {"cells": rows, "sign_agree_pre_vs_is": f"{agree}/{n}",
                  "corr_t_is_pre": float(np.corrcoef(is_t, pre_t)[0, 1]), "corr_t_is_oos": float(np.corrcoef(is_t, oos_t)[0, 1]),
                  "corr_t_pre_oos": float(np.corrcoef(pre_t, oos_t)[0, 1])}  # fmt: skip
        print(s, "pre-sample: sign agreement with IS", f"{agree}/{n}", "corr t IS/pre", round(res[s]["corr_t_is_pre"], 2),
              "IS/OOS", round(res[s]["corr_t_is_oos"], 2), "pre/OOS", round(res[s]["corr_t_pre_oos"], 2))  # fmt: skip
        for r_ in sorted(rows, key=lambda x: -abs(x["is_t"]))[:12]:
            print(f"   {r_['cell']:32s} IS t={r_['is_t']:5.2f} d={r_['is_diff_bp']:7.2f} | pre t={r_['pre_t']:5.2f} d={r_['pre_diff_bp']:7.2f} | OOS t={r_['oos_t']:5.2f}")
    save_json(res, "part_presample.json")


def run_singles() -> None:
    """Ячейки ALT-индекса на отдельных ликвидных альтах (IS и OOS), для проверки «альтовых» эффектов."""
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    tests = {(c["family"], c["cell"]): c for c in load_json("part_tests.json")["cells"] if c["asset"] == "ALT"}
    D = len(ts) // 24
    dts = ts[: D * 24 : 24]
    is_rows = (dts >= ms(IS[0])) & (dts < ms(IS[1]))
    oos_rows = (dts >= ms(OOS[0])) & (dts < ms(OOS[1]))
    res = {}
    for s in SINGLES:
        st_is = _cells_simple(A[s], ts, is_rows)
        st_oos = _cells_simple(A[s], ts, oos_rows)
        res[s] = {"is": st_is, "oos": st_oos}
    # сводка по топ-ячейкам ALT (по IS |t|)
    top = sorted(tests.values(), key=lambda c: -abs(c["t"]))[:8]
    summ = []
    for c in top:
        k = f"{c['family']}/{c['cell']}"
        if c["family"] == "funding":
            continue
        sgn = np.sign(c["diff_bp"])
        is_same = sum(np.sign(res[s]["is"][k]["diff_bp"]) == sgn for s in SINGLES if k in res[s]["is"])
        oos_same = sum(np.sign(res[s]["oos"][k]["diff_bp"]) == sgn for s in SINGLES if k in res[s]["oos"])
        med_is = float(np.median([res[s]["is"][k]["diff_bp"] for s in SINGLES]))
        med_oos = float(np.median([res[s]["oos"][k]["diff_bp"] for s in SINGLES]))
        summ.append({"cell": k, "alt_is_t": c["t"], "alt_is_diff_bp": c["diff_bp"], "singles_is_same_sign": f"{is_same}/{len(SINGLES)}",
                     "singles_oos_same_sign": f"{oos_same}/{len(SINGLES)}", "singles_median_is_diff_bp": med_is, "singles_median_oos_diff_bp": med_oos})  # fmt: skip
        print(summ[-1])
    save_json({"summary": summ, "per_coin": res}, "part_singles.json")


# ----------------------------------------------------------------------------- D: устойчивость лучшего


def pick_best(rows: list[dict]) -> dict:
    """Лучший по IS Sharpe (после издержек) среди всех конфигураций с ≥30 сделками на IS, без бенчмарков.
    OOS при выборе не используется."""
    ok = [r for r in rows if r["is"].get("trades", 0) >= 30 and r["meta"].get("family") != "baseline"]
    return max(ok, key=lambda r: r["is"]["sharpe"])


def tom_mask(cal: dict, before: int, after: int) -> np.ndarray:
    """Окно перелома месяца: последние `before` календарных дней месяца + первые `after` дней (UTC)."""
    m = cal["dom"] <= after
    if before > 0:
        m = m | (cal["dfe"] <= before - 1)
    return m


def _short(st: dict) -> dict:
    return {"is_sharpe": st["is"]["sharpe"], "is_cagr": st["is"]["cagr_pct"], "oos_sharpe": st["oos"]["sharpe"],
            "oos_cagr": st["oos"]["cagr_pct"], "oos_dd": st["oos"]["max_dd_pct"], "oos_pos_halfyears": f"{st['oos_pos_halfyears']}/{st['oos_n_halfyears']}",
            "is_trades": st["is"].get("trades"), "per_year": st["per_year"]}  # fmt: skip


def run_robust() -> None:
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    cal = calendar(ts)
    part = load_json("part_strats.json")
    rows = part["configs"]
    best = pick_best(rows)
    print("IS-chosen:", best["name"], best["meta"], "IS", best["is"]["sharpe"], "OOS", best["oos"]["sharpe"])
    out = {"best_name": best["name"], "variants": {}}
    meta, s, ex = best["meta"], best["asset"], best["exec"]
    run = lambda w, asset=s, exe=ex, a_over=None: strat_stats(ts, sim_maker(w, a_over or A[asset]) if exe == "maker" else sim_taker(w, a_over or A[asset]))  # noqa: E731
    if meta["family"] == "tom":
        base_ba = {"tom_m1_p3": (1, 3), "tom_m2_p2": (2, 2), "tom_p1_p3": (0, 3), "tom_d1": (0, 1), "tom_last": (1, 0)}.get(meta["cell"])
        if base_ba:
            bb, aa = base_ba
            for b2, a2 in sorted({(bb, aa), (max(bb - 1, 0), aa), (bb + 1, aa), (bb, max(aa - 1, 0)), (bb, aa + 1), (bb + 1, aa + 1), (max(bb - 1, 0), max(aa - 1, 0))}):
                if b2 + a2 == 0:
                    continue
                w = tom_mask(cal, b2, a2).astype(float) * (1.0 if meta["dir"] > 0 else -1.0)
                out["variants"][f"window_last{b2}_first{a2}"] = _short(run(w))
            w0 = tom_mask(cal, bb, aa).astype(float) * (1.0 if meta["dir"] > 0 else -1.0)
            # исполнение и издержки
            out["variants"]["exec_taker"] = _short(run(w0, exe="taker"))
            a_hi = dict(A[s])
            a_hi["cost"] = A[s]["cost"] * 1.25
            out["variants"]["exec_maker_cost_x1.25"] = _short(strat_stats(ts, sim_maker(w0, a_hi, fee=MAKER_FEE * 1.25)))
            out["variants"]["exec_taker_cost_x1.25"] = _short(strat_stats(ts, sim_taker(w0, a_hi)))
            out["variants"]["maker_thr_5bp"] = _short(strat_stats(ts, sim_maker(w0, A[s], thr=5e-4)))
            # сдвиг входа/выхода на ±6 часов
            for sh in (-6, 6):
                out["variants"][f"shift_{sh:+d}h"] = _short(run(np.roll(w0, sh)))
            # тот же rule на других активах
            for s2 in ["BTC", "ETH", "ALT", "ALT30"] + SINGLES:
                ex2 = "maker" if s2 not in ("ALT", "ALT30") else "taker"
                out["variants"][f"asset_{s2}"] = _short(run(w0, asset=s2, exe=ex2))
            # пессимистичный вариант для корзин
            for s2 in ("ALT", "ALT30"):
                out["variants"][f"asset_{s2}_pess"] = _short(strat_stats(ts, sim_taker(w0, A[s2], pess=True)))
    # случайные подмножества дней недели (тот же оборот, лонг) — нулевое распределение для DOW-правил
    from itertools import combinations

    rnd = {}
    for s2 in ("BTC", "ETH"):
        for k in (2, 3):
            sh = []
            for comb in combinations(range(7), k):
                w = np.isin(cal["wday"], comb).astype(float)
                st = strat_stats(ts, sim_maker(w, A[s2]))
                sh.append((st["is"]["sharpe"], st["oos"]["sharpe"]))
            sh = np.array(sh)
            rnd[f"{s2}_dow_k{k}"] = {"n": len(sh), "is_median": float(np.median(sh[:, 0])), "oos_median": float(np.median(sh[:, 1])),
                                     "oos_max": float(sh[:, 1].max()), "corr_is_oos": float(np.corrcoef(sh[:, 0], sh[:, 1])[0, 1])}  # fmt: skip
            print("random DOW", s2, k, rnd[f"{s2}_dow_k{k}"])
    out["random_dow_subsets_maker"] = rnd
    # плацебо для правил «перелома месяца»: такое же 4-дневное окно, но в другие дни месяца
    if meta["family"] == "tom":
        L = sum(base_ba) if base_ba else 4
        plc = []
        for st0 in range(3, 29 - L + 1):
            w = ((cal["dom"] >= st0) & (cal["dom"] < st0 + L)).astype(float)
            st = strat_stats(ts, sim_maker(w, A[s]) if ex == "maker" else sim_taker(w, A[s]))
            plc.append({"first_day": st0, "is_sharpe": st["is"]["sharpe"], "oos_sharpe": st["oos"]["sharpe"]})
        isv = np.array([x["is_sharpe"] for x in plc])
        oov = np.array([x["oos_sharpe"] for x in plc])
        out["tom_placebo_windows"] = {"windows": plc, "is_median": float(np.median(isv)), "is_max": float(isv.max()),
                                      "oos_median": float(np.median(oov)), "share_is_above_best": float(np.mean(isv >= best["is"]["sharpe"])),
                                      "best_oos_rank_pct": float(np.mean(oov <= best["oos"]["sharpe"]))}  # fmt: skip
        print("TOM placebo windows:", {k: v for k, v in out["tom_placebo_windows"].items() if k != "windows"})
    # ПОСТ-ХОК диагностика (не участвует в выборе; добавлена после того, как профиль дней недели ETH
    # повторился в OOS — см. part_profiles.json): лонг 2 лучших дня IS / шорт 2 худших дня IS
    tests = load_json("part_tests.json")["cells"]
    ph = {}
    for s2 in ["BTC", "ETH", "ALT"] + SINGLES:
        src = s2 if s2 in ("BTC", "ETH", "ALT") else "ALT"  # дни для альтов — по IS-профилю ALT-индекса
        dw = {c["cell"]: c for c in tests if c["asset"] == src and c["family"] == "dow"}
        order = sorted(dw, key=lambda k: -dw[k]["t"])
        names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        lo = np.isin(cal["wday"], [names.index(k) for k in order[:2]])
        sh_ = np.isin(cal["wday"], [names.index(k) for k in order[-2:]])
        w = lo.astype(float) - sh_.astype(float)
        exe = "taker" if s2 == "ALT" else "maker"
        st = strat_stats(ts, sim_maker(w, A[s2]) if exe == "maker" else sim_taker(w, A[s2]))
        ph[s2] = {"long_days": order[:2], "short_days": order[-2:], "exec": exe, **_short(st), "oos_halfyears": st["oos_halfyears"],
                  "oos_cost_pct": st["oos"]["sum_cost_pct"], "oos_gross_pct": st["oos"]["sum_gross_pct"]}  # fmt: skip
        print("POSTHOC dow L/S", s2, order[:2], order[-2:], exe, "IS", st["is"]["sharpe"], "OOS", st["oos"]["sharpe"], st["per_year"])
    out["posthoc_dow_long_short"] = ph
    # TOM на споте 2017-08…2019-12 (до IS, только валовая доходность минус 0.07%/сторону)
    from scripts.edge_lab.seasonality_spot import load as load_spot

    pre = {}
    for s2 in ("BTC", "ETH"):
        d = load_spot(f"{s2}USDT")
        t = pd.to_datetime(d["ts"], unit="ms")
        dd = pd.Series(d["ohlc"][:, 3], index=t).resample("1D").last().dropna()
        rr = dd.pct_change().dropna()
        idx = rr.index
        dom = np.asarray(idx.day)
        dfe = np.asarray(idx.days_in_month) - dom
        if meta["family"] == "tom" and base_ba:
            bb, aa = base_ba
            m = (dom <= aa) | ((dfe <= bb - 1) if bb > 0 else np.zeros(len(dom), bool))
            sig = m.astype(float) * (1.0 if meta["dir"] > 0 else -1.0)
            ent = np.r_[sig[0] != 0, (sig[1:] != sig[:-1])]
            net = sig * rr.values - ent * 0.0007 * np.abs(np.r_[sig[0], np.diff(sig)])
            tsd = np.asarray(idx.as_unit("ms").asi8, dtype=np.int64)
            pre[s2] = metrics(tsd, net, trades=int(np.sum(np.diff(np.r_[0, sig]) > 0)))
            pre[s2]["per_year"] = per_year(tsd, net)
            print("pre-sample spot TOM", s2, pre[s2])
    out["presample_spot_rule"] = pre
    for k, v in out["variants"].items():
        print(f"{k:28s} IS {v['is_sharpe']:5.2f} ({v['is_cagr']:6.1f}%) | OOS {v['oos_sharpe']:5.2f} ({v['oos_cagr']:6.1f}%) dd {v['oos_dd']:5.1f} hy+ {v['oos_pos_halfyears']}")
    save_json(out, "part_robust.json")


# ----------------------------------------------------------------------------- профили: устойчивость формы


def _perm_corr(x: np.ndarray, y: np.ndarray, B: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Корреляция профилей и односторонний перестановочный p (H0: профиль OOS не связан с IS)."""
    from itertools import permutations

    c = float(np.corrcoef(x, y)[0, 1])
    if len(x) <= 8:
        perms = np.array(list(permutations(range(len(x)))))
    else:
        rng = np.random.default_rng(seed)
        perms = np.argsort(rng.random((B, len(x))), axis=1)
    yy = y[perms]
    xc = x - x.mean()
    yc = yy - yy.mean(1, keepdims=True)
    cs = (yc @ xc) / (np.sqrt((yc**2).sum(1)) * np.sqrt((xc**2).sum()))
    return c, float(np.mean(cs >= c - 1e-12))


def run_profiles() -> None:
    """Повторяется ли ФОРМА сезонного профиля (24 часа, 7 дней недели, 6 блоков по 4 ч) от периода к
    периоду: корреляция разностей «ячейка − остальное» IS↔OOS, IS↔до-IS (спот 2017–2019) и между
    соседними годами внутри IS; p — перестановочный. Это проверка стабильности, не выбор параметров."""
    tests = load_json("part_tests.json")["cells"]
    pre = load_json("part_presample.json")
    res = {}
    for s in ASSETS:
        for fam, order in (("hod", [f"h{h:02d}" for h in range(24)]), ("dow", ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]),
                           ("block4h", [f"b{4 * k:02d}-{4 * k + 4:02d}" for k in range(6)])):  # fmt: skip
            cells = {c["cell"]: c for c in tests if c["asset"] == s and c["family"] == fam}
            x = np.array([cells[k]["diff_bp"] for k in order])
            y = np.array([cells[k]["oos"]["diff_bp"] for k in order])
            c, pv = _perm_corr(x, y)
            d = {"corr_is_oos": c, "p_is_oos": pv}
            yrs = {y_: np.array([cells[k]["year_diff_bp"][str(y_)] for k in order], dtype=float) for y_ in IS_YEARS + OOS_YEARS}
            d["corr_consecutive_years"] = {f"{y_}-{y_ + 1}": float(np.corrcoef(yrs[y_], yrs[y_ + 1])[0, 1]) for y_ in IS_YEARS + OOS_YEARS[:-1]}
            if s in pre:
                pc = {r["cell"]: r for r in pre[s]["cells"]}
                z = np.array([pc[f"{fam}/{k}"]["pre_diff_bp"] for k in order])
                d["corr_pre_is"], d["p_pre_is"] = _perm_corr(z, x)
                d["corr_pre_oos"], d["p_pre_oos"] = _perm_corr(z, y)
            res[f"{s}/{fam}"] = d
            print(f"{s:6s} {fam:8s} IS↔OOS r={c:5.2f} p={pv:.3f} | " + (f"pre↔IS r={d['corr_pre_is']:5.2f} p={d['p_pre_is']:.3f} pre↔OOS r={d['corr_pre_oos']:5.2f} p={d['p_pre_oos']:.3f} | " if 'corr_pre_is' in d else "")
                  + "yrs " + " ".join(f"{v:5.2f}" for v in d["corr_consecutive_years"].values()))  # fmt: skip
    save_json(res, "part_profiles.json")


# ----------------------------------------------------------------------------- справочно: сезонность волатильности


def run_vol() -> None:
    """Сезонность ВОЛАТИЛЬНОСТИ и объёма (не доходности): средний |r| часа по блокам 4 ч UTC и
    выходные/будни, IS и OOS. Полезно для настройки шага сетки/DCA, но направленного края не даёт."""
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    cal = calendar(ts)
    res = {}
    for s in ("BTC", "ETH", "ALT"):
        a = A[s]
        r = np.abs(a["r"])
        v = a["valid"] & np.isfinite(r)
        qv = b[f"{s}_qvol"] if s in ("BTC", "ETH") else None
        d = {}
        for name, win in (("is", IS), ("oos", OOS)):
            m = v & (ts >= ms(win[0])) & (ts < ms(win[1]))
            base = r[m].mean()
            d[name] = {f"b{4 * k:02d}-{4 * k + 4:02d}": round(float(r[m & (cal["hour"] // 4 == k)].mean() / base), 3) for k in range(6)}
            d[name]["weekend_vs_all"] = round(float(r[m & (cal["wday"] >= 5)].mean() / base), 3)
            d[name]["us_cash_vs_all"] = round(float(r[m & (cal["et_hour"] >= 9) & (cal["et_hour"] <= 15) & cal["nyse"]].mean() / base), 3)
            d[name]["mean_abs_hourly_bp"] = round(float(base * 1e4), 1)
            if qv is not None:
                q = qv[m]
                d[name]["volume_weekend_vs_all"] = round(float(q[cal["wday"][m] >= 5].mean() / q.mean()), 3)
        res[s] = d
        print(s, d)
    save_json(res, "part_vol.json")


# ----------------------------------------------------------------------------- E: отчёт


def daily_from_hourly(ts: np.ndarray, net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = len(ts) // 24
    r = net[: D * 24].reshape(D, 24)
    return ts[: D * 24 : 24].astype(np.int64), np.prod(1 + r, axis=1) - 1


def run_report() -> None:
    b = load_base()
    ts = b["ts"]
    A = assets(b)
    cal = calendar(ts)
    tests = load_json("part_tests.json")
    strats = load_json("part_strats.json")
    rows = strats["configs"]
    best = pick_best(rows)
    # пересчёт лучшего: часовой ряд → дневной
    cfgs = {c["name"]: c for c in build_configs(b, A, cal, tests["cells"])}
    cfg = cfgs[best["name"]]
    res = run_config(cfg, A)
    st = strat_stats(ts, res)
    dts, dret = daily_from_hourly(ts, res["net"])
    ODIR.mkdir(parents=True, exist_ok=True)
    np.savez(ODIR / "best_daily.npz", ts=dts, ret=dret)
    # ёмкость: 1% среднего часового объёма (по 720 прошлым часам) в моменты входа
    w = res["w"]
    wp = np.r_[0.0, w[:-1]]
    ent = np.where((w != 0) & (w != wp))[0]
    a = A[cfg["asset"]]
    cap = a["adv_prev"][ent] / 24 * 0.01
    cap_d = {}
    for name, win in (("is", IS), ("oos", OOS)):
        m = (ts[ent] >= ms(win[0])) & (ts[ent] < ms(win[1]))
        cap_d[name] = {"min_usd": float(np.nanmin(cap[m])), "median_usd": float(np.nanmedian(cap[m]))}
    fund_share = {}
    for name in ("is", "oos"):
        g, f = st[name]["sum_gross_pct"], st[name]["sum_fund_pct"]
        fund_share[name] = {"price_pnl_pct": g, "funding_pnl_pct": f, "cost_pct": st[name]["sum_cost_pct"],
                            "funding_share_of_gross_abs": round(abs(f) / (abs(g) + abs(f)), 3) if (abs(g) + abs(f)) > 0 else None}  # fmt: skip
    # распределение OOS по всем конфигурациям
    ok = [r for r in rows if r["is"].get("trades", 0) >= 30 and r["meta"].get("family") != "baseline"]
    oos_sh = np.array([r["oos"]["sharpe"] for r in ok])
    is_sh = np.array([r["is"]["sharpe"] for r in ok])
    dist = {"count": len(ok), "median_oos_sharpe": float(np.median(oos_sh)), "share_oos_positive": float(np.mean(oos_sh > 0)),
            "share_oos_sharpe_ge_1": float(np.mean(oos_sh >= 1)), "median_is_sharpe": float(np.median(is_sh)),
            "corr_is_oos_sharpe": float(np.corrcoef(is_sh, oos_sh)[0, 1])}  # fmt: skip
    # топ-10% по IS → их OOS
    q = np.quantile(is_sh, 0.9)
    dist["top10pct_is_median_oos_sharpe"] = float(np.median(oos_sh[is_sh >= q]))
    fam = {}
    for r in ok:
        fam.setdefault(r["meta"]["family"], []).append(r)
    fam_sum = {}
    for f, rr in fam.items():
        bb = max(rr, key=lambda r: r["is"]["sharpe"])
        o = np.array([r["oos"]["sharpe"] for r in rr])
        fam_sum[f] = {"n": len(rr), "median_oos_sharpe": float(np.median(o)), "share_oos_pos": float(np.mean(o > 0)),
                      "is_best": bb["name"], "is_best_is_sharpe": bb["is"]["sharpe"], "is_best_oos_sharpe": bb["oos"]["sharpe"],
                      "is_best_oos_cagr": bb["oos"]["cagr_pct"], "is_best_oos_pos_halfyears": f"{bb['oos_pos_halfyears']}/{bb['oos_n_halfyears']}"}  # fmt: skip
        print(f"{f:9s} n={len(rr):4d} medOOS={np.median(o):6.2f} pos={np.mean(o > 0):.2f} | IS-best {bb['name']:42s} IS {bb['is']['sharpe']:5.2f} OOS {bb['oos']['sharpe']:5.2f} ({bb['oos']['cagr_pct']:6.1f}%)")
    sig = [c for c in tests["cells"] if c["significant"]]
    fw = [c for c in tests["cells"] if c["p_fwer_perm"] < 0.05]
    out = {
        "family": "seasonality",
        "protocol": {"IS": IS, "OOS": OOS, "selection": "max IS Sharpe after costs among all configs with >=30 IS trades",
                     "cost_model": "taker: cost_per_side(ADV до t-1) = 0.05% + проскальзывание; maker: 0.02%, лимит по close[t-1], исполнение только если цена прошла сквозь лимит на 1 bp, иначе рыночная по close[t]",
                     "funding": "по позиции на момент расчёта; при смене позиции на границе — худший вариант",
                     "exposure": "1x номинала капитала в позиции (без плеча), вне позиции — 0"},  # fmt: skip
        "tests_summary": {"n_cells": tests["n_cells"], "holm_significant": len(sig), "perm_fwer_lt_0.05_within_family": [
            {k: c[k] for k in ("asset", "family", "cell", "diff_bp", "t", "p_fwer_perm", "p_holm_global", "is_years_same_sign")} | {"oos_diff_bp": c["oos"].get("diff_bp"), "oos_t": c["oos"].get("t")} for c in fw]},
        "cells": tests["cells"],
        "cme": load_json("part_cme.json"),
        "profiles": load_json("part_profiles.json"),
        "presample_spot_2017_2019": load_json("part_presample.json"),
        "singles": load_json("part_singles.json")["summary"],
        "configs": rows,
        "random_hour_baselines": strats["random_hours"],
        "all_configs_oos_distribution": dist,
        "family_summary": fam_sum,
        "robust": load_json("part_robust.json"),
        "volatility_seasonality": load_json("part_vol.json") if (ODIR / "part_vol.json").exists() else None,
        "best": {"name": best["name"], "meta": best["meta"], "stats": st, "capacity_1pct_hourly_volume": cap_d,
                 "pnl_decomposition": fund_share, "daily_file": "output/edge_lab/seasonality/best_daily.npz"},
    }  # fmt: skip
    save_json(out, "results.json")
    print("best:", best["name"], "IS", st["is"], "\nOOS", st["oos"], "\nper_year", st["per_year"], "\nhalfyears", st["oos_halfyears"])
    print("capacity", cap_d, "\npnl", fund_share, "\ndist", dist)
    print("daily file", len(dts), dts[0], dts[-1], "sum check", float(np.prod(1 + dret)), float(np.prod(1 + res["net"])))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tests"
    {"tests": run_tests, "cme": run_cme, "strats": run_strats, "presample": run_presample, "singles": run_singles, "robust": run_robust, "profiles": run_profiles, "vol": run_vol, "report": run_report}[cmd]()
