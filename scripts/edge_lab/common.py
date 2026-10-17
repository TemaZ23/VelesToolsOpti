"""
Общие данные, издержки и метрики для поиска рыночных неэффективностей (scripts/edge_lab).

Данные — scripts/edge_lab/universe.py: все USDT-M бессрочные фьючерсы Binance, включая
делистинговые (без «ошибки выжившего»), часовые свечи и фандинг.

    from scripts.edge_lab.common import panel, metrics, IS, OOS, cost_per_side

    p = panel("1d")            # dict: ts (открытие бара, мс), symbols, open/high/low/close/qvol/taker_buy_qvol
                               # [T, N] (NaN — монета не торговалась), funding [T, N] (сумма ставок за бар)
    eligible = liquid_mask(p, min_age_bars=30, min_adv_usd=5e6)  # только прошлые данные на момент t

Правила, общие для всех исследований:
- сигнал по закрытию бара t, исполнение по открытию бара t+1 (open[t+1]) или позже;
- в выборку на момент t входят только монеты, которые торговались до t (возраст, ликвидность —
  по прошлым барам), делистинговые монеты остаются в истории до последнего бара;
- позиция в монете, которую делистнули, закрывается по последней цене (close последнего бара);
- издержки: cost_per_side() — комиссия тейкера + проскальзывание по ликвидности; фандинг
  платится/получается по факту (+ ставка — лонг платит, шорт получает);
- подбор параметров только на IS, OOS не подсматривается.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "edge_lab"
OUT = ROOT / "output" / "edge_lab"
HOUR = 3_600_000
DAY = 86_400_000

IS = ("2020-01-01", "2024-01-01")
OOS = ("2024-01-01", "2026-10-01")

TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
FREQ_MS = {"1h": HOUR, "4h": 4 * HOUR, "1d": DAY}


def ms(date: str) -> int:
    return int(np.datetime64(date, "ms").astype(np.int64))


def universe() -> dict:
    return json.loads((CACHE / "universe.json").read_text())


def load_hourly(sym: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / "1h" / f"{sym}.npz") as z:
        return {k: z[k] for k in z.files}


def load_funding(sym: str) -> dict[str, np.ndarray]:
    path = CACHE / "funding" / f"{sym}.npz"
    if not path.exists():
        return {"ts": np.empty(0, np.int64), "rate": np.empty(0)}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _build_panel(freq: str) -> dict[str, np.ndarray]:
    step = FREQ_MS[freq]
    uni = universe()
    syms = sorted(uni)
    t0 = min(ms(v["first"]) for v in uni.values()) // step * step
    t1 = max(ms(v["last"]) for v in uni.values()) // step * step
    ts = np.arange(t0, t1 + step, step, dtype=np.int64)
    T, N = len(ts), len(syms)
    dt = np.float32 if freq == "1h" else np.float64  # часовая панель ~2 ГБ даже во float32
    nan = lambda: np.full((T, N), np.nan, dtype=dt)  # noqa: E731
    o, h, l, c, qv, tb = nan(), nan(), nan(), nan(), nan(), nan()
    fund, hours = np.zeros((T, N), dtype=dt), np.zeros((T, N), dtype=np.int8)
    for j, s in enumerate(syms):
        d = load_hourly(s)
        b = ((d["ts"] - t0) // step).astype(np.int64)
        bars, first = np.unique(b, return_index=True)
        last = np.r_[first[1:], len(b)] - 1
        o[bars, j] = d["ohlc"][first, 0]
        c[bars, j] = d["ohlc"][last, 3]
        h[bars, j] = np.maximum.reduceat(d["ohlc"][:, 1], first)
        l[bars, j] = np.minimum.reduceat(d["ohlc"][:, 2], first)
        qv[bars, j] = np.add.reduceat(d["qvol"], first)
        tb[bars, j] = np.add.reduceat(d["taker_buy_qvol"], first)
        hours[bars, j] = np.diff(np.r_[first, len(b)])
        f = load_funding(s)
        if len(f["ts"]):
            fb = (f["ts"] - t0) // step
            ok = (fb >= 0) & (fb < T)
            np.add.at(fund[:, j], fb[ok].astype(np.int64), f["rate"][ok])
    return {"ts": ts, "symbols": np.array(syms), "open": o, "high": h, "low": l, "close": c, "qvol": qv,
            "taker_buy_qvol": tb, "funding": fund, "hours": hours}  # fmt: skip


def panel(freq: str = "1d") -> dict[str, np.ndarray]:
    """Выровненная панель [T, N] по всем символам; кешируется в cache/edge_lab/panel_<freq>.npz.
    "1h" — во float32 (≈2 ГБ в памяти); для часовых задач по отдельным монетам дешевле load_hourly()."""
    path = CACHE / f"panel_{freq}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    p = _build_panel(freq)
    np.savez(path, **p)
    return p


def rolling_mean_past(x: np.ndarray, n: int) -> np.ndarray:
    """Среднее за n прошлых баров включая текущий (NaN → 0 в сумме, делитель — число валидных)."""
    v = np.nan_to_num(x)
    cnt = (~np.isnan(x)).astype(float)
    cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), v]), axis=0)
    cc = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), cnt]), axis=0)
    s = cs[n:] - cs[:-n]
    k = cc[n:] - cc[:-n]
    out = np.full(x.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1 :] = np.where(k > 0, s / k, np.nan)
    return out


def liquid_mask(p: dict, min_age_bars: int = 30, min_adv_usd: float = 5e6, adv_bars: int | None = None) -> np.ndarray:
    """Монета допустима на баре t: торгуется на t, история ≥ min_age_bars, средний объём за прошлые
    бары (в пересчёте на сутки) ≥ min_adv_usd. Использует только данные до t включительно."""
    step = int(p["ts"][1] - p["ts"][0])
    per_day = DAY / step
    if adv_bars is None:
        adv_bars = int(max(1, round(30 * per_day)))
    alive = ~np.isnan(p["close"])
    age = np.cumsum(alive, axis=0)
    adv = rolling_mean_past(p["qvol"], adv_bars) * per_day
    return alive & (age >= min_age_bars) & (np.nan_to_num(adv) >= min_adv_usd)


def adv_usd(p: dict, bars: int | None = None) -> np.ndarray:
    step = int(p["ts"][1] - p["ts"][0])
    per_day = DAY / step
    bars = bars or int(max(1, round(30 * per_day)))
    return rolling_mean_past(p["qvol"], bars) * per_day


def cost_per_side(adv: np.ndarray | float, maker: bool = False) -> np.ndarray | float:
    """Комиссия + проскальзывание на одну сторону (доля номинала) по среднесуточному объёму в USDT.
    Для небольшого счёта (сделка ≪ 1% часового объёма)."""
    a = np.asarray(adv, dtype=float)
    slip = np.where(a >= 2e8, 0.0002, np.where(a >= 5e7, 0.0005, np.where(a >= 1e7, 0.001, np.where(a >= 2e6, 0.002, 0.004))))
    fee = MAKER_FEE if maker else TAKER_FEE
    out = fee + (0.0 if maker else slip)
    return float(out) if np.ndim(out) == 0 else out


def window_idx(ts: np.ndarray, window: tuple[str, str]) -> tuple[int, int]:
    return int(np.searchsorted(ts, ms(window[0]))), int(np.searchsorted(ts, ms(window[1])))


def metrics(ts: np.ndarray, ret: np.ndarray, trades: int | None = None) -> dict:
    """Метрики по ряду доходностей капитала за бар (доли, с учётом издержек). ts — открытие баров."""
    ret = np.nan_to_num(np.asarray(ret, dtype=float))
    if len(ret) == 0:
        return {}
    eq = np.cumprod(1 + np.maximum(ret, -1))
    step = int(ts[1] - ts[0]) if len(ts) > 1 else DAY
    years = len(ret) * step / DAY / 365.25
    days = (ts // DAY).astype(np.int64)
    dlast = np.r_[days[1:] != days[:-1], True]
    deq = np.r_[1.0, eq[dlast]]
    dret = deq[1:] / np.where(deq[:-1] > 0, deq[:-1], np.nan) - 1
    dret = np.nan_to_num(dret)
    months = ts.astype("datetime64[ms]").astype("datetime64[M]")
    mlast = np.r_[months[1:] != months[:-1], True]
    meq = np.r_[1.0, eq[mlast]]
    mret = meq[1:] / np.where(meq[:-1] > 0, meq[:-1], np.nan) - 1
    mret = np.nan_to_num(mret, nan=-1.0)
    final = float(eq[-1])
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    out = {
        "final_x": round(final, 4),
        "cagr_pct": round(((final ** (1 / years) - 1) * 100) if final > 0 and years > 0 else -100.0, 2),
        "avg_month_pct": round(float(np.mean(mret)) * 100, 2),
        "median_month_pct": round(float(np.median(mret)) * 100, 2),
        "sharpe": round(float(np.mean(dret) / np.std(dret) * np.sqrt(365)) if np.std(dret) > 0 else 0.0, 2),
        "max_dd_pct": round(float(np.max(1 - eq / peak)) * 100, 2),
        "worst_month_pct": round(float(np.min(mret)) * 100, 2),
        "pos_months_pct": round(float(np.mean(mret > 0)) * 100, 1),
        "months": int(len(mret)),
    }
    if trades is not None:
        out["trades"] = int(trades)
    return out


def per_year(ts: np.ndarray, ret: np.ndarray) -> dict[str, float]:
    years = ts.astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    out = {}
    for y in np.unique(years):
        r = np.nan_to_num(ret[years == y])
        out[str(y)] = round((float(np.prod(1 + np.maximum(r, -1))) - 1) * 100, 1)
    return out
