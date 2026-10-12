"""
Оптимизатор параметров ботов (только то, что можно выставить в боте Veles).

Идея: доходность и просадка портфеля почти линейно масштабируются размером бота
(депозит на бота × плечо), поэтому ищем максимальную ЭФФЕКТИВНОСТЬ — среднемесячную
доходность на 1% максимальной просадки (от депозита) — при эталонном размере, а затем
подбираем размер под целевую доходность / допустимую просадку.

Задача (Problem) = пространство параметров + базовая точка + сборщик портфеля.
Поиск: (μ + λ)-эволюция по нормированному вектору параметров, батчи считаются
параллельно (numba prange). Обучение — на in-sample окне, проверка — на out-of-sample.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .portfolio import Market, Result, run_batch
from .strategy import (
    SL_FIXED,
    SL_NONE,
    SL_SIGNAL,
    TP_FIXED,
    TP_SIGNAL,
    BotConfig,
    Condition,
    Order,
    PortfolioConfig,
    Sleeve,
)

# ═══════════════════════════════════════════════════════════════════════
# ПРОСТРАНСТВО ПАРАМЕТРОВ
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class P:
    name: str
    lo: float
    hi: float
    kind: str = "float"  # float | int | cat
    choices: tuple = ()


Builder = Callable[[dict], tuple[list[Sleeve], PortfolioConfig]]


@dataclass
class Problem:
    name: str
    space: list[P]
    baseline: dict
    build: Builder


VOL = Condition("VOL@1h", ">", 100_000)

# ── общие параметры сетки ──
GRID_SPACE = [
    P("n_orders", 5, 10, "int"),
    P("g_first", 1.0, 8.0),  # отступ 2-го ордера, %
    P("g_depth", 25.0, 85.0),  # отступ последнего ордера, %
    P("g_pow", 0.6, 2.2),  # >1 — ордера плотнее у цены входа, реже внизу
    P("g_base_indent", 0.0, 1.0),
    P("g_mart", 1.0, 2.0),  # мартингейл объёмов
    P("pull_up", 0.2, 2.0),
]


def grid_from_params(p: dict) -> tuple[list[float], list[float]]:
    n = int(round(p["n_orders"]))
    indents = [round(p["g_base_indent"], 2)]
    first = p["g_first"]
    depth = max(p["g_depth"], first + 1)
    pw = p["g_pow"]
    for k in range(1, n):
        x = (k - 1) / max(n - 2, 1)  # 0..1
        indents.append(round(first + (depth - first) * x**pw, 2))
    vols = [p["g_mart"] ** k for k in range(n)]
    s = sum(vols)
    vols = [round(v / s * 100, 2) for v in vols]
    vols[-1] = round(100 - sum(vols[:-1]), 2)
    return indents, vols


# ═══════════════════════════════════════════════════════════════════════
# LONG (шаблон LONG_STRATEGY 22.X)
# ═══════════════════════════════════════════════════════════════════════

LONG_SPACE: list[P] = [
    P("e_cci1h", -50, 250),
    P("e_cmo30", -100, -20),
    P("e_wr30", -100, -50),
    P("e_atrp5", 0.0, 1.2),
    P("e_kc30", 0, 1, "cat", (0, 1)),
    P("e_rsilev", 0, 1, "cat", (0, 1)),
    *GRID_SPACE,
    P("o_adx15", 15, 60),
    P("o_mfi5", 8, 40),
    P("o_cmo30", -100, -50),
    P("o_vol_on", 0, 1, "cat", (0, 1)),
    P("tp_kind", 0, 5, "cat", (0, 1, 2, 3, 4, 5)),
    P("tp_min", 0.3, 3.0),
    P("tp_pct", 0.5, 4.0),
    P("sl_kind", 0, 2, "cat", (0, 1, 2)),
    P("sl_indent", 0.0, 15.0),
    P("sl_pct", 10.0, 60.0),
    P("block", 8, 63, "int"),
    P("dyn_order", 0, 8, "cat", (0, 3, 4, 5, 6, 7, 8)),
]

# Параметры, при которых шаблон близок к авторской LONG_STRATEGY 22.X
LONG_BASELINE: dict = {
    "e_cci1h": 80,
    "e_cmo30": -80,
    "e_wr30": -80,
    "e_atrp5": 0.5,
    "e_kc30": 1,
    "e_rsilev": 1,
    "n_orders": 8,
    "g_first": 5.0,
    "g_depth": 65.0,
    "g_pow": 1.8,
    "g_base_indent": 0.3,
    "g_mart": 1.5,
    "pull_up": 0.4,
    "o_adx15": 40,
    "o_mfi5": 19,
    "o_cmo30": -90,
    "o_vol_on": 1,
    "tp_kind": 0,
    "tp_min": 0.5,
    "tp_pct": 1.5,
    "sl_kind": 0,
    "sl_indent": 0.0,
    "sl_pct": 30.0,
    "block": 60,
    "dyn_order": 5,
}
# совместимость со старыми скриптами
SPACE = LONG_SPACE
BASELINE_PARAMS = LONG_BASELINE

LONG_TP_KINDS = {
    0: [Condition("BBL@1m", "<", 0), Condition("TZU@5m", ">", 0)],  # автор: Turtle 5m + BB 1m (обратный)
    1: [Condition("TZU@5m", ">", 0)],  # только Turtle Zone 5m
    2: [Condition("BBU@1m", ">", 0)],  # BB 1m: закрытие выше верхней полосы
    3: [],  # фиксированный тейк
    4: [Condition("BBL@1m", "<", 0), Condition("TZU@15m", ">", 0)],  # Turtle 15m + BB 1m (обратный)
    5: [Condition("BBL@1m", "<", 0), Condition("TZU@1h", ">", 0)],  # Turtle 1h + BB 1m (обратный)
}


def long_order_conditions(k: int, p: dict) -> list[Condition]:
    """Сигналы k-го ордера (k ≥ 2) по шаблону автора с настраиваемыми порогами."""
    vol = [VOL] if p["o_vol_on"] >= 0.5 else []
    cmo = Condition("CMO@30m", ">", round(p["o_cmo30"], 1))
    mfi = Condition("MFI@5m", "<", round(p["o_mfi5"], 1))
    pattern = {
        2: [Condition("ADX@15m", ">", round(p["o_adx15"], 1)), cmo],
        3: [mfi],
        4: [Condition("STOCH_LEVELS@1m", ">", 0.5), Condition("BBL@5m", "<", 0), cmo],
        5: [mfi],
        6: [Condition("KCL@5m", "<", 0), cmo],
    }
    return pattern.get(k, [Condition("CCI_LEVELS@1m", ">", 0.5), cmo]) + vol


def build_long_bot(p: dict, name: str = "LONG_OPT") -> BotConfig:
    indents, vols = grid_from_params(p)
    orders = [Order(indents[0], vols[0])]
    for k in range(2, len(indents) + 1):
        orders.append(Order(indents[k - 1], vols[k - 1], long_order_conditions(k, p)))
    entry: list[Condition] = []
    if p["e_rsilev"] >= 0.5:
        entry.append(Condition("RSI_LEVELS@1m", ">", 0.5))
    entry += [
        Condition("CCI@1h", "<", round(p["e_cci1h"], 1)),
        Condition("CMO@30m", ">", round(p["e_cmo30"], 1)),
        Condition("WR@30m", "<", round(p["e_wr30"], 1)),
    ]
    if p["e_kc30"] >= 0.5:
        entry.append(Condition("KCL@30m", "<", 0))
    if p["e_atrp5"] > 0.01:
        entry.append(Condition("ATRP@5m", ">", round(p["e_atrp5"], 2)))
    entry.append(VOL)
    tp_kind = int(p["tp_kind"])
    sl_kind = int(p["sl_kind"])
    return BotConfig(
        name=name,
        entry=entry,
        orders=orders,
        pull_up=round(p["pull_up"], 2),
        tp_mode=TP_FIXED if tp_kind == 3 else TP_SIGNAL,
        tp_min_pnl=round(p["tp_min"], 2),
        tp_pct=round(p["tp_pct"], 2),
        tp_conditions=list(LONG_TP_KINDS[tp_kind]),
        sl_mode={0: SL_SIGNAL, 1: SL_NONE, 2: SL_FIXED}[sl_kind],
        sl_indent=round(p["sl_indent"], 2),
        sl_pct=round(p["sl_pct"], 1),
        sl_conditions=[Condition("MACD_ZERO_DOWN@4h", ">", 0.5)] if sl_kind == 0 else [],
        direction=1,
    )


def build_from_params(p: dict, deposit: float = 1000.0, bot_share: float = 0.75, leverage: float = 20.0):
    """LONG-бот + портфель (совместимость со старыми скриптами)."""
    bot = build_long_bot(p)
    pf = PortfolioConfig(
        deposit=deposit,
        bot_share=bot_share,
        leverage=leverage,
        block=int(round(p["block"])),
        dyn_block_order=int(p["dyn_order"]),
    )
    return bot, pf


def long_sleeve(p: dict, share: float = 0.75, coins: list[str] | None = None) -> Sleeve:
    return Sleeve(build_long_bot(p), share, 20.0, int(round(p["block"])), int(p["dyn_order"]), coins)


def long_problem(share: float = 0.75, coins: list[str] | None = None) -> Problem:
    return Problem("long", LONG_SPACE, LONG_BASELINE, lambda p: ([long_sleeve(p, share, coins)], PortfolioConfig()))


# ═══════════════════════════════════════════════════════════════════════
# SHORT (зеркало шаблона)
# ═══════════════════════════════════════════════════════════════════════

SHORT_SPACE: list[P] = [
    P("e_cci1h", -250, 50),  # CCI@1h > X
    P("e_cmo30", 20, 100),  # CMO@30m < X
    P("e_wr30", -50, 0),  # WR@30m > X
    P("e_atrp5", 0.0, 1.2),
    P("e_kc30", 0, 1, "cat", (0, 1)),
    P("e_rsilev", 0, 1, "cat", (0, 1)),
    *GRID_SPACE,
    P("o_adx15", 15, 60),
    P("o_mfi5", 60, 92),  # MFI@5m > X
    P("o_cmo30", 50, 100),  # CMO@30m < X
    P("o_vol_on", 0, 1, "cat", (0, 1)),
    P("tp_kind", 0, 4, "cat", (0, 1, 2, 3, 4)),
    P("tp_min", 0.3, 3.0),
    P("tp_pct", 0.5, 4.0),
    P("sl_kind", 0, 2, "cat", (0, 1, 2)),
    P("sl_indent", 0.0, 15.0),
    P("sl_pct", 10.0, 60.0),
    P("block", 8, 63, "int"),
    P("dyn_order", 0, 8, "cat", (0, 3, 4, 5, 6, 7, 8)),
]

SHORT_BASELINE: dict = dict(LONG_BASELINE)
SHORT_BASELINE.update({"e_cci1h": -80, "e_cmo30": 80, "e_wr30": -20, "o_mfi5": 81, "o_cmo30": 90})

SHORT_TP_KINDS = {
    0: [Condition("BBU@1m", ">", 0), Condition("TZL@5m", "<", 0)],
    1: [Condition("TZL@5m", "<", 0)],
    2: [Condition("BBL@1m", "<", 0)],
    3: [],
    4: [Condition("BBU@1m", ">", 0), Condition("TZL@15m", "<", 0)],
}


def short_order_conditions(k: int, p: dict) -> list[Condition]:
    vol = [VOL] if p["o_vol_on"] >= 0.5 else []
    cmo = Condition("CMO@30m", "<", round(p["o_cmo30"], 1))
    mfi = Condition("MFI@5m", ">", round(p["o_mfi5"], 1))
    pattern = {
        2: [Condition("ADX@15m", ">", round(p["o_adx15"], 1)), cmo],
        3: [mfi],
        4: [Condition("STOCH_LEVELS_DN@1m", ">", 0.5), Condition("BBU@5m", ">", 0), cmo],
        5: [mfi],
        6: [Condition("KCU@5m", ">", 0), cmo],
    }
    return pattern.get(k, [Condition("CCI_LEVELS_DN@1m", ">", 0.5), cmo]) + vol


def build_short_bot(p: dict, name: str = "SHORT_OPT") -> BotConfig:
    indents, vols = grid_from_params(p)
    orders = [Order(indents[0], vols[0])]
    for k in range(2, len(indents) + 1):
        orders.append(Order(indents[k - 1], vols[k - 1], short_order_conditions(k, p)))
    entry: list[Condition] = []
    if p["e_rsilev"] >= 0.5:
        entry.append(Condition("RSI_LEVELS_DN@1m", ">", 0.5))
    entry += [
        Condition("CCI@1h", ">", round(p["e_cci1h"], 1)),
        Condition("CMO@30m", "<", round(p["e_cmo30"], 1)),
        Condition("WR@30m", ">", round(p["e_wr30"], 1)),
    ]
    if p["e_kc30"] >= 0.5:
        entry.append(Condition("KCU@30m", ">", 0))
    if p["e_atrp5"] > 0.01:
        entry.append(Condition("ATRP@5m", ">", round(p["e_atrp5"], 2)))
    entry.append(VOL)
    tp_kind = int(p["tp_kind"])
    sl_kind = int(p["sl_kind"])
    return BotConfig(
        name=name,
        entry=entry,
        orders=orders,
        pull_up=round(p["pull_up"], 2),
        tp_mode=TP_FIXED if tp_kind == 3 else TP_SIGNAL,
        tp_min_pnl=round(p["tp_min"], 2),
        tp_pct=round(p["tp_pct"], 2),
        tp_conditions=list(SHORT_TP_KINDS[tp_kind]),
        sl_mode={0: SL_SIGNAL, 1: SL_NONE, 2: SL_FIXED}[sl_kind],
        sl_indent=round(p["sl_indent"], 2),
        sl_pct=round(p["sl_pct"], 1),
        sl_conditions=[Condition("MACD_ZERO_UP@4h", ">", 0.5)] if sl_kind == 0 else [],
        direction=-1,
    )


def short_sleeve(p: dict, share: float = 0.75, coins: list[str] | None = None) -> Sleeve:
    return Sleeve(build_short_bot(p), share, 20.0, int(round(p["block"])), int(p["dyn_order"]), coins)


def short_problem(share: float = 0.75, coins: list[str] | None = None) -> Problem:
    return Problem("short", SHORT_SPACE, SHORT_BASELINE, lambda p: ([short_sleeve(p, share, coins)], PortfolioConfig()))


# ═══════════════════════════════════════════════════════════════════════
# ЛОВЕЦ ПРОЛИВОВ (LONG, вход только внутри резкого падения)
# ═══════════════════════════════════════════════════════════════════════

CRASH_SPACE: list[P] = [
    P("c_entry", 0, 3, "cat", (0, 1, 2, 3)),
    P("c_rsi1", 3, 25),  # RSI@1m < X
    P("c_cci1", -500, -150),  # CCI@1m < X
    P("c_atrp5", 0.3, 3.0),  # ATRP@5m > X
    P("n_orders", 2, 7, "int"),
    P("g_first", 2.0, 15.0),
    P("g_depth", 15.0, 70.0),
    P("g_pow", 0.6, 2.2),
    P("g_mart", 1.0, 2.0),
    P("c_ord_sig", 0, 1, "cat", (0, 1)),  # 1 — ордера по сигналу CCI levels, 0 — лимитные
    P("tp_kind", 0, 2, "cat", (0, 1, 2)),  # 0 — Turtle5m+BB1m, 1 — фикс., 2 — Turtle5m
    P("tp_min", 0.3, 4.0),
    P("tp_pct", 1.0, 10.0),
    P("sl_kind", 0, 1, "cat", (0, 1)),
    P("block", 5, 63, "int"),
]

CRASH_BASELINE: dict = {
    "c_entry": 0,
    "c_rsi1": 10,
    "c_cci1": -300,
    "c_atrp5": 1.0,
    "n_orders": 5,
    "g_first": 5.0,
    "g_depth": 35.0,
    "g_pow": 1.0,
    "g_mart": 1.3,
    "c_ord_sig": 1,
    "tp_kind": 0,
    "tp_min": 1.0,
    "tp_pct": 3.0,
    "sl_kind": 0,
    "block": 63,
}


def build_crash_bot(p: dict, name: str = "CRASH_OPT") -> BotConfig:
    kind = int(p["c_entry"])
    rsi = Condition("RSI@1m", "<", round(p["c_rsi1"], 1))
    cci = Condition("CCI@1m", "<", round(p["c_cci1"], 0))
    entry = {
        0: [rsi, Condition("KCL4@5m", "<", 0)],
        1: [cci, Condition("BBL3@5m", "<", 0)],
        2: [rsi, Condition("BBL3@5m", "<", 0)],
        3: [Condition("KCL4@5m", "<", 0), Condition("ATRP@5m", ">", round(p["c_atrp5"], 2))],
    }[kind] + [VOL]
    q = dict(p)
    q["g_base_indent"] = 0.0
    indents, vols = grid_from_params(q)
    sig = [Condition("CCI_LEVELS@1m", ">", 0.5)] if p["c_ord_sig"] >= 0.5 else []
    orders = [Order(indents[0], vols[0])] + [Order(indents[k], vols[k], list(sig)) for k in range(1, len(indents))]
    tp_kind = int(p["tp_kind"])
    tp_conds = {0: [Condition("BBL@1m", "<", 0), Condition("TZU@5m", ">", 0)], 1: [], 2: [Condition("TZU@5m", ">", 0)]}
    return BotConfig(
        name=name,
        entry=entry,
        orders=orders,
        pull_up=0.5,
        tp_mode=TP_FIXED if tp_kind == 1 else TP_SIGNAL,
        tp_min_pnl=round(p["tp_min"], 2),
        tp_pct=round(p["tp_pct"], 2),
        tp_conditions=tp_conds[tp_kind],
        sl_mode=SL_SIGNAL if int(p["sl_kind"]) == 0 else SL_NONE,
        sl_conditions=[Condition("MACD_ZERO_DOWN@4h", ">", 0.5)] if int(p["sl_kind"]) == 0 else [],
        direction=1,
    )


def crash_sleeve(p: dict, share: float = 0.75, coins: list[str] | None = None) -> Sleeve:
    return Sleeve(build_crash_bot(p), share, 20.0, int(round(p["block"])), 0, coins)


def crash_problem(share: float = 0.75, coins: list[str] | None = None) -> Problem:
    return Problem("crash", CRASH_SPACE, CRASH_BASELINE, lambda p: ([crash_sleeve(p, share, coins)], PortfolioConfig()))


# ═══════════════════════════════════════════════════════════════════════
# СЕРИИ, НУЖНЫЕ ЗАДАЧАМ (загружаем только их — экономия памяти)
# ═══════════════════════════════════════════════════════════════════════

LONG_SERIES = [
    "RSI_LEVELS@1m", "STOCH_LEVELS@1m", "CCI_LEVELS@1m", "BBL@1m", "BBU@1m",
    "ATRP@5m", "MFI@5m", "BBL@5m", "KCL@5m", "TZU@5m",
    "ADX@15m", "TZU@15m",
    "CMO@30m", "WR@30m", "KCL@30m",
    "CCI@1h", "VOL@1h", "TZU@1h",
    "MACD_ZERO_DOWN@4h",
]  # fmt: skip
SHORT_SERIES = [
    "RSI_LEVELS_DN@1m", "STOCH_LEVELS_DN@1m", "CCI_LEVELS_DN@1m", "BBL@1m", "BBU@1m",
    "ATRP@5m", "MFI@5m", "BBU@5m", "KCU@5m", "TZL@5m",
    "ADX@15m", "TZL@15m",
    "CMO@30m", "WR@30m", "KCU@30m",
    "CCI@1h", "VOL@1h",
    "MACD_ZERO_UP@4h",
]  # fmt: skip
CRASH_SERIES = [
    "RSI@1m", "CCI@1m", "CCI_LEVELS@1m", "BBL@1m",
    "KCL4@5m", "BBL3@5m", "ATRP@5m", "TZU@5m",
    "VOL@1h", "MACD_ZERO_DOWN@4h",
]  # fmt: skip


def series_for(*names: str) -> list[str]:
    """Объединение серий задач: series_for('long', 'crash')."""
    lists = {"long": LONG_SERIES, "short": SHORT_SERIES, "crash": CRASH_SERIES}
    out: list[str] = []
    for n in names:
        for s in lists[n]:
            if s not in out:
                out.append(s)
    return out


# ═══════════════════════════════════════════════════════════════════════
# КОДИРОВАНИЕ / МУТАЦИИ
# ═══════════════════════════════════════════════════════════════════════


def encode(p: dict, space: list[P] = LONG_SPACE) -> np.ndarray:
    x = np.zeros(len(space))
    for i, s in enumerate(space):
        v = p[s.name]
        if s.kind == "cat":
            x[i] = s.choices.index(int(v)) / max(len(s.choices) - 1, 1)
        else:
            x[i] = (v - s.lo) / (s.hi - s.lo)
    return np.clip(x, 0, 1)


def decode(x: np.ndarray, space: list[P] = LONG_SPACE) -> dict:
    p: dict = {}
    for i, s in enumerate(space):
        v = float(np.clip(x[i], 0, 1))
        if s.kind == "cat":
            p[s.name] = s.choices[int(round(v * (len(s.choices) - 1)))]
        elif s.kind == "int":
            p[s.name] = int(round(s.lo + v * (s.hi - s.lo)))
        else:
            p[s.name] = s.lo + v * (s.hi - s.lo)
    return p


def mutate(x, rng, sigma: float, rate: float, frozen: set[str], space: list[P] = LONG_SPACE) -> np.ndarray:
    y = x.copy()
    for i, s in enumerate(space):
        if s.name in frozen:
            continue
        if rng.random() < rate:
            if s.kind == "cat":
                y[i] = rng.integers(0, len(s.choices)) / max(len(s.choices) - 1, 1)
            else:
                y[i] = y[i] + rng.normal(0, sigma)
    return np.clip(y, 0, 1)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(a)) < 0.5
    return np.where(mask, a, b)


# ═══════════════════════════════════════════════════════════════════════
# ЦЕЛЕВАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════


def fold_efficiency(res: Result, n_folds: int) -> list[float]:
    """Эффективность (%/мес на 1% просадки от депозита) на равных под-окнах по дневным данным."""
    eq = res.daily[:, 0]
    lows = res.daily[:, 1]
    if eq.size == 0:
        return [0.0] * n_folds
    edges = np.linspace(0, len(eq), n_folds + 1).astype(int)
    out = []
    start_eq = res.deposit
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        peak = start_eq
        dd = 0.0
        for d in range(a, b):
            dd = max(dd, peak - lows[d])
            peak = max(peak, eq[d])
        months = (b - a) / 30.4375
        ret = (eq[b - 1] - start_eq) / res.deposit * 100 / months
        out.append(ret / max(dd / res.deposit * 100, 2.0))
        start_eq = eq[b - 1]
    return out


def score(res: Result, n_folds: int = 3, min_deals: int = 1000) -> float:
    """
    Устойчивая эффективность: половина — по всему окну, половина — худший фолд.
    Ликвидация — жёсткий штраф; мало сделок — штраф (защита от подгонки под редкие события).
    """
    if res.liquidated:
        return -10.0
    full = res.efficiency
    folds = fold_efficiency(res, n_folds)
    s = 0.5 * full + 0.5 * min(folds)
    deals = res.stats[6]
    if deals < min_deals:
        s *= deals / min_deals
    return float(s)


# ═══════════════════════════════════════════════════════════════════════
# ЭВОЛЮЦИЯ
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Candidate:
    x: np.ndarray
    params: dict
    score: float
    summary: dict


def evaluate(
    market: Market,
    problem: Problem,
    xs: list[np.ndarray],
    t_from: int,
    t_to: int,
    batch: int = 8,
    min_deals: int = 1000,
) -> list[Candidate]:
    out: list[Candidate] = []
    for s in range(0, len(xs), batch):
        chunk = xs[s : s + batch]
        params = [decode(x, problem.space) for x in chunk]
        results = run_batch(market, [problem.build(p) for p in params], t_from, t_to)
        for x, p, r in zip(chunk, params, results):
            out.append(Candidate(x, p, score(r, min_deals=min_deals), r.summary()))
    return out


def evolve(
    market: Market,
    problem: Problem,
    t_from: int,
    t_to: int,
    generations: int = 30,
    pop: int = 24,
    children: int = 24,
    seed: int = 7,
    frozen: set[str] | None = None,
    start: list[dict] | None = None,
    log_path: str | None = None,
    time_budget_s: float | None = None,
    min_deals: int = 1000,
) -> list[Candidate]:
    space = problem.space
    names = [s.name for s in space]
    rng = np.random.default_rng(seed)
    frozen = frozen or set()
    base = encode(problem.baseline, space)
    init = [base] + [encode(p, space) for p in (start or [])]
    while len(init) < pop:
        if rng.random() < 0.5:
            x = mutate(base, rng, 0.25, 0.5, frozen, space)
        else:
            x = rng.random(len(space))
            for name in frozen:
                x[names.index(name)] = base[names.index(name)]
        init.append(x)
    t0 = time.time()
    population = evaluate(market, problem, init, t_from, t_to, min_deals=min_deals)
    population.sort(key=lambda c: -c.score)
    history = []
    for gen in range(generations):
        sigma = 0.15 * (1 - gen / max(generations, 1)) + 0.03
        kids = []
        for _ in range(children):
            a = population[min(int(rng.exponential(pop / 4)), len(population) - 1)]
            if rng.random() < 0.4:
                b = population[rng.integers(0, min(pop, len(population)))]
                x = mutate(crossover(a.x, b.x, rng), rng, sigma, 0.2, frozen, space)
            else:
                x = mutate(a.x, rng, sigma, 0.3, frozen, space)
            kids.append(x)
        evaluated = evaluate(market, problem, kids, t_from, t_to, min_deals=min_deals)
        population = sorted(population + evaluated, key=lambda c: -c.score)[:pop]
        best = population[0]
        line = {"gen": gen, "elapsed_s": round(time.time() - t0), "best_score": round(best.score, 4), "best": best.summary}
        history.append(line)
        print(
            f"[{problem.name}] gen {gen:3d} | {line['elapsed_s']:5d}s | score {best.score:.4f} | "
            f"{best.summary['avg_month_pct']:.2f}%/мес | DDabs {best.summary['dd_abs_pct']:.1f}% | "
            f"eff {best.summary['eff']:.3f} | сделок {best.summary['deals']}",
            flush=True,
        )
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "problem": problem.name,
                        "history": history,
                        "top": [{"score": c.score, "params": c.params, "summary": c.summary} for c in population[:12]],
                    },
                    f,
                    ensure_ascii=False,
                    indent=1,
                    default=float,
                )
        if time_budget_s and time.time() - t0 > time_budget_s:
            break
    return population
