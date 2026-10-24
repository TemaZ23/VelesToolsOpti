"""
Стратегии для OG/USDT (OG Fan Token) на движке DCA Lab — только то, что собирается из настроек
обычного DCA-бота Veles (вход по индикаторам, сетка, тейк, стоп, направление).

Наблюдения (scripts/og_lab/analyze.py): цена возвращается к среднему на всех ТФ (отрицательная
автокорреляция, variance ratio < 1), резкие часовые пампы в большинстве случаев сдуваются
(после +10% за час через сутки медианно −8%), а после резких падений отскока нет.

Задачи для эволюционного поиска (scripts.dca_lab.optimize.evolve):
  og_fade — SHORT «гасим памп»: вход после разгона (RSI/CCI 1h высоко, пробой полос), усреднения
            вверх, тейк фиксированный или по сигналу, стоп по выбору, фильтр тренда 4h.
  og_long — LONG «возврат к среднему»: зеркально, вход после перепроданности.
"""

from __future__ import annotations

from scripts.dca_lab.optimize import P, Problem, grid_from_params
from scripts.dca_lab.strategy import (
    SL_FIXED,
    SL_NONE,
    TP_FIXED,
    TP_SIGNAL,
    BotConfig,
    Condition,
    Order,
    PortfolioConfig,
    Sleeve,
)

# Размер одного бота на одной монете: 10% депозита × плечо 10 = номинал сетки до 100% депозита.
# Эффективность (%/мес на 1% просадки) от размера не зависит, размер выбирается потом.
BOT_SHARE = 10.0
LEVERAGE = 10.0

SERIES = [
    "RSI@1h", "CCI@1h", "ADX@1h",
    "KCU@30m", "KCL@30m",
    "BBU@5m", "BBL@5m", "ATRP@5m",
    "RSI_LEVELS_DN@1m", "RSI_LEVELS@1m", "CCI_LEVELS_DN@1m", "CCI_LEVELS@1m",
    "BBU@1m", "BBL@1m",
    "RSI@4h", "MACD@4h",
]  # fmt: skip

GRID = [
    P("n_orders", 1, 8, "int"),
    P("g_first", 1.0, 12.0),  # отступ 2-го ордера, %
    P("g_depth", 4.0, 70.0),  # отступ последнего ордера, %
    P("g_pow", 0.6, 2.2),
    P("g_base_indent", 0.0, 1.5),
    P("g_mart", 1.0, 2.0),
    P("pull_up", 0.2, 3.0),
]

COMMON = [
    P("e_rsi1h", 50, 95),  # SHORT: RSI@1h > X; LONG: RSI@1h < 100 − X
    P("e_cci1h", 0, 350),  # SHORT: CCI@1h > X; LONG: CCI@1h < −X
    P("e_band", 0, 2, "cat", (0, 1, 2)),  # 0 — нет, 1 — Кельтнер 30m, 2 — Боллинджер 5m
    P("e_atrp5", 0.0, 3.0),  # ATRP@5m > X (всплеск волатильности), 0 — выкл.
    P("e_trigger", 0, 2, "cat", (0, 1, 2)),  # 0 — нет, 1 — RSI уровни 1m, 2 — CCI уровни 1m
    P("e_trend", 0, 2, "cat", (0, 1, 2)),  # 0 — нет, 1 — RSI@4h, 2 — MACD@4h
    P("e_trend_lvl", 30, 80),  # SHORT: RSI@4h < X; LONG: RSI@4h > 100 − X
    P("o_signal", 0, 1, "cat", (0, 1)),  # ордера сетки: 0 — лимитные, 1 — по сигналу уровней CCI 1m
    *GRID,
    P("tp_kind", 0, 1, "cat", (0, 1)),  # 0 — фиксированный, 1 — по сигналу полосы 1m при P&L ≥ мин.
    P("tp_pct", 0.8, 15.0),
    P("tp_min", 0.5, 8.0),
    P("sl_kind", 0, 1, "cat", (0, 1)),  # 0 — без стопа, 1 — фиксированный от средней
    P("sl_pct", 3.0, 80.0),
]

BASELINE = {
    "e_rsi1h": 78, "e_cci1h": 180, "e_band": 1, "e_atrp5": 0.0, "e_trigger": 1, "e_trend": 0, "e_trend_lvl": 60,
    "o_signal": 0, "n_orders": 4, "g_first": 4.0, "g_depth": 25.0, "g_pow": 1.0, "g_base_indent": 0.0,
    "g_mart": 1.4, "pull_up": 0.6, "tp_kind": 0, "tp_pct": 4.0, "tp_min": 2.0, "sl_kind": 0, "sl_pct": 40.0,
}  # fmt: skip


def _entry(p: dict, short: bool) -> list[Condition]:
    out: list[Condition] = []
    trig = int(p["e_trigger"])
    if trig == 1:
        out.append(Condition("RSI_LEVELS_DN@1m" if short else "RSI_LEVELS@1m", ">", 0.5))
    elif trig == 2:
        out.append(Condition("CCI_LEVELS_DN@1m" if short else "CCI_LEVELS@1m", ">", 0.5))
    rsi = round(p["e_rsi1h"], 1)
    cci = round(p["e_cci1h"], 1)
    out.append(Condition("RSI@1h", ">", rsi) if short else Condition("RSI@1h", "<", round(100 - rsi, 1)))
    if cci > 1:
        out.append(Condition("CCI@1h", ">", cci) if short else Condition("CCI@1h", "<", -cci))
    band = int(p["e_band"])
    if band == 1:
        out.append(Condition("KCU@30m", ">", 0) if short else Condition("KCL@30m", "<", 0))
    elif band == 2:
        out.append(Condition("BBU@5m", ">", 0) if short else Condition("BBL@5m", "<", 0))
    if p["e_atrp5"] > 0.05:
        out.append(Condition("ATRP@5m", ">", round(p["e_atrp5"], 2)))
    trend = int(p["e_trend"])
    lvl = round(p["e_trend_lvl"], 1)
    if trend == 1:
        out.append(Condition("RSI@4h", "<", lvl) if short else Condition("RSI@4h", ">", round(100 - lvl, 1)))
    elif trend == 2:
        # SHORT только когда 4h-импульс не бычий, LONG — когда не медвежий.
        out.append(Condition("MACD@4h", "<", 0) if short else Condition("MACD@4h", ">", 0))
    return out


def build_bot(p: dict, short: bool, name: str) -> BotConfig:
    indents, vols = grid_from_params(p)
    sig = [Condition("CCI_LEVELS_DN@1m" if short else "CCI_LEVELS@1m", ">", 0.5)] if p["o_signal"] >= 0.5 else []
    orders = [Order(indents[0], vols[0])] + [
        Order(indents[k], vols[k], list(sig)) for k in range(1, len(indents))
    ]
    tp_signal = int(p["tp_kind"]) == 1
    return BotConfig(
        name=name,
        entry=_entry(p, short),
        orders=orders,
        pull_up=round(p["pull_up"], 2),
        tp_mode=TP_SIGNAL if tp_signal else TP_FIXED,
        tp_min_pnl=round(p["tp_min"], 2),
        tp_pct=round(p["tp_pct"], 2),
        tp_conditions=[Condition("BBL@1m" if short else "BBU@1m", "<" if short else ">", 0)] if tp_signal else [],
        sl_mode=SL_FIXED if int(p["sl_kind"]) == 1 else SL_NONE,
        sl_pct=round(p["sl_pct"], 1),
        direction=-1 if short else 1,
    )


def _problem(name: str, short: bool, coin: str, share: float = BOT_SHARE, leverage: float = LEVERAGE) -> Problem:
    def build(p: dict):
        bot = build_bot(p, short, name.upper())
        return [Sleeve(bot, share, leverage, 1, 0, [coin])], PortfolioConfig(bot_share=share, leverage=leverage)

    return Problem(name, COMMON, dict(BASELINE), build)


def fade_problem(coin: str = "OGSPOT") -> Problem:
    return _problem("og_fade", True, coin)


def long_problem(coin: str = "OGSPOT") -> Problem:
    return _problem("og_long", False, coin)


def constant_short_bot() -> BotConfig:
    """Эталон: шорт «всегда в позиции» (вход без условий, без сетки, без тейка и стопа)."""
    return BotConfig(
        name="CONST_SHORT",
        entry=[],
        orders=[Order(0.0, 100.0)],
        pull_up=50.0,
        tp_mode=TP_FIXED,
        tp_pct=1e6,
        sl_mode=SL_NONE,
        direction=-1,
    )
