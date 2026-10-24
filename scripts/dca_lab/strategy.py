"""
Описание DCA-бота Veles (режим «Сигнал») и пресеты LONG_STRATEGY.

Condition — одно условие: серия из меню indicators.SERIES_MENU, операция и порог.
Для сигнальных индикаторов (RSI_LEVELS, CCI_LEVELS, ...) используйте op='>' и value=0.5.
Для канальных (BBL/KCL — «закрытие ниже нижней полосы») — op='<' и value=0.
Для Turtle Zone в тейке (TZU — «закрытие выше верхней зоны») — op='>' и value=0.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .indicators import SERIES_INDEX

MAX_ORDERS = 12
MAX_CONDS = 8
# Группы условий в движке
G_ENTRY = 0
G_TP = MAX_ORDERS  # 12
G_SL = MAX_ORDERS + 1  # 13
N_GROUPS = MAX_ORDERS + 2

TP_SIGNAL = 0  # закрытие по сигналу индикатора при P&L >= min_pnl
TP_FIXED = 1  # лимитный тейк, % от средней
TP_FIXED_OR_SIGNAL = 2

SL_NONE = 0
SL_SIGNAL = 1  # стоп по сигналу, если цена ниже средней на sl_indent %
SL_FIXED = 2  # стоп по цене, % от средней


@dataclass
class Condition:
    series: str
    op: str  # '<' | '>'
    value: float

    def label(self) -> str:
        return f"{self.series} {self.op} {self.value:g}"


@dataclass
class Order:
    indent: float  # % от цены входа (для 1-го ордера — от цены сигнала)
    volume: float  # % от депозита бота × плечо
    conditions: list[Condition] = field(default_factory=list)


@dataclass
class BotConfig:
    name: str
    entry: list[Condition]
    orders: list[Order]
    pull_up: float = 0.4  # %
    tp_mode: int = TP_SIGNAL
    tp_min_pnl: float = 0.5  # %
    tp_pct: float = 1.0  # % (для TP_FIXED)
    tp_conditions: list[Condition] = field(default_factory=list)
    sl_mode: int = SL_SIGNAL
    sl_indent: float = 0.0  # % ниже средней, после которого сигнал стопа активен
    sl_pct: float = 30.0  # % (для SL_FIXED)
    sl_conditions: list[Condition] = field(default_factory=list)
    direction: int = 1  # +1 LONG, -1 SHORT

    def clone(self, **changes) -> "BotConfig":
        new = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(new, k, v)
        return new

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BotConfig":
        d = copy.deepcopy(d)
        d["entry"] = [Condition(**c) for c in d["entry"]]
        d["orders"] = [Order(o["indent"], o["volume"], [Condition(**c) for c in o["conditions"]]) for o in d["orders"]]
        d["tp_conditions"] = [Condition(**c) for c in d["tp_conditions"]]
        d["sl_conditions"] = [Condition(**c) for c in d["sl_conditions"]]
        return BotConfig(**d)

    def describe(self) -> str:
        lines = [f"Бот {self.name} ({'LONG' if self.direction > 0 else 'SHORT'})", "Вход:"]
        lines += [f"  - {c.label()}" for c in self.entry]
        lines.append(f"Подтяжка сетки: {self.pull_up}%")
        lines.append(f"Сетка (отступ {'вниз' if self.direction > 0 else 'вверх'} от цены входа / объём / сигналы):")
        for i, o in enumerate(self.orders, 1):
            conds = "; ".join(c.label() for c in o.conditions) or "лимитный"
            lines.append(f"  {i}. {o.indent:g}% / {o.volume:g}% — {conds}")
        tp = {TP_SIGNAL: "сигнал", TP_FIXED: "фикс.", TP_FIXED_OR_SIGNAL: "фикс. или сигнал"}[self.tp_mode]
        lines.append(f"Тейк: {tp}, мин. P&L {self.tp_min_pnl}%, фикс {self.tp_pct}%")
        lines += [f"  - {c.label()}" for c in self.tp_conditions]
        sl = {SL_NONE: "нет", SL_SIGNAL: f"сигнал (от средней -{self.sl_indent}%)", SL_FIXED: f"{self.sl_pct}% от средней"}[
            self.sl_mode
        ]
        lines.append(f"Стоп: {sl}")
        lines += [f"  - {c.label()}" for c in self.sl_conditions]
        return "\n".join(lines)


@dataclass
class PortfolioConfig:
    deposit: float = 1000.0  # общий депозит, USDT
    bot_share: float = 0.75  # % депозита на бота
    leverage: float = 20.0
    block: int = 60  # блокировка «по ботам»
    dyn_block_order: int = 5  # при наборе N-го ордера блокировка -1 (0 — выкл.)
    # ↑ размер/блокировка используются только для одиночного бота (см. sleeve_from_portfolio)
    maker_fee: float = 0.02  # %
    taker_fee: float = 0.055  # %
    funding_8h: float = 0.01  # % за 8 часов (лонг платит)
    mmr: float = 1.0  # % maintenance margin от номинала (кросс)
    compound: bool = False  # депозит на бота = доля от текущего реализованного эквити при входе в сделку

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# ПРЕСЕТ: LONG_STRATEGY 22.X (как в статье автора)
# ═══════════════════════════════════════════════════════════════════════

VOL_FILTER = Condition("VOL@1h", ">", 100_000)
CMO90 = Condition("CMO@30m", ">", -90)


def long_strategy_22x() -> BotConfig:
    return BotConfig(
        name="LONG_STRATEGY 22.X",
        entry=[
            Condition("RSI_LEVELS@1m", ">", 0.5),  # самый редкий — первым
            Condition("CCI@1h", "<", 80),
            Condition("CMO@30m", ">", -80),
            Condition("WR@30m", "<", -80),
            Condition("KCL@30m", "<", 0),
            Condition("ATRP@5m", ">", 0.5),
            VOL_FILTER,
        ],
        orders=[
            Order(0.3, 2),
            Order(5, 3, [Condition("ADX@15m", ">", 40), CMO90, VOL_FILTER]),
            Order(10, 4, [Condition("MFI@5m", "<", 19), VOL_FILTER]),
            Order(15, 6, [Condition("STOCH_LEVELS@1m", ">", 0.5), Condition("BBL@5m", "<", 0), CMO90, VOL_FILTER]),
            Order(20, 10, [Condition("MFI@5m", "<", 19), VOL_FILTER]),
            Order(30, 16, [Condition("KCL@5m", "<", 0), CMO90, VOL_FILTER]),
            Order(40, 25, [Condition("CCI_LEVELS@1m", ">", 0.5), CMO90, VOL_FILTER]),
            Order(65, 34, [Condition("CCI_LEVELS@1m", ">", 0.5), CMO90, VOL_FILTER]),
        ],
        pull_up=0.4,
        tp_mode=TP_SIGNAL,
        tp_min_pnl=0.5,
        tp_conditions=[Condition("BBL@1m", "<", 0), Condition("TZU@5m", ">", 0)],
        sl_mode=SL_SIGNAL,
        sl_indent=0.0,
        sl_conditions=[Condition("MACD_ZERO_DOWN@4h", ">", 0.5)],
    )


def long_strategy_22x_portfolio() -> PortfolioConfig:
    return PortfolioConfig(bot_share=0.75, leverage=20, block=60, dyn_block_order=5)


def short_mirror_22x() -> BotConfig:
    """Зеркальная SHORT-версия 22.X: вход на перекупленности, усреднение вверх, тейк на откате вниз."""
    cmo = Condition("CMO@30m", "<", 90)
    return BotConfig(
        name="SHORT_MIRROR 22.X",
        direction=-1,
        entry=[
            Condition("RSI_LEVELS_DN@1m", ">", 0.5),
            Condition("CCI@1h", ">", -80),
            Condition("CMO@30m", "<", 80),
            Condition("WR@30m", ">", -20),
            Condition("KCU@30m", ">", 0),
            Condition("ATRP@5m", ">", 0.5),
            VOL_FILTER,
        ],
        orders=[
            Order(0.3, 2),
            Order(5, 3, [Condition("ADX@15m", ">", 40), cmo, VOL_FILTER]),
            Order(10, 4, [Condition("MFI@5m", ">", 81), VOL_FILTER]),
            Order(15, 6, [Condition("STOCH_LEVELS_DN@1m", ">", 0.5), Condition("BBU@5m", ">", 0), cmo, VOL_FILTER]),
            Order(20, 10, [Condition("MFI@5m", ">", 81), VOL_FILTER]),
            Order(30, 16, [Condition("KCU@5m", ">", 0), cmo, VOL_FILTER]),
            Order(40, 25, [Condition("CCI_LEVELS_DN@1m", ">", 0.5), cmo, VOL_FILTER]),
            Order(65, 34, [Condition("CCI_LEVELS_DN@1m", ">", 0.5), cmo, VOL_FILTER]),
        ],
        pull_up=0.4,
        tp_mode=TP_SIGNAL,
        tp_min_pnl=0.5,
        tp_conditions=[Condition("BBU@1m", ">", 0), Condition("TZL@5m", "<", 0)],
        sl_mode=SL_SIGNAL,
        sl_indent=0.0,
        sl_conditions=[Condition("MACD_ZERO_UP@4h", ">", 0.5)],
    )


@dataclass
class Sleeve:
    """Набор одинаковых ботов (одна монета — один бот) со своим размером и блокировкой."""

    bot: BotConfig
    bot_share: float = 0.75  # % депозита на бота
    leverage: float = 20.0
    block: int = 60
    dyn_block_order: int = 5
    coins: list[str] | None = None  # None — все монеты рынка

    def to_dict(self) -> dict:
        return {
            "bot": self.bot.to_dict(),
            "bot_share": self.bot_share,
            "leverage": self.leverage,
            "block": self.block,
            "dyn_block_order": self.dyn_block_order,
            "coins": self.coins,
        }

    @staticmethod
    def from_dict(d: dict) -> "Sleeve":
        return Sleeve(
            BotConfig.from_dict(d["bot"]), d["bot_share"], d["leverage"], d["block"], d["dyn_block_order"], d["coins"]
        )


def sleeve_from_portfolio(bot: BotConfig, pf: PortfolioConfig, coins: list[str] | None = None) -> Sleeve:
    return Sleeve(bot, pf.bot_share, pf.leverage, pf.block, pf.dyn_block_order, coins)


# ═══════════════════════════════════════════════════════════════════════
# УПАКОВКА В МАССИВЫ ДЛЯ ДВИЖКА
# ═══════════════════════════════════════════════════════════════════════

OPS = {"<": 0, ">": 1}


def _fill_group(conds, g, ser, op, thr, index: dict[str, int]) -> None:
    if len(conds) > MAX_CONDS:
        raise ValueError(f"слишком много условий в группе {g}")
    for m, c in enumerate(conds):
        if c.series not in index:
            raise KeyError(f"серия {c.series} не загружена в Market")
        ser[g, m] = index[c.series]
        op[g, m] = OPS[c.op]
        thr[g, m] = c.value


def pack_sleeve(sl: Sleeve, index: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """Sleeve → массивы одной стратегии для engine.simulate."""
    index = index or SERIES_INDEX
    bot = sl.bot
    if len(bot.orders) > MAX_ORDERS:
        raise ValueError("слишком много ордеров")
    cser = np.full((N_GROUPS, MAX_CONDS), -1, dtype=np.int32)
    cop = np.zeros((N_GROUPS, MAX_CONDS), dtype=np.int8)
    cthr = np.zeros((N_GROUPS, MAX_CONDS), dtype=np.float32)
    _fill_group(bot.entry, G_ENTRY, cser, cop, cthr, index)
    for i, o in enumerate(bot.orders[1:], 1):
        _fill_group(o.conditions, i, cser, cop, cthr, index)
    _fill_group(bot.tp_conditions, G_TP, cser, cop, cthr, index)
    _fill_group(bot.sl_conditions, G_SL, cser, cop, cthr, index)
    indent = np.zeros(MAX_ORDERS)
    volume = np.zeros(MAX_ORDERS)
    for i, o in enumerate(bot.orders):
        indent[i] = o.indent / 100.0
        volume[i] = o.volume / 100.0
    fparams = np.array(
        [
            bot.pull_up / 100.0,
            bot.tp_min_pnl / 100.0,
            bot.tp_pct / 100.0,
            bot.sl_indent / 100.0,
            bot.sl_pct / 100.0,
            sl.bot_share / 100.0,
            sl.leverage,
        ]
    )
    iparams = np.array(
        [len(bot.orders), bot.tp_mode, bot.sl_mode, sl.block, sl.dyn_block_order, 1 if bot.direction > 0 else -1],
        dtype=np.int64,
    )
    return {"cser": cser, "cop": cop, "cthr": cthr, "indent": indent, "volume": volume, "fparams": fparams, "iparams": iparams}


def pack_global(pf: PortfolioConfig) -> np.ndarray:
    return np.array(
        [
            pf.deposit,
            pf.maker_fee / 100.0,
            pf.taker_fee / 100.0,
            pf.funding_8h / 100.0,
            pf.mmr / 100.0,
            1.0 if pf.compound else 0.0,
        ]
    )


def save_config(path: str, sleeves: list[Sleeve], pf: PortfolioConfig, extra: dict | None = None) -> None:
    payload = {"sleeves": [s.to_dict() for s in sleeves], "portfolio": pf.to_dict()}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_config(path: str) -> tuple[list[Sleeve], PortfolioConfig]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return [Sleeve.from_dict(d) for d in payload["sleeves"]], PortfolioConfig(**payload["portfolio"])
