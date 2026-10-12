"""Проверки механики движка на синтетических свечах: python -m pytest scripts/dca_lab/tests"""

import numpy as np

from scripts.dca_lab import engine as E
from scripts.dca_lab.indicators import SERIES_INDEX, SERIES_NAMES
from scripts.dca_lab.portfolio import Market, run
from scripts.dca_lab.strategy import SL_FIXED, SL_NONE, TP_FIXED, BotConfig, Condition, Order, PortfolioConfig


def _market(close: np.ndarray, events: dict[str, np.ndarray] | None = None) -> Market:
    """Одна монета, все серии 1m-событий нулевые кроме заданных; спред свечи ±0.01%."""
    n = len(close)
    px = np.stack([close * 1.0001, close * 0.9999, close], axis=1).astype(np.float32)
    ser_parts, offs, lens, tfs = [], [], [], []
    pos = 0
    for name in SERIES_NAMES:
        arr = np.zeros(n, dtype=np.float32)
        if events and name in events:
            arr = events[name].astype(np.float32)
        ser_parts.append(arr)
        offs.append(pos)
        lens.append(n)
        tfs.append(1)  # в тесте все серии минутные
        pos += n
    return Market(
        coins=["TEST"],
        series_names=list(SERIES_NAMES),
        px=px,
        coin_off=np.array([0]),
        coin_len=np.array([n]),
        coin_start=np.array([0]),
        ser=np.concatenate(ser_parts),
        ser_off=np.array([offs]),
        ser_len=np.array([lens]),
        ser_tf=np.array(tfs),
        t_end=n,
    )


def _pf(**kw) -> PortfolioConfig:
    base = dict(deposit=1000, bot_share=10, leverage=1, block=10, dyn_block_order=0, maker_fee=0, taker_fee=0, funding_8h=0, mmr=0)
    base.update(kw)
    return PortfolioConfig(**base)


def test_fixed_tp_and_limit_grid():
    # вход на 1-й минуте, падение на 10% (исполняется 2-й лимитный ордер), возврат к 101
    close = np.concatenate([np.full(5, 100.0), np.linspace(100, 90, 20), np.linspace(90, 101, 30), np.full(10, 101.0)])
    m = _market(close)
    bot = BotConfig(
        name="t",
        entry=[],
        orders=[Order(0.0, 50), Order(5.0, 50)],
        tp_mode=TP_FIXED,
        tp_pct=2.0,
        sl_mode=SL_NONE,
    )
    r = run(m, bot, _pf(), 0, len(close), record_deals=True)
    assert r.stats[E.S_TP] >= 1
    # первая сделка: 50 USDT по 100 и 50 USDT по 95 → средняя ≈ 97.44, тейк +2%
    first = r.deals["pnl"][0]
    qty = 50 / 100 + 50 / 95
    avg = 100 / qty
    assert abs(first - (qty * avg * 1.02 - 100)) < 0.05
    assert r.deals["nf"][0] == 2


def test_signal_order_requires_event_and_indent():
    close = np.concatenate([np.full(3, 100.0), np.linspace(100, 80, 40), np.full(20, 80.0), np.linspace(80, 100, 40)])
    ev = np.zeros(len(close))
    ev[20] = 1  # цена ещё выше -15% — ордер не должен исполниться
    ev[60] = 1  # цена 80 (-20%) — исполняется
    m = _market(close, {"CCI_LEVELS@1m": ev})
    bot = BotConfig(
        name="t",
        entry=[],
        orders=[Order(0.0, 50), Order(15.0, 50, [Condition("CCI_LEVELS@1m", ">", 0.5)])],
        tp_mode=TP_FIXED,
        tp_pct=5.0,
        sl_mode=SL_NONE,
    )
    r = run(m, bot, _pf(), 0, len(close), record_deals=True)
    assert r.deals["nf"][0] == 2
    # средняя = 100 / (0.5 + 50/80/1) ≈ 88.9 → тейк при 93.3
    assert r.stats[E.S_TP] >= 1


def test_pull_up_cancels_pending_order():
    close = np.concatenate([np.full(3, 100.0), np.linspace(100, 102, 30)])
    m = _market(close)
    bot = BotConfig(name="t", entry=[], orders=[Order(1.0, 100)], pull_up=0.5, tp_mode=TP_FIXED, tp_pct=1, sl_mode=SL_NONE)
    r = run(m, bot, _pf(), 0, len(close), record_deals=True)
    assert r.stats[E.S_PULLUPS] >= 1
    assert r.stats[E.S_DEALS] == 0


def test_fixed_stop_and_block():
    close = np.concatenate([np.full(3, 100.0), np.linspace(100, 60, 50)])
    m = _market(close)
    bot = BotConfig(name="t", entry=[], orders=[Order(0.0, 100)], tp_mode=TP_FIXED, tp_pct=50, sl_mode=SL_FIXED, sl_pct=20)
    r = run(m, bot, _pf(), 0, len(close), record_deals=True)
    assert r.stats[E.S_SL] >= 1
    assert abs(r.deals["pnl"][0] + 20) < 0.01  # 100 USDT × -20%
    r0 = run(m, bot, _pf(block=0), 0, len(close))
    assert r0.stats[E.S_DEALS] == 0


def test_series_index_unique():
    assert len(SERIES_INDEX) == len(SERIES_NAMES)
