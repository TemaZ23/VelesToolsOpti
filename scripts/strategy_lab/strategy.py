"""
Параметризованная торговая стратегия.

Стратегия принимает конфигурацию от Optuna:
- Какие индикаторы использовать (из 37 доступных)
- Параметры каждого индикатора
- Логика комбинирования (AND/OR/VOTE)
- Параметры входа/выхода (threshold, trailing stop, take profit)
- Risk management (leverage, position size, max drawdown)

Генерирует signals DataFrame: direction (+1 long, -1 short, 0 flat).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .indicators import INDICATOR_REGISTRY, compute_indicator


@dataclass
class StrategyConfig:
    """Полная конфигурация стратегии, найденная оптимизатором."""

    # ── ИНДИКАТОРЫ ──
    # Список кортежей: (name, params, weight, threshold_long, threshold_short)
    entry_indicators: list[dict[str, Any]] = field(default_factory=list)

    # ── ЛОГИКА КОМБИНИРОВАНИЯ ──
    # "vote" = majority vote, "all" = all must agree, "weighted" = weighted sum
    combine_mode: str = "weighted"
    # Порог для weighted mode (e.g., 0.6 = 60% weighted signals must agree)
    combine_threshold: float = 0.5

    # ── ФИЛЬТРЫ ──
    # ADX filter: не торговать в слабом тренде
    adx_filter_enabled: bool = False
    adx_filter_period: int = 14
    adx_filter_min: float = 20.0

    # Volume filter: не торговать при низком объеме
    vol_filter_enabled: bool = False
    vol_filter_period: int = 48
    vol_filter_min_zscore: float = -0.5

    # Kill zone: торговать только в определенные часы (UTC)
    time_filter_enabled: bool = False
    trade_hours_start: int = 0
    trade_hours_end: int = 24

    # ── RISK MANAGEMENT ──
    leverage: float = 10.0
    stop_loss_pct: float = 2.0        # % от entry price (used when exit_mode="fixed")
    take_profit_pct: float = 4.0       # % от entry price (used when exit_mode="fixed")
    trailing_stop_pct: float = 0.0     # 0 = disabled (used when exit_mode="fixed")
    max_positions: int = 1             # concurrent trades
    risk_per_trade_pct: float = 10.0   # % of balance per trade

    # ── DYNAMIC EXITS ──
    exit_mode: str = "fixed"           # "fixed" | "atr" | "hybrid"
    atr_period: int = 14               # ATR period for dynamic exits
    atr_sl_mult: float = 2.0           # SL = ATR * mult (when exit_mode="atr"|"hybrid")
    atr_tp_mult: float = 3.0           # TP = ATR * mult (when exit_mode="atr"|"hybrid")
    atr_trailing_mult: float = 1.5     # Trailing = ATR * mult (0=disabled)

    # ── EXIT INDICATORS (optional, on top of SL/TP) ──
    exit_indicators: list[dict[str, Any]] = field(default_factory=list)
    # Each: {"name": str, "params": dict, "long_exit_thr": float, "short_exit_thr": float}

    # ── EXIT ──
    max_hold_bars: int = 96            # Max bars to hold (0=unlimited)
    exit_on_signal_flip: bool = True   # Close when signal reverses

    # ── META ──
    name: str = ""
    generation: int = 0


def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Генерация сигналов на основе конфигурации стратегии.

    Returns:
        DataFrame с колонками:
        - signal: +1 (long), -1 (short), 0 (flat)
        - strength: 0..1 (сила сигнала)
        - stop_loss: price
        - take_profit: price
    """
    n = len(df)
    result = pd.DataFrame({
        "signal": np.zeros(n),
        "strength": np.zeros(n),
    }, index=df.index)

    if not config.entry_indicators:
        return result

    # ── Compute all indicator values ──
    indicator_values: list[tuple[pd.Series, float, float, float]] = []
    # Each: (values, weight, long_threshold, short_threshold)

    for ind_cfg in config.entry_indicators:
        name = ind_cfg["name"]
        params = ind_cfg.get("params", {})
        weight = ind_cfg.get("weight", 1.0)
        long_thr = ind_cfg.get("long_threshold", 0.0)
        short_thr = ind_cfg.get("short_threshold", 0.0)

        try:
            values = compute_indicator(name, df, params)
            indicator_values.append((values, weight, long_thr, short_thr))
        except Exception:
            continue

    if not indicator_values:
        return result

    # ── Combine signals ──
    if config.combine_mode == "weighted":
        weighted_signal = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for values, weight, long_thr, short_thr in indicator_values:
            sig = pd.Series(0.0, index=df.index)
            sig[values > long_thr] = 1.0
            sig[values < short_thr] = -1.0
            weighted_signal += sig * weight
            total_weight += abs(weight)

        if total_weight > 0:
            normalized = weighted_signal / total_weight
            result["signal"] = np.where(
                normalized > config.combine_threshold, 1,
                np.where(normalized < -config.combine_threshold, -1, 0)
            )
            result["strength"] = normalized.abs()

    elif config.combine_mode == "vote":
        votes = pd.DataFrame(index=df.index)
        for i, (values, weight, long_thr, short_thr) in enumerate(indicator_values):
            sig = pd.Series(0.0, index=df.index)
            sig[values > long_thr] = 1.0
            sig[values < short_thr] = -1.0
            votes[f"v_{i}"] = sig

        majority = len(indicator_values) * config.combine_threshold
        vote_sum = votes.sum(axis=1)
        result["signal"] = np.where(
            vote_sum > majority, 1,
            np.where(vote_sum < -majority, -1, 0)
        )
        result["strength"] = vote_sum.abs() / len(indicator_values)

    elif config.combine_mode == "all":
        all_long = pd.Series(True, index=df.index)
        all_short = pd.Series(True, index=df.index)

        for values, weight, long_thr, short_thr in indicator_values:
            all_long &= (values > long_thr)
            all_short &= (values < short_thr)

        result["signal"] = np.where(all_long, 1, np.where(all_short, -1, 0))
        result["strength"] = (all_long | all_short).astype(float)

    # ── Apply filters ──
    if config.adx_filter_enabled:
        adx_vals = compute_indicator("adx", df, {"period": config.adx_filter_period})
        weak_trend = adx_vals < config.adx_filter_min
        result.loc[weak_trend, "signal"] = 0

    if config.vol_filter_enabled:
        vol_z = compute_indicator("vol_zscore", df, {"period": config.vol_filter_period})
        low_vol = vol_z < config.vol_filter_min_zscore
        result.loc[low_vol, "signal"] = 0

    if config.time_filter_enabled and "timestamp" in df.columns:
        hour = pd.to_datetime(df["timestamp"]).dt.hour
        outside = (hour < config.trade_hours_start) | (hour >= config.trade_hours_end)
        result.loc[outside, "signal"] = 0

    return result


def generate_exit_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Generate per-bar exit indicator signals for dynamic exit mode.

    Returns:
        DataFrame with columns:
        - exit_long: True where long should be exited by indicator
        - exit_short: True where short should be exited by indicator
    """
    n = len(df)
    exit_long = np.zeros(n, dtype=bool)
    exit_short = np.zeros(n, dtype=bool)

    for ind_cfg in config.exit_indicators:
        name = ind_cfg["name"]
        params = ind_cfg.get("params", {})
        long_exit_thr = ind_cfg.get("long_exit_thr", 0.7)
        short_exit_thr = ind_cfg.get("short_exit_thr", -0.7)

        if name not in INDICATOR_REGISTRY:
            continue

        try:
            values = compute_indicator(name, df, params)
            # Exit long when indicator crosses above long_exit_thr (overbought)
            exit_long |= (values > long_exit_thr).values
            # Exit short when indicator crosses below short_exit_thr (oversold)
            exit_short |= (values < short_exit_thr).values
        except Exception:
            continue

    return pd.DataFrame({
        "exit_long": exit_long,
        "exit_short": exit_short,
    }, index=df.index)


def config_to_dict(config: StrategyConfig) -> dict[str, Any]:
    """Сериализация конфигурации для сохранения."""
    return {
        "entry_indicators": config.entry_indicators,
        "combine_mode": config.combine_mode,
        "combine_threshold": config.combine_threshold,
        "adx_filter_enabled": config.adx_filter_enabled,
        "adx_filter_period": config.adx_filter_period,
        "adx_filter_min": config.adx_filter_min,
        "vol_filter_enabled": config.vol_filter_enabled,
        "vol_filter_period": config.vol_filter_period,
        "vol_filter_min_zscore": config.vol_filter_min_zscore,
        "time_filter_enabled": config.time_filter_enabled,
        "trade_hours_start": config.trade_hours_start,
        "trade_hours_end": config.trade_hours_end,
        "leverage": config.leverage,
        "stop_loss_pct": config.stop_loss_pct,
        "take_profit_pct": config.take_profit_pct,
        "trailing_stop_pct": config.trailing_stop_pct,
        "max_hold_bars": config.max_hold_bars,
        "exit_on_signal_flip": config.exit_on_signal_flip,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "exit_mode": config.exit_mode,
        "atr_period": config.atr_period,
        "atr_sl_mult": config.atr_sl_mult,
        "atr_tp_mult": config.atr_tp_mult,
        "atr_trailing_mult": config.atr_trailing_mult,
        "exit_indicators": config.exit_indicators,
        "name": config.name,
        "generation": config.generation,
    }
