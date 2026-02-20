"""
Самоулучшающийся оптимизатор стратегий (v4 — NSGA-II Multi-Objective).

Ключевые отличия от v3 (TPE single-objective):
- **NSGA-II**: многокритериальная оптимизация с Pareto front.
  Три объектива: median_return, min_split_return, worst_drawdown.
  Нет ручных весов или штрафов — trade-off решает Pareto front.
- **Кросс-валидация**: каждый trial тестируется на ВСЕХ Walk-Forward сплитах.
- **Pareto front**: трейдер сам выбирает из набора решений
  с разными профилями риска.
- **Жёсткие фильтры**: минимум 30 сделок НА КАЖДОМ тестовом сплите.
- **Эволюционный мета-цикл**: ротация индикаторов между раундами
  на основе IndicatorImportanceTracker.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler

from .backtest import BacktestResult, run_backtest
from .indicators import INDICATOR_REGISTRY
from .strategy import StrategyConfig, config_to_dict, generate_exit_signals, generate_signals

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = Path("output/strategy_lab")

# === Индикаторы и группы ===
ALL_INDICATORS = list(INDICATOR_REGISTRY.keys())

# Группы индикаторов по категориям (все индикаторы из INDICATOR_REGISTRY должны быть покрыты)
INDICATOR_GROUPS = {
    "trend": ["ema_cross", "ema_distance", "supertrend", "adx_direction", "adx", "triple_ema", "lr_slope",
              "ichimoku", "ichimoku_cloud", "hull_ma", "heikin_ashi"],
    "momentum": ["rsi", "rsi_signal", "stochastic", "macd_signal", "macd_cross", "roc", "momentum",
                 "williams_r", "cci", "aroon"],
    "volatility": ["bb_position", "bb_width", "keltner_pos", "atr_norm", "donchian_pos", "squeeze",
                   "real_vol", "atr_pctl"],
    "volume": ["obv_slope", "vol_zscore", "vwap_dist", "taker_delta", "mfi", "cmf", "vwma_dist"],
    "structure": ["funding", "basis", "fear_greed"],
    "multitf": ["mtf_ema_1h", "mtf_ema_4h", "mtf_rsi_1h"],
    "pattern": ["price_zscore", "consec_candles", "body_ratio", "upper_shadow", "lower_shadow",
                "price_change", "drawdown", "range_pos"],
}

# Режимы выбора индикаторов для оптимизации
INDICATOR_SELECTION_MODES = [
    "all",                # Все индикаторы (старое поведение)
    "groups",             # Перебор по группам (категориям)
    "custom",             # Пользовательский список (через параметр)
    "meta_rotate",        # Ротация групп между раундами
    "evolutionary",       # Мутация лучших комбинаций между раундами
    "importance_guided",  # Фокус на самых полезных индикаторах
]

# Indicators suitable for exit signals (react to overbought/oversold)
EXIT_INDICATOR_CANDIDATES = [
    "rsi", "stochastic", "mfi", "williams_r", "cci", "bb_position",
    "keltner_pos", "price_zscore", "range_pos", "aroon", "cmf",
]


# ═══════════════════════════════════════════════════════════════════════
# INDICATOR IMPORTANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════

class IndicatorImportanceTracker:
    """Отслеживает важность индикаторов на основе результатов оптимизации."""

    def __init__(self) -> None:
        # Сколько раз индикатор входил в лучшие стратегии
        self.usage_count: dict[str, int] = {name: 0 for name in ALL_INDICATORS}
        # Средний скор стратегий, использовавших этот индикатор
        self.avg_score: dict[str, float] = {name: 0.0 for name in ALL_INDICATORS}
        # Сумма скоров (для расчёта среднего)
        self._score_sum: dict[str, float] = {name: 0.0 for name in ALL_INDICATORS}
        # Взаимные пары — какие комбинации индикаторов дают лучший результат
        self.pair_scores: dict[tuple[str, str], list[float]] = {}
        # Общее число обработанных стратегий
        self.total_strategies: int = 0

    def update(self, strategies: list[dict]) -> None:
        """Обновить трекер на основе новых результатов."""
        for strat in strategies:
            score = strat.get("score", 0.0)
            if score <= 0:
                continue
            config = strat.get("config", {})
            indicators = [ind["name"] for ind in config.get("entry_indicators", [])]
            self.total_strategies += 1

            for name in indicators:
                if name in self.usage_count:
                    self.usage_count[name] += 1
                    self._score_sum[name] += score
                    self.avg_score[name] = self._score_sum[name] / self.usage_count[name]

            # Обновить парные скоры
            for i, a in enumerate(indicators):
                for b in indicators[i + 1:]:
                    pair = tuple(sorted([a, b]))
                    if pair not in self.pair_scores:
                        self.pair_scores[pair] = []
                    self.pair_scores[pair].append(score)

    def get_top_indicators(self, n: int = 15, min_usage: int = 3) -> list[str]:
        """Вернуть топ-N самых полезных индикаторов."""
        candidates = [
            (name, self.avg_score[name])
            for name in ALL_INDICATORS
            if self.usage_count[name] >= min_usage
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:n]]

    def get_best_pairs(self, n: int = 10) -> list[tuple[str, str, float]]:
        """Вернуть топ-N лучших пар индикаторов."""
        pair_avg = [
            (*pair, float(np.mean(scores)))
            for pair, scores in self.pair_scores.items()
            if len(scores) >= 2
        ]
        pair_avg.sort(key=lambda x: x[2], reverse=True)
        return pair_avg[:n]

    def get_underexplored(self, n: int = 10) -> list[str]:
        """Вернуть наименее исследованные индикаторы."""
        by_usage = sorted(ALL_INDICATORS, key=lambda name: self.usage_count[name])
        return by_usage[:n]

    def get_importance_weights(self) -> dict[str, float]:
        """Вернуть веса важности (вероятность выбора) для каждого индикатора."""
        if self.total_strategies < 5:
            # Недостаточно данных — равномерные веса
            return {name: 1.0 / len(ALL_INDICATORS) for name in ALL_INDICATORS}

        scores = []
        for name in ALL_INDICATORS:
            if self.usage_count[name] > 0:
                # Комбинированный скор: avg_score * log(usage + 1)
                s = self.avg_score[name] * np.log1p(self.usage_count[name])
                scores.append(max(s, 0.01))  # минимальный вес
            else:
                scores.append(0.5)  # неисследованные получают средний вес

        total = sum(scores)
        return {name: s / total for name, s in zip(ALL_INDICATORS, scores)}

    def summary(self) -> str:
        """Текстовой отчёт о важности индикаторов."""
        lines = [f"\n📊 Indicator Importance (top-20, {self.total_strategies} strategies analyzed):"]
        ranked = sorted(
            ALL_INDICATORS,
            key=lambda n: self.avg_score.get(n, 0),
            reverse=True,
        )
        for i, name in enumerate(ranked[:20], 1):
            cnt = self.usage_count[name]
            avg = self.avg_score.get(name, 0)
            lines.append(f"   {i:2d}. {name:<20s}  used={cnt:3d}  avg_score={avg:+.2f}")

        best_pairs = self.get_best_pairs(5)
        if best_pairs:
            lines.append(f"\n🔗 Best indicator pairs:")
            for a, b, s in best_pairs:
                lines.append(f"   {a} + {b}  →  avg_score={s:+.2f}")

        return "\n".join(lines)


def _mutate_indicator_pool(
    base_indicators: list[str],
    tracker: IndicatorImportanceTracker,
    mutation_rate: float = 0.3,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """
    Мутировать пул индикаторов: заменить часть на новые
    на основе важности и exploration.
    """
    if rng is None:
        rng = np.random.default_rng()

    pool = list(base_indicators)
    n_mutate = max(1, int(len(pool) * mutation_rate))

    # Выбрать кандидатов на замену (из тех, что НЕ в текущем пуле)
    available = [n for n in ALL_INDICATORS if n not in set(pool)]
    if not available:
        return pool

    # Веса: 70% importance, 30% exploration (неисследованные)
    weights = tracker.get_importance_weights()
    explore_boost = tracker.get_underexplored(len(available))
    explore_set = set(explore_boost[:max(3, len(available) // 3)])

    selection_weights = []
    for name in available:
        w = weights.get(name, 0.01)
        if name in explore_set:
            w *= 2.0  # бонус за неисследованность
        selection_weights.append(w)

    total_w = sum(selection_weights)
    if total_w <= 0:
        return pool
    selection_weights = [w / total_w for w in selection_weights]

    # Заменить n_mutate индикаторов
    n_replace = min(n_mutate, len(available), len(pool))
    replacements = rng.choice(available, size=n_replace, replace=False, p=selection_weights)

    # Убрать наименее важные из текущего пула
    pool_scores = [(name, weights.get(name, 0.0)) for name in pool]
    pool_scores.sort(key=lambda x: x[1])
    remove_names = {name for name, _ in pool_scores[:n_replace]}

    new_pool = [n for n in pool if n not in remove_names] + list(replacements)
    return new_pool


def _get_group_rotation_pool(
    round_num: int,
    primary_groups: int = 2,
    secondary_count: int = 3,
) -> list[str]:
    """
    Ротация групп между раундами: каждый раунд фокусируется на
    2 основных группах + несколько индикаторов из остальных.
    """
    group_names = list(INDICATOR_GROUPS.keys())
    n_groups = len(group_names)

    # Основные группы ротируются по раундам
    idx1 = (round_num - 1) % n_groups
    idx2 = (round_num - 1 + n_groups // 2) % n_groups
    if idx2 == idx1:
        idx2 = (idx1 + 1) % n_groups
    primary = group_names[idx1]
    secondary = group_names[idx2]

    pool = list(INDICATOR_GROUPS[primary]) + list(INDICATOR_GROUPS[secondary])

    # Добавить по 1 индикатору из оставшихся групп
    rng = np.random.default_rng(seed=round_num * 37)
    for gname in group_names:
        if gname not in (primary, secondary):
            g_inds = INDICATOR_GROUPS[gname]
            if g_inds:
                pick = rng.choice(g_inds)
                if pick not in pool:
                    pool.append(pick)
                    if len(pool) - len(INDICATOR_GROUPS[primary]) - len(INDICATOR_GROUPS[secondary]) >= secondary_count:
                        break

    return pool


# ═══════════════════════════════════════════════════════════════════════
# OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════


def _build_config_from_trial(
    trial: optuna.Trial,
    indicator_selection_mode: str = "all",
    custom_indicators: list[str] | None = None,
    indicator_groups: dict | None = None,
    indicator_pool_override: list[str] | None = None,
) -> StrategyConfig:
    """
    Построить StrategyConfig из параметров Optuna trial.

    Режимы перебора индикаторов:
    - "all"                — все индикаторы (старое поведение)
    - "groups"             — перебор по группам (категориям)
    - "custom"             — пользовательский список
    - "meta_rotate"        — предзаданный пул (передаётся через indicator_pool_override)
    - "evolutionary"       — предзаданный мутированный пул
    - "importance_guided"  — предзаданный пул на основе рейтинга важности

    indicator_pool_override: если задан, используется напрямую как пул индикаторов
    (применяется для meta_rotate / evolutionary / importance_guided).
    """
    if indicator_groups is None:
        indicator_groups = INDICATOR_GROUPS

    # Выбор индикаторов согласно режиму
    if indicator_pool_override is not None:
        # Для meta_rotate / evolutionary / importance_guided
        indicator_pool = indicator_pool_override
    elif indicator_selection_mode == "all":
        indicator_pool = ALL_INDICATORS
    elif indicator_selection_mode == "groups":
        # Сэмплируем группу, затем используем все индикаторы группы
        group_name = trial.suggest_categorical("indicator_group", list(indicator_groups.keys()))
        indicator_pool = list(indicator_groups[group_name])
    elif indicator_selection_mode == "custom" and custom_indicators:
        indicator_pool = custom_indicators
    else:
        indicator_pool = ALL_INDICATORS

    selected: list[str] = []
    all_params: dict[str, dict] = {}

    # Для каждого индикатора в пуле — бинарный флаг
    for ind_name in indicator_pool:
        use_it = trial.suggest_categorical(f"use_{ind_name}", [True, False])
        info = INDICATOR_REGISTRY[ind_name]
        params: dict[str, Any] = {}
        for pname, (pmin, pmax) in info["params"].items():
            if isinstance(pmin, float):
                params[pname] = trial.suggest_float(f"{ind_name}_{pname}", pmin, pmax)
            else:
                params[pname] = trial.suggest_int(f"{ind_name}_{pname}", int(pmin), int(pmax))
        weight = trial.suggest_float(f"{ind_name}_weight", 0.1, 3.0)
        long_thr = trial.suggest_float(f"{ind_name}_long_thr", -0.5, 1.0)
        short_thr = trial.suggest_float(f"{ind_name}_short_thr", -1.0, 0.5)
        all_params[ind_name] = {
            "params": params,
            "weight": weight,
            "long_threshold": long_thr,
            "short_threshold": short_thr,
        }
        if use_it:
            selected.append(ind_name)

    # Гарантировать минимум 2 индикатора
    if len(selected) < 2:
        selected = indicator_pool[:2]
    # Ограничить до 7 (больше → медленнее и шумнее)
    if len(selected) > 7:
        selected = selected[:7]

    # ── Собрать entry_indicators для выбранных ──
    entry_indicators: list[dict[str, Any]] = []
    for ind_name in selected:
        p = all_params[ind_name]
        entry_indicators.append({
            "name": ind_name,
            "params": p["params"],
            "weight": p["weight"],
            "long_threshold": p["long_threshold"],
            "short_threshold": p["short_threshold"],
        })

    # ── Combine mode ──
    combine_mode = trial.suggest_categorical("combine_mode", ["weighted", "vote", "all"])
    combine_threshold = trial.suggest_float("combine_threshold", 0.2, 0.8)

    # ── Filters ──
    adx_filter = trial.suggest_categorical("adx_filter", [True, False])
    vol_filter = trial.suggest_categorical("vol_filter", [True, False])
    time_filter = trial.suggest_categorical("time_filter", [True, False])

    # ── Risk Management (tighter ranges to reduce overfitting) ──
    leverage = trial.suggest_float("leverage", 2.0, 10.0)
    risk_per_trade = trial.suggest_float("risk_per_trade_pct", 5.0, 20.0)

    # ── Exit Mode ──
    exit_mode = trial.suggest_categorical("exit_mode", ["fixed", "atr", "hybrid"])

    # Fixed % exits (always suggest for stable search space)
    stop_loss = trial.suggest_float("stop_loss_pct", 1.0, 8.0)
    take_profit = trial.suggest_float("take_profit_pct", 1.5, 20.0)
    trailing_stop = trial.suggest_float("trailing_stop_pct", 0.5, 5.0)

    # ATR dynamic exits
    atr_period = trial.suggest_int("atr_exit_period", 7, 28)
    atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 5.0)
    atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.5, 8.0)
    atr_trailing_mult = trial.suggest_float("atr_trailing_mult", 0.5, 4.0)

    # ── Exit Indicators ──
    use_exit_indicators = trial.suggest_categorical("use_exit_indicators", [True, False])
    exit_indicators: list[dict[str, Any]] = []
    # Always suggest params for stable search space
    for exit_ind_name in EXIT_INDICATOR_CANDIDATES:
        use_exit_ind = trial.suggest_categorical(f"exit_use_{exit_ind_name}", [True, False])
        long_exit_thr = trial.suggest_float(f"exit_{exit_ind_name}_long_thr", 0.3, 0.95)
        short_exit_thr = trial.suggest_float(f"exit_{exit_ind_name}_short_thr", -0.95, -0.3)
        # Indicator params — reuse entry params if selected, else suggest fresh
        exit_params: dict[str, Any] = {}
        info = INDICATOR_REGISTRY[exit_ind_name]
        for pname, (pmin, pmax) in info["params"].items():
            if isinstance(pmin, float):
                exit_params[pname] = trial.suggest_float(f"exit_{exit_ind_name}_{pname}", pmin, pmax)
            else:
                exit_params[pname] = trial.suggest_int(f"exit_{exit_ind_name}_{pname}", int(pmin), int(pmax))

        if use_exit_indicators and use_exit_ind:
            exit_indicators.append({
                "name": exit_ind_name,
                "params": exit_params,
                "long_exit_thr": long_exit_thr,
                "short_exit_thr": short_exit_thr,
            })

    # Limit exit indicators to max 3
    if len(exit_indicators) > 3:
        exit_indicators = exit_indicators[:3]

    max_hold = trial.suggest_int("max_hold_bars", 16, 672)  # 4h to 7 days
    exit_flip = trial.suggest_categorical("exit_on_signal_flip", [True, False])

    # ── Filter params (always suggest to keep stable search space) ──
    adx_period = trial.suggest_int("adx_period", 7, 28)
    adx_min = trial.suggest_float("adx_min", 15.0, 35.0)
    vol_period = trial.suggest_int("vol_period", 24, 96)
    vol_min_z = trial.suggest_float("vol_min_z", -1.5, 0.5)
    trade_h_start = trial.suggest_int("trade_h_start", 0, 12)
    trade_h_end = trial.suggest_int("trade_h_end", 12, 24)

    return StrategyConfig(
        entry_indicators=entry_indicators,
        combine_mode=combine_mode,
        combine_threshold=combine_threshold,
        adx_filter_enabled=adx_filter,
        adx_filter_period=adx_period,
        adx_filter_min=adx_min,
        vol_filter_enabled=vol_filter,
        vol_filter_period=vol_period,
        vol_filter_min_zscore=vol_min_z,
        time_filter_enabled=time_filter,
        trade_hours_start=trade_h_start,
        trade_hours_end=trade_h_end,
        leverage=leverage,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        trailing_stop_pct=trailing_stop,
        max_hold_bars=max_hold,
        exit_on_signal_flip=exit_flip,
        risk_per_trade_pct=risk_per_trade,
        exit_mode=exit_mode,
        atr_period=atr_period,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        atr_trailing_mult=atr_trailing_mult,
        exit_indicators=exit_indicators,
    )


# ═══════════════════════════════════════════════════════════════════════
# NSGA-II MULTI-OBJECTIVE UTILITIES
# ═══════════════════════════════════════════════════════════════════════

# Objective names and directions for Optuna NSGA-II
OBJECTIVE_NAMES = ("median_return", "min_split_return", "worst_drawdown")
OBJECTIVE_DIRECTIONS = ["maximize", "maximize", "maximize"]
# Note: worst_drawdown is negative (e.g. -30%); maximizing means less negative = better.

# Sentinel values for infeasible / bad trials (dominated by any real solution)
_BAD_TRIAL = (-100.0, -100.0, -100.0)
_WEAK_TRIAL = (-50.0, -50.0, -50.0)


def compute_utility(values: tuple[float, ...] | list[float]) -> float:
    """Lightweight scalar utility for ranking Pareto-optimal solutions.

    Used for seeding next rounds, importance tracking, and display ordering.
    NOT used inside NSGA-II optimisation — no manual weights affect the search.

    values: (median_return, min_split_return, worst_drawdown)
    """
    median_ret, min_ret, worst_dd = values[0], values[1], values[2]
    return 0.5 * median_ret + 0.3 * min_ret + 0.2 * worst_dd


def create_objective(
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    min_trades_per_split: int = 30,
    indicator_selection_mode: str = "all",
    custom_indicators: list[str] | None = None,
    indicator_groups: dict | None = None,
    indicator_pool_override: list[str] | None = None,
) -> callable:
    """
    NSGA-II multi-objective: один trial тестируется на ВСЕХ Walk-Forward сплитах.

    Возвращает 3 объектива (все maximize):
      1) median_return      — медиана monthly return по сплитам (прибыльность)
      2) min_split_return   — worst-split return (робастность во времени)
      3) worst_drawdown     — максимальная просадка (риск; менее отрицательный = лучше)

    Нет ручных весов / штрафов / бонусов — Pareto front показывает все trade-off.
    """

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        try:
            config = _build_config_from_trial(
                trial,
                indicator_selection_mode=indicator_selection_mode,
                custom_indicators=custom_indicators,
                indicator_groups=indicator_groups,
                indicator_pool_override=indicator_pool_override,
            )

            split_monthly_returns: list[float] = []
            split_sharpes: list[float] = []
            split_drawdowns: list[float] = []
            split_profit_factors: list[float] = []
            split_win_rates: list[float] = []
            split_trades: list[int] = []
            total_trades = 0
            failed_splits = 0

            for split_idx, (train_df, test_df) in enumerate(splits):
                # ── Generate signals on FULL train+test for correct expanding normalization ──
                # Then slice out the test portion. This matches real-world usage where
                # indicators see all prior history before the test period.
                full_split_df = pd.concat([train_df, test_df], ignore_index=True)
                full_signals = generate_signals(full_split_df, config)

                # Generate exit indicator signals if configured
                full_exit_signals = None
                if config.exit_indicators:
                    full_exit_signals = generate_exit_signals(full_split_df, config)
                
                train_len = len(train_df)
                train_signals = full_signals.iloc[:train_len].reset_index(drop=True)
                test_signals = full_signals.iloc[train_len:].reset_index(drop=True)

                train_exit_signals = None
                test_exit_signals = None
                if full_exit_signals is not None:
                    train_exit_signals = full_exit_signals.iloc[:train_len].reset_index(drop=True)
                    test_exit_signals = full_exit_signals.iloc[train_len:].reset_index(drop=True)
                
                # ── Train backtest (quick rejection) ──
                train_result = run_backtest(train_df, train_signals, config, exit_signals=train_exit_signals)

                if train_result.total_trades < min_trades_per_split // 3:
                    return _BAD_TRIAL  # strategy produces almost no trades

                # Prune on train: if train DD is catastrophic, skip
                if train_result.max_drawdown_pct < -80:
                    return _BAD_TRIAL
                # Prune: if train has terrible win rate
                if train_result.total_trades > 20 and train_result.win_rate < 20:
                    return _BAD_TRIAL

                # ── Test backtest (out-of-sample) ──
                test_result = run_backtest(test_df, test_signals, config, exit_signals=test_exit_signals)

                if test_result.total_trades < min_trades_per_split:
                    failed_splits += 1
                    if failed_splits > len(splits) // 2:
                        return _WEAK_TRIAL  # too few trades on majority of splits
                    continue  # skip this split but keep going

                split_monthly_returns.append(test_result.avg_monthly_return)
                split_sharpes.append(test_result.sharpe_ratio)
                split_drawdowns.append(test_result.max_drawdown_pct)
                split_profit_factors.append(test_result.profit_factor)
                split_win_rates.append(test_result.win_rate)
                split_trades.append(test_result.total_trades)
                total_trades += test_result.total_trades

            # Need results from at least half the splits
            if len(split_monthly_returns) < max(1, len(splits) // 2):
                return _WEAK_TRIAL

            median_return = float(np.median(split_monthly_returns))
            mean_return = float(np.mean(split_monthly_returns))
            std_return = float(np.std(split_monthly_returns)) if len(split_monthly_returns) > 1 else 0.0
            min_return = float(np.min(split_monthly_returns))
            max_return = float(np.max(split_monthly_returns))
            median_sharpe = float(np.median(split_sharpes))
            worst_dd = float(np.min(split_drawdowns))  # most negative
            median_pf = float(np.median(split_profit_factors))
            median_wr = float(np.median(split_win_rates))
            n_indicators = len(config.entry_indicators)

            # ══════════════════════════════════════════════════════════
            # NSGA-II OBJECTIVES (no manual weights / bonuses / penalties)
            #   1) median_return      — profitability     (maximize)
            #   2) min_split_return   — robustness        (maximize)
            #   3) worst_drawdown     — risk              (maximize = less negative)
            # All trade-offs are resolved by the Pareto front, not by us.
            # ══════════════════════════════════════════════════════════

            # Anti-scalping feasibility filter (micro-scalping is unreproducible)
            avg_trades = total_trades / max(len(split_monthly_returns), 1)
            if avg_trades > 0:
                total_bars_in_splits = sum(len(s[1]) for s in splits if len(s) > 1)
                avg_bars_per_trade = total_bars_in_splits / max(avg_trades, 1)
                if avg_bars_per_trade < 2:
                    return _BAD_TRIAL  # extreme scalping

            # Store detailed metrics for analysis / display
            trial.set_user_attr("median_monthly_return", median_return)
            trial.set_user_attr("mean_monthly_return", mean_return)
            trial.set_user_attr("std_monthly_return", std_return)
            trial.set_user_attr("min_monthly_return", min_return)
            trial.set_user_attr("max_monthly_return", max_return)
            trial.set_user_attr("median_sharpe", median_sharpe)
            trial.set_user_attr("worst_drawdown", worst_dd)
            trial.set_user_attr("median_pf", median_pf)
            trial.set_user_attr("median_wr", median_wr)
            trial.set_user_attr("total_trades", total_trades)
            trial.set_user_attr("n_indicators", n_indicators)
            trial.set_user_attr("leverage", config.leverage)
            trial.set_user_attr("exit_mode", config.exit_mode)
            trial.set_user_attr("n_exit_indicators", len(config.exit_indicators))
            trial.set_user_attr("split_returns", split_monthly_returns)
            trial.set_user_attr("failed_splits", failed_splits)
            trial.set_user_attr("config", config_to_dict(config))

            return (median_return, min_return, worst_dd)

        except KeyboardInterrupt:
            trial.set_user_attr("error", "KeyboardInterrupt")
            return _BAD_TRIAL
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return _BAD_TRIAL

    return objective


# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD DATA SPLIT
# ═══════════════════════════════════════════════════════════════════════

def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.5,
    gap_bars: int = 96,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Anchored Walk-Forward splits: train всегда начинается от начала данных,
    тест постепенно сдвигается.

    Для 2 лет (70,080 bars @15m):
    Split 0: Train [0..35,040], Gap, Test [35,136..46,720]  (6 mo test)
    Split 1: Train [0..46,816], Gap, Test [46,912..58,496]  (6 mo test)
    Split 2: Train [0..58,592], Gap, Test [58,688..70,080]  (5.9 mo test)

    Позволяет оценить стратегию на разных рыночных периодах,
    при этом train всегда растёт (как в реальной торговле).
    """
    n = len(df)
    min_test_bars = 2880 * 2  # min 2 months test

    # Calculate test window size (approximately equal slices of the last portion)
    total_test_portion = int(n * (1 - train_ratio))
    test_window = total_test_portion // n_splits

    if test_window < min_test_bars:
        test_window = min_test_bars

    splits = []
    for i in range(n_splits):
        # Test window slides forward
        test_end = n - (n_splits - 1 - i) * test_window
        test_start = test_end - test_window
        train_end = test_start - gap_bars

        if train_end < min_test_bars:  # not enough train data
            continue
        if test_end > n:
            test_end = n
        if test_start < 0:
            continue

        train_df = df.iloc[:train_end].reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)

        if len(test_df) < min_test_bars:
            continue

        splits.append((train_df, test_df))

    if not splits:
        # Fallback: simple 70/30
        split_idx = int(n * train_ratio)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx + gap_bars:].reset_index(drop=True)
        splits.append((train_df, test_df))

    return splits


# ═══════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

class StrategyOptimizer:
    """Многокритериальный оптимизатор стратегий (v4 — NSGA-II + Indicator Auto-Permutation)."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_monthly_return: float = 30.0,
        max_drawdown_limit: float = -60.0,
        min_trades_per_split: int = 30,
        n_walk_forward_splits: int = 5,
        train_ratio: float = 0.5,
        n_jobs: int = 1,
    ):
        self.df = df
        self.target = target_monthly_return
        self.max_dd = max_drawdown_limit
        self.min_trades = min_trades_per_split
        self.n_jobs = n_jobs
        self.best_strategies: list[dict] = []
        self.history: list[dict] = []

        # Indicator importance tracker (auto-permutation)
        self.indicator_tracker = IndicatorImportanceTracker()

        # Walk-forward splits
        self.splits = walk_forward_split(df, n_splits=n_walk_forward_splits, train_ratio=train_ratio)
        print(f"\n📊 Walk-Forward: {len(self.splits)} split(s)")
        for i, (tr, te) in enumerate(self.splits):
            tr_months = len(tr) / (4 * 24 * 30)
            te_months = len(te) / (4 * 24 * 30)
            print(f"   Split {i}: Train={len(tr):,} bars ({tr_months:.1f}mo), Test={len(te):,} bars ({te_months:.1f}mo)")

        # Ensure output dir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_rounds: int = 10,
        trials_per_round: int = 200,
        patience: int = 3,
        indicator_mode: str = "all",
        custom_indicators: list[str] | None = None,
        mutation_rate: float = 0.3,
    ) -> list[dict]:
        """
        NSGA-II многокритериальная оптимизация (v4) с автоперебором индикаторов.

        3 объектива (Pareto front, без ручных весов):
          - median_return     — прибыльность (%/мес)
          - min_split_return  — робастность (худший сплит)
          - worst_drawdown    — риск (менее отрицательный = лучше)

        indicator_mode:
          - "all"                — все индикаторы
          - "groups"             — перебор по группам внутри trial
          - "custom"             — фиксированный список custom_indicators
          - "meta_rotate"        — ротация групп между раундами
          - "evolutionary"       — мутация пула на основе лучших стратегий
          - "importance_guided"  — фокус на самых успешных индикаторах

        Выход: список стратегий с Pareto-оптимальными решениями.
        """
        print(f"\n{'='*70}")
        print(f"🚀 STRATEGY LAB v4 — NSGA-II MULTI-OBJECTIVE + INDICATOR AUTO-PERMUTATION")
        print(f"   Target: {self.target:.1f}% monthly return")
        print(f"   Max Drawdown: {self.max_dd:.1f}%")
        print(f"   Min trades per split: {self.min_trades}")
        print(f"   Rounds: {n_rounds}, Trials/round: {trials_per_round}")
        print(f"   Walk-Forward splits: {len(self.splits)}")
        print(f"   Indicator mode: {indicator_mode}")
        if indicator_mode == "custom" and custom_indicators:
            print(f"   Custom indicators: {custom_indicators}")
        print(f"{'='*70}\n")

        best_score = -float("inf")
        no_improve_count = 0
        total_start = time.time()

        # Для evolutionary/importance_guided — текущий пул индикаторов
        current_indicator_pool: list[str] | None = None
        rng = np.random.default_rng(seed=42)

        for round_num in range(1, n_rounds + 1):
            round_start = time.time()
            print(f"\n{'─'*50}")
            print(f"ROUND {round_num}/{n_rounds}")
            print(f"{'─'*50}")

            # ── Определить пул индикаторов для этого раунда ──
            pool_override: list[str] | None = None
            effective_mode = indicator_mode

            if indicator_mode == "meta_rotate":
                pool_override = _get_group_rotation_pool(round_num)
                print(f"  🔄 Meta-rotate pool ({len(pool_override)} indicators): {pool_override}")

            elif indicator_mode == "evolutionary":
                if round_num == 1 or current_indicator_pool is None:
                    # Первый раунд — все индикаторы
                    current_indicator_pool = list(ALL_INDICATORS)
                else:
                    # Мутировать пул на основе текущего рейтинга
                    current_indicator_pool = _mutate_indicator_pool(
                        current_indicator_pool, self.indicator_tracker,
                        mutation_rate=mutation_rate, rng=rng,
                    )
                pool_override = current_indicator_pool
                print(f"  🧬 Evolutionary pool ({len(pool_override)} indicators): {pool_override}")

            elif indicator_mode == "importance_guided":
                if self.indicator_tracker.total_strategies >= 5:
                    top_inds = self.indicator_tracker.get_top_indicators(n=20, min_usage=2)
                    # Добавить несколько неисследованных для exploration
                    unexplored = self.indicator_tracker.get_underexplored(5)
                    merged = list(dict.fromkeys(top_inds + unexplored))  # unique, order preserved
                    pool_override = merged[:25]  # не больше 25
                    print(f"  🎯 Importance-guided pool ({len(pool_override)} indicators): {pool_override[:10]}...")
                else:
                    print(f"  📊 Not enough data yet, using all indicators")

            # Create NSGA-II study (multi-objective)
            sampler = NSGAIISampler(
                population_size=min(50, trials_per_round // 4),
                seed=round_num * 42,
            )
            study = optuna.create_study(
                directions=OBJECTIVE_DIRECTIONS,
                sampler=sampler,
                study_name=f"round_{round_num}",
            )

            # Seed with best known strategies
            if self.best_strategies:
                self._seed_study(study, self.best_strategies[:5], indicator_pool=pool_override)

            # Create cross-validated multi-objective
            objective = create_objective(
                splits=self.splits,
                min_trades_per_split=self.min_trades,
                indicator_selection_mode=effective_mode,
                custom_indicators=custom_indicators,
                indicator_pool_override=pool_override,
            )

            # Run optimization
            study.optimize(
                objective,
                n_trials=trials_per_round,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
            )

            # Collect Pareto-optimal and top trials
            completed = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.values is not None
                and t.values[0] > -50  # filter out infeasible
            ]
            # Sort by utility for seeding / display
            sorted_trials = sorted(
                completed,
                key=lambda t: compute_utility(t.values),
                reverse=True,
            )
            # Pareto front from Optuna
            pareto_trials = study.best_trials
            pareto_set = set(id(t) for t in pareto_trials)

            round_results: list[dict] = []
            for trial in sorted_trials[:30]:
                obj = trial.values
                result_dict = {
                    "score": compute_utility(obj),  # backward-compat scalar
                    "objectives": {
                        "median_return": obj[0],
                        "min_split_return": obj[1],
                        "worst_drawdown": obj[2],
                    },
                    "pareto_optimal": id(trial) in pareto_set,
                    "round": round_num,
                    **{k: v for k, v in trial.user_attrs.items() if k != "config"},
                }
                if "config" in trial.user_attrs:
                    result_dict["config"] = trial.user_attrs["config"]
                round_results.append(result_dict)

            # ── Анализ раунда ──
            n_pareto = len([r for r in round_results if r.get("pareto_optimal")])

            if round_results:
                top = round_results[0]
                round_best = top["score"]
                obj = top["objectives"]

                print(f"\n  📈 Round {round_num}: Pareto front = {n_pareto} solutions")
                print(f"     ── Top by utility ({round_best:.2f}) ──")
                print(f"     Median Return: {obj['median_return']:.2f}%")
                print(f"     Min Split Ret: {obj['min_split_return']:.2f}% (worst split)")
                print(f"     Worst DD:      {obj['worst_drawdown']:.1f}%")
                print(f"     Sharpe:        {top.get('median_sharpe', 0):.2f}")
                print(f"     PF:            {top.get('median_pf', 0):.2f}")
                print(f"     WR:            {top.get('median_wr', 0):.1f}%")
                print(f"     Trades:        {top.get('total_trades', 0)}")
                print(f"     Indicators:    {top.get('n_indicators', 0)}")
                if "split_returns" in top:
                    print(f"     Per-split:     {['%.1f%%' % r for r in top['split_returns']]}")

                # Show Pareto extremes
                pareto_results = [r for r in round_results if r.get("pareto_optimal")]
                if len(pareto_results) >= 2:
                    by_ret = max(pareto_results, key=lambda r: r["objectives"]["median_return"])
                    by_dd = max(pareto_results, key=lambda r: r["objectives"]["worst_drawdown"])
                    print(f"     ── Pareto: best return  → {by_ret['objectives']['median_return']:+.1f}%/mo, DD={by_ret['objectives']['worst_drawdown']:.1f}%")
                    print(f"     ── Pareto: lowest risk  → {by_dd['objectives']['median_return']:+.1f}%/mo, DD={by_dd['objectives']['worst_drawdown']:.1f}%")

                # Update best strategies
                for res in round_results[:30]:
                    self.best_strategies.append(res)

                # Sort by utility and keep top 50
                self.best_strategies.sort(key=lambda s: s["score"], reverse=True)
                self.best_strategies = self.best_strategies[:50]

                # Обновить трекер важности индикаторов
                self.indicator_tracker.update(round_results)

                # Check improvement
                if round_best > best_score:
                    best_score = round_best
                    no_improve_count = 0
                    print(f"\n  ✅ NEW BEST utility: {best_score:.2f}")
                else:
                    no_improve_count += 1
                    print(f"\n  ⏳ No improvement ({no_improve_count}/{patience})")

                # Check target (by median return of best Pareto solution)
                best_median = max(r["objectives"]["median_return"] for r in round_results)
                if best_median >= self.target:
                    print(f"\n  🎯 TARGET ACHIEVED! {best_median:.2f}% >= {self.target}%")
                    break

                # Patience exhausted — expand search
                if no_improve_count >= patience:
                    print(f"\n  🔄 Patience exhausted. Expanding search...")
                    no_improve_count = 0
                    trials_per_round = min(trials_per_round + 100, 1000)
            else:
                all_trials = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
                ]
                n_bad = sum(1 for t in all_trials if t.values[0] <= -100)
                n_weak = sum(1 for t in all_trials if -100 < t.values[0] <= -50)
                print(f"\n  ⚠️ No valid strategies. Total trials: {len(all_trials)}")
                print(f"     infeasible (train prune): {n_bad}")
                print(f"     weak (too few test trades): {n_weak}")
                no_improve_count += 1

            round_time = time.time() - round_start
            print(f"\n  ⏱️ Round time: {round_time:.0f}s ({round_time/max(trials_per_round,1):.1f}s/trial)")

            # History
            self.history.append({
                "round": round_num,
                "best_utility": best_score,
                "round_best_utility": round_results[0]["score"] if round_results else 0,
                "n_candidates": len(round_results),
                "n_pareto": n_pareto,
                "time_s": round_time,
                # Legacy compat keys
                "best_score": best_score,
                "round_best": round_results[0]["score"] if round_results else 0,
            })

        total_time = time.time() - total_start
        n_pareto_total = len([s for s in self.best_strategies if s.get("pareto_optimal")])
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE (NSGA-II)")
        print(f"  Total time: {total_time / 60:.1f} min")
        print(f"  Best utility score: {best_score:.2f}")
        print(f"  Pareto-optimal in pool: {n_pareto_total}")
        print(f"  Top strategies found: {len(self.best_strategies)}")
        print(f"  Indicator mode: {indicator_mode}")
        print(f"{'='*70}")

        # Отчёт по важности индикаторов
        if self.indicator_tracker.total_strategies > 0:
            print(self.indicator_tracker.summary())

        # Save results
        self._save_results()

        return self.best_strategies

    def _seed_study(
        self,
        study: optuna.Study,
        strategies: list[dict],
        indicator_pool: list[str] | None = None,
    ) -> None:
        """Сидировать study лучшими известными стратегиями.

        indicator_pool: если задан, ограничить бинарные флаги и параметры
        только этим набором индикаторов (для meta_rotate / evolutionary / importance_guided).
        """
        pool = indicator_pool if indicator_pool is not None else ALL_INDICATORS
        pool_set = set(pool)

        for strat in strategies:
            if "config" not in strat:
                continue
            try:
                cfg = strat["config"]
                params: dict[str, Any] = {}

                # Binary flags for indicator selection (only for indicators in pool)
                selected_names = {ind["name"] for ind in cfg.get("entry_indicators", [])}
                for ind_name in pool:
                    params[f"use_{ind_name}"] = ind_name in selected_names

                # Indicator params (only for indicators in pool)
                for ind in cfg.get("entry_indicators", []):
                    ind_name = ind["name"]
                    if ind_name not in pool_set:
                        continue
                    for pname, pval in ind.get("params", {}).items():
                        params[f"{ind_name}_{pname}"] = pval
                    params[f"{ind_name}_weight"] = ind.get("weight", 1.0)
                    params[f"{ind_name}_long_thr"] = ind.get("long_threshold", 0.0)
                    params[f"{ind_name}_short_thr"] = ind.get("short_threshold", 0.0)

                params["combine_mode"] = cfg.get("combine_mode", "weighted")
                params["combine_threshold"] = cfg.get("combine_threshold", 0.5)
                params["leverage"] = cfg.get("leverage", 10.0)
                params["stop_loss_pct"] = cfg.get("stop_loss_pct", 2.0)
                params["take_profit_pct"] = cfg.get("take_profit_pct", 4.0)
                params["trailing_stop_pct"] = cfg.get("trailing_stop_pct", 0.0)
                params["max_hold_bars"] = cfg.get("max_hold_bars", 96)
                params["risk_per_trade_pct"] = cfg.get("risk_per_trade_pct", 10.0)
                params["adx_filter"] = cfg.get("adx_filter_enabled", False)
                params["vol_filter"] = cfg.get("vol_filter_enabled", False)
                params["time_filter"] = cfg.get("time_filter_enabled", False)
                params["exit_on_signal_flip"] = cfg.get("exit_on_signal_flip", True)
                params["exit_mode"] = cfg.get("exit_mode", "fixed")
                params["atr_exit_period"] = cfg.get("atr_period", 14)
                params["atr_sl_mult"] = cfg.get("atr_sl_mult", 2.0)
                params["atr_tp_mult"] = cfg.get("atr_tp_mult", 3.0)
                params["atr_trailing_mult"] = cfg.get("atr_trailing_mult", 1.5)
                params["use_exit_indicators"] = bool(cfg.get("exit_indicators", []))

                # Seed exit indicator flags
                exit_ind_names = {ei["name"] for ei in cfg.get("exit_indicators", [])}
                for exit_ind_name in EXIT_INDICATOR_CANDIDATES:
                    params[f"exit_use_{exit_ind_name}"] = exit_ind_name in exit_ind_names
                    for ei in cfg.get("exit_indicators", []):
                        if ei["name"] == exit_ind_name:
                            params[f"exit_{exit_ind_name}_long_thr"] = ei.get("long_exit_thr", 0.7)
                            params[f"exit_{exit_ind_name}_short_thr"] = ei.get("short_exit_thr", -0.7)
                            for pname, pval in ei.get("params", {}).items():
                                params[f"exit_{exit_ind_name}_{pname}"] = pval

                study.enqueue_trial(params)
            except Exception:
                continue

    def _save_results(self) -> None:
        """Сохранить результаты на диск."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best strategies
        strategies_file = OUTPUT_DIR / f"strategies_{timestamp}.json"
        save_data = []
        for s in self.best_strategies[:20]:
            entry = {k: v for k, v in s.items() if k != "config"}
            if "config" in s:
                entry["config"] = s["config"]
            save_data.append(entry)

        with open(strategies_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n💾 Strategies saved: {strategies_file}")

        # Save optimization history
        history_file = OUTPUT_DIR / f"history_{timestamp}.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 History saved: {history_file}")

    def validate_strategy(
        self,
        config: StrategyConfig,
    ) -> BacktestResult:
        """
        Полная валидация стратегии на полных данных.
        (для финальной проверки лучших кандидатов)
        """
        signals = generate_signals(self.df, config)
        exit_sigs = generate_exit_signals(self.df, config) if config.exit_indicators else None
        return run_backtest(self.df, signals, config, exit_signals=exit_sigs)


def load_strategy_from_json(path: str) -> StrategyConfig:
    """Загрузить конфигурацию стратегии из JSON файла."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        data = data[0]  # take best

    cfg = data.get("config", data)

    return StrategyConfig(
        entry_indicators=cfg.get("entry_indicators", []),
        combine_mode=cfg.get("combine_mode", "weighted"),
        combine_threshold=cfg.get("combine_threshold", 0.5),
        adx_filter_enabled=cfg.get("adx_filter_enabled", False),
        adx_filter_period=cfg.get("adx_filter_period", 14),
        adx_filter_min=cfg.get("adx_filter_min", 20.0),
        vol_filter_enabled=cfg.get("vol_filter_enabled", False),
        vol_filter_period=cfg.get("vol_filter_period", 48),
        vol_filter_min_zscore=cfg.get("vol_filter_min_zscore", -0.5),
        time_filter_enabled=cfg.get("time_filter_enabled", False),
        trade_hours_start=cfg.get("trade_hours_start", 0),
        trade_hours_end=cfg.get("trade_hours_end", 24),
        leverage=cfg.get("leverage", 10.0),
        stop_loss_pct=cfg.get("stop_loss_pct", 2.0),
        take_profit_pct=cfg.get("take_profit_pct", 4.0),
        trailing_stop_pct=cfg.get("trailing_stop_pct", 0.0),
        max_hold_bars=cfg.get("max_hold_bars", 96),
        exit_on_signal_flip=cfg.get("exit_on_signal_flip", True),
        risk_per_trade_pct=cfg.get("risk_per_trade_pct", 10.0),
        exit_mode=cfg.get("exit_mode", "fixed"),
        atr_period=cfg.get("atr_period", 14),
        atr_sl_mult=cfg.get("atr_sl_mult", 2.0),
        atr_tp_mult=cfg.get("atr_tp_mult", 3.0),
        atr_trailing_mult=cfg.get("atr_trailing_mult", 1.5),
        exit_indicators=cfg.get("exit_indicators", []),
    )
