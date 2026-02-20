"""
CLI entry-point для Strategy Lab.

Использование:
    python -m scripts.strategy_lab.run [OPTIONS]

Примеры:
    # Полный запуск (5 лет, 10 раундов, 200 триалов)
    python -m scripts.strategy_lab.run

    # Быстрый тест (2 года, 3 раунда, 50 триалов)
    python -m scripts.strategy_lab.run --fast

    # Целевой monthly return 50%
    python -m scripts.strategy_lab.run --target 50

    # Валидация сохранённой стратегии
    python -m scripts.strategy_lab.run --validate output/strategy_lab/strategies_*.json

    # Параллельный поиск
    python -m scripts.strategy_lab.run --jobs 4
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ignore SIGINT on Windows so external Ctrl-C doesn't kill optimization ──
if os.name == "nt":
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.crash_analysis.data_loader import load_all_data
from scripts.strategy_lab.backtest import run_backtest
from scripts.strategy_lab.optimizer import (
    OUTPUT_DIR,
    StrategyOptimizer,
    compute_utility,
    load_strategy_from_json,
)
from scripts.strategy_lab.report import generate_optimization_report, generate_report
from scripts.strategy_lab.strategy import StrategyConfig, generate_exit_signals, generate_signals


def prepare_dataframe(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Подготовить единый DataFrame из всех загруженных данных.
    Базовый таймфрейм — 15m фьючерсы, к нему мержим остальное.
    """
    df = data["futures_15m"].copy()

    # Добавить spot данные для basis
    if "spot_15m" in data and len(data["spot_15m"]) > 0:
        spot = data["spot_15m"][["timestamp", "close"]].rename(columns={"close": "spot_close"})
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            spot.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["basis"] = (df["close"] - df["spot_close"]) / df["spot_close"]
    else:
        df["basis"] = 0.0

    # Добавить funding rate
    if "funding" in data and len(data["funding"]) > 0:
        funding = data["funding"][["timestamp", "funding_rate"]]
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            funding.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)
    else:
        df["funding_rate"] = 0.0

    # Добавить fear & greed
    if "fear_greed" in data and len(data["fear_greed"]) > 0:
        fg = data["fear_greed"][["timestamp", "fear_greed_value"]].rename(columns={"fear_greed_value": "fear_greed"})
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            fg.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["fear_greed"] = df["fear_greed"].ffill().fillna(50)
    else:
        df["fear_greed"] = 50.0

    # Добавить taker buy/sell
    if "taker_volume" in data and len(data["taker_volume"]) > 0:
        tv = data["taker_volume"]
        if "buy_vol" in tv.columns and "sell_vol" in tv.columns:
            tv = tv[["timestamp", "buy_vol", "sell_vol"]].copy()
            tv["taker_delta"] = tv["buy_vol"] - tv["sell_vol"]
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                tv[["timestamp", "taker_delta"]].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            df["taker_delta"] = df["taker_delta"].ffill().fillna(0)
        else:
            df["taker_delta"] = 0.0
    else:
        df["taker_delta"] = 0.0

    # Добавить multi-timeframe данные
    for tf_name, tf_key in [("1h", "futures_1h"), ("4h", "futures_4h"), ("1d", "futures_1d")]:
        if tf_key in data and len(data[tf_key]) > 0:
            tf_df = data[tf_key][["timestamp", "close", "volume"]].rename(
                columns={"close": f"close_{tf_name}", "volume": f"volume_{tf_name}"}
            )
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                tf_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

    # Fill NaN
    df = df.ffill().bfill()
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    print(f"\n📊 Prepared DataFrame: {len(df):,} bars × {len(df.columns)} columns")
    print(f"   Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"   Columns: {', '.join(df.columns[:15])}...")

    return df


def run_optimization(args: argparse.Namespace) -> None:
    """Запустить оптимизацию."""
    print("=" * 70)
    print("🔬 STRATEGY LAB — Self-Improving Trading Strategy")
    print("=" * 70)

    # ── 1. Load data ──
    print(f"\n📡 Loading {args.years}y data for {args.symbol}...")
    start = time.time()
    data = load_all_data(symbol=args.symbol, years=args.years, use_cache=True)
    load_time = time.time() - start
    print(f"   ⏱️ Data loaded in {load_time:.0f}s")

    # ── 2. Prepare DataFrame ──
    df = prepare_dataframe(data)

    # ── 3. Run optimizer ──
    optimizer = StrategyOptimizer(
        df=df,
        target_monthly_return=args.target,
        max_drawdown_limit=args.max_dd,
        min_trades_per_split=args.min_trades,
        n_walk_forward_splits=args.splits,
        train_ratio=args.train_ratio,
        n_jobs=args.jobs,
    )

    best_strategies = optimizer.run(
        n_rounds=args.rounds,
        trials_per_round=args.trials,
        patience=args.patience,
        indicator_mode=args.indicator_mode,
        mutation_rate=args.mutation_rate,
    )

    # ── 4. Validate and report top strategies ──
    if best_strategies:
        print(f"\n{'='*70}")
        print("📊 FINAL VALIDATION — Top Strategies")
        print(f"{'='*70}")

        for i, strat in enumerate(best_strategies[:3]):
            print(f"\n── Strategy #{i+1} ──")
            if "config" not in strat:
                print("   ⚠️ No config saved")
                continue

            config = load_strategy_from_json_dict(strat["config"])
            config.name = f"strategy_{i+1}"

            result = optimizer.validate_strategy(config)
            generate_report(result, config, prefix=f"strategy_{i+1}")

            print(f"   Return:  {result.total_return_pct:+.2f}%")
            print(f"   Monthly: {result.avg_monthly_return:+.2f}%")
            print(f"   DD:      {result.max_drawdown_pct:.2f}%")
            print(f"   Sharpe:  {result.sharpe_ratio:.2f}")
            print(f"   Trades:  {result.total_trades}")

        # Optimization report
        generate_optimization_report(optimizer.history, best_strategies)

    print(f"\n✅ Done! Results saved to: {OUTPUT_DIR}")


def run_validation(args: argparse.Namespace) -> None:
    """Валидировать сохранённую стратегию."""
    print(f"\n📋 Validating strategy from: {args.validate}")

    config = load_strategy_from_json(args.validate)

    print(f"\n📡 Loading {args.years}y data...")
    data = load_all_data(symbol=args.symbol, years=args.years, use_cache=True)
    df = prepare_dataframe(data)

    print(f"\n🔄 Running backtest...")
    signals = generate_signals(df, config)
    exit_sigs = generate_exit_signals(df, config) if config.exit_indicators else None
    result = run_backtest(df, signals, config, exit_signals=exit_sigs)

    generate_report(result, config, prefix="validated")

    print(f"\n{'='*50}")
    print(f"Total Return:       {result.total_return_pct:+.2f}%")
    print(f"Avg Monthly Return: {result.avg_monthly_return:+.2f}%")
    print(f"Max Drawdown:       {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"Win Rate:           {result.win_rate:.1f}%")
    print(f"Total Trades:       {result.total_trades}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")
    print(f"{'='*50}")


def load_strategy_from_json_dict(cfg: dict) -> StrategyConfig:
    """Загрузить StrategyConfig из dict."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy Lab — Self-Improving Trading Strategy Optimizer"
    )

    # Data params
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")

    # Optimization params
    parser.add_argument("--target", type=float, default=30.0, help="Target monthly return %% (default: 30)")
    parser.add_argument("--max-dd", type=float, default=-60.0, help="Max drawdown limit %% (default: -60)")
    parser.add_argument("--rounds", type=int, default=10, help="Optimization rounds (default: 10)")
    parser.add_argument("--trials", type=int, default=200, help="Trials per round (default: 200)")
    parser.add_argument("--patience", type=int, default=3, help="Patience before expanding search (default: 3)")
    parser.add_argument("--splits", type=int, default=5, help="Walk-Forward splits (default: 5)")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Train portion ratio (default: 0.5)")
    parser.add_argument("--min-trades", type=int, default=50, help="Min trades to consider (default: 50)")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Optuna workers (default: 1)")

    # Indicator auto-permutation
    parser.add_argument("--indicator-mode", type=str, default="all",
                        choices=["all", "groups", "custom", "meta_rotate", "evolutionary", "importance_guided"],
                        help="Indicator selection mode (default: all)")
    parser.add_argument("--mutation-rate", type=float, default=0.3,
                        help="Mutation rate for evolutionary/importance_guided mode (default: 0.3)")

    # Fast mode
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 2y data, 3 rounds, 50 trials")

    # Validation mode
    parser.add_argument("--validate", type=str, default=None,
                        help="Path to strategy JSON to validate")

    args = parser.parse_args()

    # Fast mode overrides
    if args.fast:
        args.years = 2
        args.rounds = 3
        args.trials = 50
        args.splits = 2
        args.min_trades = 20

    if args.validate:
        run_validation(args)
    else:
        run_optimization(args)


if __name__ == "__main__":
    main()
