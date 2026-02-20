"""
Визуализация и отчёты по результатам оптимизации.

Генерирует:
1. Equity curve
2. Monthly returns heatmap
3. Drawdown chart
4. Trade distribution
5. Optimization progress
6. Текстовый отчёт
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .strategy import StrategyConfig, config_to_dict

OUTPUT_DIR = Path("output/strategy_lab")


def generate_report(
    result: BacktestResult,
    config: StrategyConfig,
    save_dir: Optional[Path] = None,
    prefix: str = "best",
) -> None:
    """Сгенерировать полный отчёт: графики + текст."""
    out = save_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    _plot_equity_curve(result, out, prefix)
    _plot_monthly_returns(result, out, prefix)
    _plot_drawdown(result, out, prefix)
    _plot_trade_distribution(result, out, prefix)
    _write_text_report(result, config, out, prefix)


def generate_optimization_report(
    history: list[dict],
    best_strategies: list[dict],
    save_dir: Optional[Path] = None,
) -> None:
    """Отчёт по процессу оптимизации."""
    out = save_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    _plot_optimization_progress(history, out)
    _write_optimization_summary(history, best_strategies, out)


# ═══════════════════════════════════════════════════════════════════════
# EQUITY CURVE
# ═══════════════════════════════════════════════════════════════════════

def _plot_equity_curve(result: BacktestResult, out: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    eq = result.equity_curve

    ax.plot(eq.index, eq.values, color="#2196F3", linewidth=1.0, label="Equity")

    # Mark trades
    for t in result.trades:
        color = "#4CAF50" if t.pnl_usd > 0 else "#F44336"
        marker = "^" if t.direction == 1 else "v"
        ax.plot(t.entry_bar, eq.iloc[min(t.entry_bar, len(eq)-1)],
                marker=marker, color=color, markersize=3, alpha=0.5)

    ax.set_title(f"Equity Curve — {result.total_return_pct:+.1f}%", fontsize=14)
    ax.set_ylabel("Balance ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Fill the area under the curve
    ax.fill_between(eq.index, eq.values, eq.values[0], alpha=0.1, color="#2196F3")

    plt.tight_layout()
    fig.savefig(out / f"{prefix}_equity.png", dpi=150)
    plt.close(fig)
    print(f"   📊 {prefix}_equity.png")


# ═══════════════════════════════════════════════════════════════════════
# MONTHLY RETURNS
# ═══════════════════════════════════════════════════════════════════════

def _plot_monthly_returns(result: BacktestResult, out: Path, prefix: str) -> None:
    rets = result.monthly_return_series

    if len(rets) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in rets.values]
    axes[0].bar(range(len(rets)), rets.values, color=colors, alpha=0.8)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axhline(np.mean(rets.values), color="#FF9800", linewidth=1.5,
                    linestyle="--", label=f"Avg: {np.mean(rets.values):.1f}%")
    axes[0].set_title("Monthly Returns (%)", fontsize=12)
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(rets.values, bins=min(20, len(rets)), color="#2196F3",
                 alpha=0.7, edgecolor="white")
    axes[1].axvline(np.mean(rets.values), color="#FF9800", linewidth=2,
                    linestyle="--", label=f"Mean: {np.mean(rets.values):.1f}%")
    axes[1].axvline(np.median(rets.values), color="#9C27B0", linewidth=2,
                    linestyle=":", label=f"Median: {np.median(rets.values):.1f}%")
    axes[1].set_title("Distribution", fontsize=12)
    axes[1].set_xlabel("Return (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / f"{prefix}_monthly.png", dpi=150)
    plt.close(fig)
    print(f"   📊 {prefix}_monthly.png")


# ═══════════════════════════════════════════════════════════════════════
# DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════

def _plot_drawdown(result: BacktestResult, out: Path, prefix: str) -> None:
    eq = result.equity_curve
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax.plot(dd.index, dd.values, color="#F44336", linewidth=0.5)
    ax.set_title(f"Drawdown — Max: {dd.min():.1f}%", fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / f"{prefix}_drawdown.png", dpi=150)
    plt.close(fig)
    print(f"   📊 {prefix}_drawdown.png")


# ═══════════════════════════════════════════════════════════════════════
# TRADE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════

def _plot_trade_distribution(result: BacktestResult, out: Path, prefix: str) -> None:
    if not result.trades:
        return

    pnls = [t.pnl_pct for t in result.trades]
    hold_times = [t.bars_held for t in result.trades]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # PnL distribution
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    axes[0].hist(pnls, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="gray", linewidth=1)
    axes[0].set_title(f"Trade P&L (%) — WR: {result.win_rate:.1f}%", fontsize=11)
    axes[0].set_xlabel("P&L (%)")

    # Hold time
    axes[1].hist(hold_times, bins=30, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[1].set_title("Hold Duration (bars)", fontsize=11)
    axes[1].set_xlabel("Bars")

    # Win/Loss by direction
    long_wins = sum(1 for t in result.trades if t.direction == 1 and t.pnl_usd > 0)
    long_losses = sum(1 for t in result.trades if t.direction == 1 and t.pnl_usd <= 0)
    short_wins = sum(1 for t in result.trades if t.direction == -1 and t.pnl_usd > 0)
    short_losses = sum(1 for t in result.trades if t.direction == -1 and t.pnl_usd <= 0)

    labels = ["Long Win", "Long Loss", "Short Win", "Short Loss"]
    sizes = [long_wins, long_losses, short_wins, short_losses]
    clrs = ["#4CAF50", "#FFCDD2", "#2196F3", "#FFAB91"]
    axes[2].bar(labels, sizes, color=clrs, edgecolor="white")
    axes[2].set_title("Trades by Direction", fontsize=11)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / f"{prefix}_trades.png", dpi=150)
    plt.close(fig)
    print(f"   📊 {prefix}_trades.png")


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZATION PROGRESS
# ═══════════════════════════════════════════════════════════════════════

def _plot_optimization_progress(history: list[dict], out: Path) -> None:
    if not history:
        return

    rounds = [h["round"] for h in history]
    # Support both old (best_score/round_best) and new (best_utility/round_best_utility) keys
    bests = [h.get("best_utility", h.get("best_score", 0)) for h in history]
    round_bests = [h.get("round_best_utility", h.get("round_best", 0)) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: utility progress
    axes[0].plot(rounds, bests, "o-", color="#4CAF50", linewidth=2, markersize=8,
            label="Global Best Utility")
    axes[0].plot(rounds, round_bests, "s--", color="#2196F3", linewidth=1, markersize=6,
            label="Round Best Utility", alpha=0.7)
    axes[0].set_title("Optimization Progress (NSGA-II)", fontsize=14)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Utility Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Pareto front size
    n_pareto = [h.get("n_pareto", 0) for h in history]
    if any(n > 0 for n in n_pareto):
        axes[1].bar(rounds, n_pareto, color="#FF9800", alpha=0.7, edgecolor="white")
        axes[1].set_title("Pareto Front Size per Round", fontsize=14)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("# Pareto Solutions")
        axes[1].grid(True, alpha=0.3)
    else:
        n_cands = [h.get("n_candidates", 0) for h in history]
        axes[1].bar(rounds, n_cands, color="#9C27B0", alpha=0.7, edgecolor="white")
        axes[1].set_title("Candidates per Round", fontsize=14)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("# Candidates")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "optimization_progress.png", dpi=150)
    plt.close(fig)
    print(f"   📊 optimization_progress.png")


# ═══════════════════════════════════════════════════════════════════════
# TEXT REPORTS
# ═══════════════════════════════════════════════════════════════════════

def _write_text_report(
    result: BacktestResult,
    config: StrategyConfig,
    out: Path,
    prefix: str,
) -> None:
    lines = [
        "=" * 60,
        "STRATEGY BACKTEST REPORT",
        "=" * 60,
        "",
        "── PERFORMANCE ──",
        f"Total Return:        {result.total_return_pct:+.2f}%",
        f"Avg Monthly Return:  {result.avg_monthly_return:+.2f}%",
        f"Min Monthly Return:  {result.min_monthly_return:+.2f}%",
        f"Max Drawdown:        {result.max_drawdown_pct:.2f}%",
        f"Sharpe Ratio:        {result.sharpe_ratio:.2f}",
        f"Sortino Ratio:       {result.sortino_ratio:.2f}",
        f"Calmar Ratio:        {result.calmar_ratio:.2f}",
        "",
        "── TRADES ──",
        f"Total Trades:        {result.total_trades}",
        f"Win Rate:            {result.win_rate:.1f}%",
        f"Avg Win:             {result.avg_win_pct:+.2f}%",
        f"Avg Loss:            {result.avg_loss_pct:.2f}%",
        f"Profit Factor:       {result.profit_factor:.2f}",
        f"Risk/Reward:         {result.avg_risk_reward:.2f}",
        f"Max Consec Losses:   {result.max_consecutive_losses}",
        "",
        f"Long:  {result.long_trades} trades, WR {result.long_win_rate:.1f}%",
        f"Short: {result.short_trades} trades, WR {result.short_win_rate:.1f}%",
        "",
        f"Time in Market:      {result.time_in_market_pct:.1f}%",
        f"Total Fees:          ${result.total_fees:.2f}",
        f"Total Funding:       ${result.total_funding:.2f}",
        "",
        "── STRATEGY CONFIG ──",
        f"Leverage:            {config.leverage:.1f}x",
        f"Stop Loss:           {config.stop_loss_pct:.2f}%",
        f"Take Profit:         {config.take_profit_pct:.2f}%",
        f"Trailing Stop:       {config.trailing_stop_pct:.2f}%",
        f"Max Hold:            {config.max_hold_bars} bars",
        f"Risk per Trade:      {config.risk_per_trade_pct:.1f}%",
        f"Combine Mode:        {config.combine_mode} (thr={config.combine_threshold:.2f})",
        "",
        "── INDICATORS ──",
    ]

    for ind in config.entry_indicators:
        lines.append(f"  • {ind['name']}: params={ind.get('params', {})}")
        lines.append(f"    weight={ind.get('weight', 1):.2f}, "
                     f"long_thr={ind.get('long_threshold', 0):.3f}, "
                     f"short_thr={ind.get('short_threshold', 0):.3f}")

    lines.extend([
        "",
        "── MONTHLY RETURNS ──",
    ])
    for i, ret in enumerate(result.monthly_returns):
        bar = "█" * max(1, int(abs(ret)))
        sign = "+" if ret > 0 else ""
        lines.append(f"  Month {i+1:3d}: {sign}{ret:7.2f}% {bar}")

    report_text = "\n".join(lines)
    report_file = out / f"{prefix}_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"   📝 {prefix}_report.txt")


def _write_optimization_summary(
    history: list[dict],
    best_strategies: list[dict],
    out: Path,
) -> None:
    lines = [
        "=" * 60,
        "OPTIMIZATION SUMMARY",
        "=" * 60,
        "",
        "── PROGRESS ──",
    ]

    for h in history:
        lines.append(
            f"  Round {h['round']:3d}: "
            f"best={h['best_score']:+7.2f}%, "
            f"round_best={h['round_best']:+7.2f}%, "
            f"candidates={h['n_candidates']}, "
            f"time={h['time_s']:.0f}s"
        )

    lines.extend(["\n", "── TOP 10 STRATEGIES (ranked by utility) ──"])
    for i, s in enumerate(best_strategies[:10]):
        obj = s.get("objectives", {})
        pareto_mark = " ★" if s.get("pareto_optimal") else ""
        lines.append(
            f"  #{i+1}{pareto_mark}: utility={s.get('score', 0):+.2f} | "
            f"median={obj.get('median_return', s.get('median_monthly_return', 0)):+.1f}%/mo | "
            f"min_split={obj.get('min_split_return', s.get('min_monthly_return', 0)):+.1f}% | "
            f"DD={obj.get('worst_drawdown', s.get('worst_drawdown', 0)):.1f}% | "
            f"Sharpe={s.get('median_sharpe', 0):.2f} | "
            f"WR={s.get('median_wr', 0):.1f}% | "
            f"PF={s.get('median_pf', 0):.2f}"
        )

    # Pareto front summary
    pareto = [s for s in best_strategies if s.get("pareto_optimal")]
    if pareto:
        lines.extend(["\n", f"── PARETO FRONT ({len(pareto)} solutions) ──"])
        by_ret = max(pareto, key=lambda r: r.get("objectives", {}).get("median_return", -999))
        by_dd = max(pareto, key=lambda r: r.get("objectives", {}).get("worst_drawdown", -999))
        by_rob = max(pareto, key=lambda r: r.get("objectives", {}).get("min_split_return", -999))
        for label, strat in [("Best Return", by_ret), ("Lowest Risk", by_dd), ("Most Robust", by_rob)]:
            o = strat.get("objectives", {})
            lines.append(
                f"  {label:14s}: median={o.get('median_return', 0):+.1f}%, "
                f"min_split={o.get('min_split_return', 0):+.1f}%, "
                f"DD={o.get('worst_drawdown', 0):.1f}%"
            )

    summary = "\n".join(lines)
    summary_file = out / "optimization_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"   📝 optimization_summary.txt")
