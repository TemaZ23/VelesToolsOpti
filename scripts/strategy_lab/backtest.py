"""
Быстрый векторизованный бэктестер для фьючерсов.

Учитывает:
- Leverage (1x-125x)
- Комиссия maker/taker (0.02%/0.04%)
- Funding Rate (каждые 8ч)
- Stop Loss / Take Profit / Trailing Stop (fixed % OR ATR-based)
- Exit-indicator signals (optional indicator-driven exits)
- Max hold time
- Position sizing (% от баланса)
- Ликвидацию (если margin < maintenance)

Возвращает детальную статистику: P&L, drawdown, Sharpe, monthly returns, trades.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import StrategyConfig


@dataclass
class Trade:
    """Одна сделка."""
    entry_bar: int
    exit_bar: int
    direction: int       # +1 long, -1 short
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: float
    pnl_usd: float
    pnl_pct: float       # % от equity на момент входа
    fees: float
    funding_paid: float
    exit_reason: str      # "signal", "stop_loss", "take_profit", "trailing", "max_hold", "liquidation"
    bars_held: int


@dataclass
class BacktestResult:
    """Результат бэктеста."""
    # Core metrics
    total_return_pct: float
    monthly_returns: list[float]
    avg_monthly_return: float
    min_monthly_return: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_risk_reward: float
    max_consecutive_losses: int

    # Long/Short breakdown
    long_trades: int
    long_win_rate: float
    short_trades: int
    short_win_rate: float

    # Raw data
    equity_curve: pd.Series
    trades: list[Trade]
    monthly_return_series: pd.Series

    # Meta
    total_bars: int
    total_fees: float
    total_funding: float
    time_in_market_pct: float


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

MAKER_FEE = 0.0002     # 0.02%
TAKER_FEE = 0.0004     # 0.04%
MAINTENANCE_MARGIN = 0.005  # 0.5% — liquidation threshold
FUNDING_INTERVAL = 32   # 15m bars = 8 hours


# ═══════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    config: StrategyConfig,
    initial_balance: float = 10000.0,
    exit_signals: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Запустить бэктест.

    Args:
        df: OHLCV DataFrame
        signals: DataFrame с signal (+1/-1/0) и strength
        config: StrategyConfig
        initial_balance: начальный баланс USD
        exit_signals: DataFrame с exit_long / exit_short (optional)

    Returns:
        BacktestResult
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Funding rates (if available)
    funding_rates = df["funding_rate"].values if "funding_rate" in df.columns else np.zeros(n)

    # ── PRE-COMPUTE ATR for dynamic exits ──
    atr_values: Optional[np.ndarray] = None
    if config.exit_mode in ("atr", "hybrid"):
        from .indicators import _atr
        atr_series = _atr(df["high"], df["low"], df["close"], config.atr_period)
        atr_values = atr_series.values

    # ── EXIT INDICATOR SIGNALS ──
    exit_long_arr = np.zeros(n, dtype=bool)
    exit_short_arr = np.zeros(n, dtype=bool)
    if exit_signals is not None and len(exit_signals) == n:
        if "exit_long" in exit_signals.columns:
            exit_long_arr = exit_signals["exit_long"].values.astype(bool)
        if "exit_short" in exit_signals.columns:
            exit_short_arr = exit_signals["exit_short"].values.astype(bool)

    # ── REALISTIC EXECUTION MODEL ──
    # Signal at bar i uses close[i]. We can act on it at bar i+1.
    # Shift signals by 1 bar and enter at open[i+1] (= open of next bar).
    raw_sig = signals["signal"].values
    sig = np.zeros(n, dtype=raw_sig.dtype)
    sig[1:] = raw_sig[:-1]  # act on PREVIOUS bar's signal

    open_prices = df["open"].values
    SLIPPAGE_PCT = 0.0003  # 0.03% additional slippage

    # State
    balance = initial_balance
    equity_curve = np.full(n, initial_balance)
    trades: list[Trade] = []

    # Current position state
    in_position = False
    pos_direction = 0
    pos_entry_price = 0.0
    pos_size_usd = 0.0
    pos_entry_bar = 0
    trailing_high = 0.0
    trailing_low = float("inf")
    pos_entry_balance = 0.0
    pos_sl_price = 0.0   # dynamic SL price (set at entry)
    pos_tp_price = 0.0   # dynamic TP price (set at entry)
    pos_trail_dist = 0.0  # trailing distance (fixed or ATR-based)

    for i in range(1, n):
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]

        # ── UPDATE POSITION ──
        if in_position:
            # Check stop loss / take profit / trailing / max hold / liquidation
            exit_reason = ""
            exit_price = current_close

            # Update trailing extremes
            if pos_direction == 1:
                trailing_high = max(trailing_high, current_high)
            else:
                trailing_low = min(trailing_low, current_low)

            # Unrealized P&L
            if pos_direction == 1:
                unreal_pnl_pct = (current_close - pos_entry_price) / pos_entry_price * config.leverage
            else:
                unreal_pnl_pct = (pos_entry_price - current_close) / pos_entry_price * config.leverage

            # Liquidation check
            liq_pct = -1.0 / config.leverage + MAINTENANCE_MARGIN
            if unreal_pnl_pct <= liq_pct:
                exit_reason = "liquidation"
                exit_price = current_close

            # Stop Loss (dynamic: SL price computed at entry)
            if not exit_reason and pos_sl_price > 0:
                if pos_direction == 1:
                    if current_low <= pos_sl_price:
                        exit_reason = "stop_loss"
                        exit_price = pos_sl_price
                else:
                    if current_high >= pos_sl_price:
                        exit_reason = "stop_loss"
                        exit_price = pos_sl_price

            # Take Profit (dynamic: TP price computed at entry)
            if not exit_reason and pos_tp_price > 0:
                if pos_direction == 1:
                    if current_high >= pos_tp_price:
                        exit_reason = "take_profit"
                        exit_price = pos_tp_price
                else:
                    if current_low <= pos_tp_price:
                        exit_reason = "take_profit"
                        exit_price = pos_tp_price

            # Trailing Stop (dynamic distance)
            if not exit_reason and pos_trail_dist > 0:
                if pos_direction == 1:
                    trail_price = trailing_high - pos_trail_dist
                    if current_low <= trail_price and trailing_high > pos_entry_price:
                        exit_reason = "trailing"
                        exit_price = trail_price
                else:
                    trail_price = trailing_low + pos_trail_dist
                    if current_high >= trail_price and trailing_low < pos_entry_price:
                        exit_reason = "trailing"
                        exit_price = trail_price

            # Exit indicator signal
            if not exit_reason:
                if pos_direction == 1 and exit_long_arr[i]:
                    exit_reason = "exit_indicator"
                elif pos_direction == -1 and exit_short_arr[i]:
                    exit_reason = "exit_indicator"

            # Max Hold
            if not exit_reason and config.max_hold_bars > 0:
                if (i - pos_entry_bar) >= config.max_hold_bars:
                    exit_reason = "max_hold"

            # Signal flip
            if not exit_reason and config.exit_on_signal_flip:
                if sig[i] != 0 and sig[i] != pos_direction:
                    exit_reason = "signal"

            # Funding payment (every 8h)
            funding_cost = 0.0
            if (i - pos_entry_bar) > 0 and i % FUNDING_INTERVAL == 0:
                fr = funding_rates[i] if i < len(funding_rates) else 0.0
                # Long pays funding when rate > 0, short pays when rate < 0
                funding_cost = pos_size_usd * fr * pos_direction
                balance -= funding_cost

            # ── CLOSE POSITION ──
            if exit_reason:
                if exit_reason == "liquidation":
                    pnl_usd = -pos_size_usd / config.leverage  # lose margin
                else:
                    if pos_direction == 1:
                        raw_pnl = (exit_price - pos_entry_price) / pos_entry_price
                    else:
                        raw_pnl = (pos_entry_price - exit_price) / pos_entry_price
                    pnl_usd = pos_size_usd * raw_pnl

                # Exit fee
                exit_fee = pos_size_usd * TAKER_FEE
                pnl_usd -= exit_fee

                balance += pnl_usd
                pnl_pct = pnl_usd / pos_entry_balance * 100

                trades.append(Trade(
                    entry_bar=pos_entry_bar,
                    exit_bar=i,
                    direction=pos_direction,
                    entry_price=pos_entry_price,
                    exit_price=exit_price,
                    size_usd=pos_size_usd,
                    leverage=config.leverage,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    fees=pos_size_usd * TAKER_FEE * 2,  # entry + exit
                    funding_paid=funding_cost,
                    exit_reason=exit_reason,
                    bars_held=i - pos_entry_bar,
                ))

                in_position = False
                pos_direction = 0

                # If signal flip, open new position immediately below
                if exit_reason == "signal" and sig[i] != 0:
                    pass  # will be handled in OPEN section
                else:
                    equity_curve[i] = balance
                    continue

        # ── OPEN POSITION ──
        if not in_position and sig[i] != 0 and balance > 10:
            pos_direction = int(sig[i])
            # Enter at this bar's open + slippage
            base_price = open_prices[i]
            if pos_direction == 1:
                pos_entry_price = base_price * (1 + SLIPPAGE_PCT)
            else:
                pos_entry_price = base_price * (1 - SLIPPAGE_PCT)
            pos_entry_bar = i
            pos_entry_balance = balance

            # Position sizing
            margin = balance * config.risk_per_trade_pct / 100
            pos_size_usd = margin * config.leverage

            # Entry fee
            entry_fee = pos_size_usd * TAKER_FEE
            balance -= entry_fee

            # ── Compute SL/TP/Trailing prices at entry ──
            if config.exit_mode == "atr" and atr_values is not None:
                # ATR-based dynamic exits
                cur_atr = atr_values[i] if not np.isnan(atr_values[i]) else pos_entry_price * 0.01
                if pos_direction == 1:
                    pos_sl_price = pos_entry_price - cur_atr * config.atr_sl_mult if config.atr_sl_mult > 0 else 0.0
                    pos_tp_price = pos_entry_price + cur_atr * config.atr_tp_mult if config.atr_tp_mult > 0 else 0.0
                else:
                    pos_sl_price = pos_entry_price + cur_atr * config.atr_sl_mult if config.atr_sl_mult > 0 else 0.0
                    pos_tp_price = pos_entry_price - cur_atr * config.atr_tp_mult if config.atr_tp_mult > 0 else 0.0
                pos_trail_dist = cur_atr * config.atr_trailing_mult if config.atr_trailing_mult > 0 else 0.0

            elif config.exit_mode == "hybrid" and atr_values is not None:
                # Hybrid: use min(fixed%, ATR-based) for SL, max(fixed%, ATR-based) for TP
                cur_atr = atr_values[i] if not np.isnan(atr_values[i]) else pos_entry_price * 0.01
                fixed_sl_dist = pos_entry_price * config.stop_loss_pct / 100
                atr_sl_dist = cur_atr * config.atr_sl_mult
                sl_dist = min(fixed_sl_dist, atr_sl_dist) if config.atr_sl_mult > 0 else fixed_sl_dist

                fixed_tp_dist = pos_entry_price * config.take_profit_pct / 100
                atr_tp_dist = cur_atr * config.atr_tp_mult
                tp_dist = max(fixed_tp_dist, atr_tp_dist) if config.atr_tp_mult > 0 else fixed_tp_dist

                if pos_direction == 1:
                    pos_sl_price = pos_entry_price - sl_dist if config.stop_loss_pct > 0 else 0.0
                    pos_tp_price = pos_entry_price + tp_dist if config.take_profit_pct > 0 else 0.0
                else:
                    pos_sl_price = pos_entry_price + sl_dist if config.stop_loss_pct > 0 else 0.0
                    pos_tp_price = pos_entry_price - tp_dist if config.take_profit_pct > 0 else 0.0

                fixed_trail_dist = pos_entry_price * config.trailing_stop_pct / 100
                atr_trail_dist = cur_atr * config.atr_trailing_mult
                pos_trail_dist = min(fixed_trail_dist, atr_trail_dist) if config.atr_trailing_mult > 0 else fixed_trail_dist

            else:
                # Fixed % mode (original)
                if pos_direction == 1:
                    pos_sl_price = pos_entry_price * (1 - config.stop_loss_pct / 100) if config.stop_loss_pct > 0 else 0.0
                    pos_tp_price = pos_entry_price * (1 + config.take_profit_pct / 100) if config.take_profit_pct > 0 else 0.0
                else:
                    pos_sl_price = pos_entry_price * (1 + config.stop_loss_pct / 100) if config.stop_loss_pct > 0 else 0.0
                    pos_tp_price = pos_entry_price * (1 - config.take_profit_pct / 100) if config.take_profit_pct > 0 else 0.0
                pos_trail_dist = pos_entry_price * config.trailing_stop_pct / 100 if config.trailing_stop_pct > 0 else 0.0

            trailing_high = current_high
            trailing_low = current_low
            in_position = True

        equity_curve[i] = balance + (
            _unrealized_pnl(pos_entry_price, current_close, pos_size_usd, pos_direction, config.leverage)
            if in_position else 0
        )

    # Close any remaining position at end
    if in_position:
        final_price = close[-1]
        if pos_direction == 1:
            raw_pnl = (final_price - pos_entry_price) / pos_entry_price
        else:
            raw_pnl = (pos_entry_price - final_price) / pos_entry_price
        pnl_usd = pos_size_usd * raw_pnl - pos_size_usd * TAKER_FEE
        balance += pnl_usd

    equity_series = pd.Series(equity_curve, index=df.index)

    return _compute_stats(equity_series, trades, initial_balance, n, df)


def _unrealized_pnl(entry: float, current: float, size: float, direction: int, leverage: float) -> float:
    if direction == 1:
        return size * (current - entry) / entry
    else:
        return size * (entry - current) / entry


# ═══════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def _compute_stats(
    equity: pd.Series,
    trades: list[Trade],
    initial_balance: float,
    total_bars: int,
    df: pd.DataFrame,
) -> BacktestResult:
    """Вычислить все метрики."""

    # Monthly returns
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        equity_ts = pd.Series(equity.values, index=ts)
        monthly = equity_ts.resample("ME").last()
        monthly_returns_raw = monthly.pct_change().dropna()
        monthly_returns = (monthly_returns_raw * 100).tolist()
        monthly_return_series = monthly_returns_raw * 100
    else:
        # Approximate: 96 bars/day * 30 = 2880 bars/month
        bars_per_month = 2880
        monthly_equity = equity.iloc[::bars_per_month]
        monthly_returns_raw = monthly_equity.pct_change().dropna()
        monthly_returns = (monthly_returns_raw * 100).tolist()
        monthly_return_series = monthly_returns_raw * 100

    # Core metrics
    total_return = (equity.iloc[-1] / initial_balance - 1) * 100
    avg_monthly = np.mean(monthly_returns) if monthly_returns else 0
    min_monthly = np.min(monthly_returns) if monthly_returns else 0

    # Drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe (monthly, annualized)
    if len(monthly_returns) > 1:
        monthly_std = np.std(monthly_returns)
        sharpe = (avg_monthly / monthly_std * np.sqrt(12)) if monthly_std > 0 else 0
    else:
        sharpe = 0

    # Sortino
    downside = [r for r in monthly_returns if r < 0]
    if downside:
        downside_std = np.std(downside)
        sortino = (avg_monthly / downside_std * np.sqrt(12)) if downside_std > 0 else 0
    else:
        sortino = float("inf") if avg_monthly > 0 else 0

    # Calmar
    calmar = (avg_monthly * 12 / abs(max_dd)) if max_dd != 0 else 0

    # Trade stats
    n_trades = len(trades)
    winners = [t for t in trades if t.pnl_usd > 0]
    losers = [t for t in trades if t.pnl_usd <= 0]
    n_win = len(winners)
    n_loss = len(losers)
    win_rate = n_win / n_trades * 100 if n_trades > 0 else 0

    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0

    gross_profit = sum(t.pnl_usd for t in winners)
    gross_loss = abs(sum(t.pnl_usd for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Max consecutive losses
    max_consec_loss = 0
    current_consec = 0
    for t in trades:
        if t.pnl_usd <= 0:
            current_consec += 1
            max_consec_loss = max(max_consec_loss, current_consec)
        else:
            current_consec = 0

    # Long/Short breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    long_wr = sum(1 for t in longs if t.pnl_usd > 0) / len(longs) * 100 if longs else 0
    short_wr = sum(1 for t in shorts if t.pnl_usd > 0) / len(shorts) * 100 if shorts else 0

    # Time in market
    bars_in_market = sum(t.bars_held for t in trades)
    time_in_market = bars_in_market / total_bars * 100

    # Total fees/funding
    total_fees = sum(t.fees for t in trades)
    total_funding = sum(t.funding_paid for t in trades)

    return BacktestResult(
        total_return_pct=total_return,
        monthly_returns=monthly_returns,
        avg_monthly_return=avg_monthly,
        min_monthly_return=min_monthly,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=n_trades,
        winning_trades=n_win,
        losing_trades=n_loss,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        avg_risk_reward=avg_rr,
        max_consecutive_losses=max_consec_loss,
        long_trades=len(longs),
        long_win_rate=long_wr,
        short_trades=len(shorts),
        short_win_rate=short_wr,
        equity_curve=equity,
        trades=trades,
        monthly_return_series=monthly_return_series,
        total_bars=total_bars,
        total_fees=total_fees,
        total_funding=total_funding,
        time_in_market_pct=time_in_market,
    )
