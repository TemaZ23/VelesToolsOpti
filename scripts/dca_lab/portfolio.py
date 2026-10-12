"""
Рыночные данные для движка и запуск портфельной симуляции.

Market — все монеты в общей минутной шкале (глобальная минута 0 = GLOBAL_T0, UTC).
Серии индикаторов считаются один раз и кешируются в cache/dca_lab/series/.
Загрузка двухпроходная: сначала кеш серий по монетам, затем заполнение заранее
выделенных массивов — без двойного пикового потребления памяти.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import engine as E
from .data import CACHE_DIR, coin_to_symbol, load_symbol
from .indicators import SERIES_MENU, SERIES_NAMES, TF_MINUTES, build_series
from .strategy import BotConfig, PortfolioConfig, Sleeve, pack_global, pack_sleeve, sleeve_from_portfolio

GLOBAL_T0 = datetime(2021, 1, 1, tzinfo=timezone.utc)
GLOBAL_T0_MS = int(GLOBAL_T0.timestamp() * 1000)
DAY_MS = 86_400_000
SERIES_DIR = CACHE_DIR / "series"
SERIES_VERSION = 6


def minute_of(date_str: str) -> int:
    """'YYYY-MM-DD' → глобальная минута."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int((dt.timestamp() * 1000 - GLOBAL_T0_MS) // 60_000)


def date_of(minute: int) -> str:
    return datetime.fromtimestamp((GLOBAL_T0_MS + minute * 60_000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class Market:
    coins: list[str]
    series_names: list[str]
    px: np.ndarray  # float32[N, 3] high, low, close
    coin_off: np.ndarray
    coin_len: np.ndarray
    coin_start: np.ndarray
    ser: np.ndarray
    ser_off: np.ndarray
    ser_len: np.ndarray
    ser_tf: np.ndarray
    t_end: int = 0

    @property
    def n_coins(self) -> int:
        return len(self.coins)

    @property
    def series_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.series_names)}

    def mask(self, coins: list[str] | None = None) -> np.ndarray:
        m = np.zeros(self.n_coins, dtype=np.uint8)
        if coins is None:
            m[:] = 1
        else:
            wanted = set(coins)
            for i, c in enumerate(self.coins):
                if c in wanted:
                    m[i] = 1
        return m


def _series_cache_path(symbol: str, tz_amp: float) -> Path:
    return SERIES_DIR / f"{symbol}_v{SERIES_VERSION}_tz{tz_amp:g}.npz"


def _aligned(symbol: str):
    """Минутки монеты, обрезанные до первых полных суток UTC и не раньше GLOBAL_T0."""
    d = load_symbol(symbol)
    ts, ohlc, qvol = d["ts"], d["ohlc"], d["qvol"]
    first_full_day = ((ts[0] + DAY_MS - 1) // DAY_MS) * DAY_MS
    cut = int((first_full_day - ts[0]) // 60_000)
    ts, ohlc, qvol = ts[cut:], ohlc[cut:], qvol[cut:]
    start = int((ts[0] - GLOBAL_T0_MS) // 60_000) if len(ts) else 0
    if start < 0:
        ts, ohlc, qvol = ts[-start:], ohlc[-start:], qvol[-start:]
        start = 0
    return start, ohlc, qvol


def _ensure_series(symbol: str, tz_amp: float) -> tuple[int, int] | None:
    """Строит кеш серий монеты (если нет). Возвращает (start, n_minutes)."""
    if not (CACHE_DIR / f"{symbol}.npz").exists():
        return None
    start, ohlc, qvol = _aligned(symbol)
    n = len(ohlc)
    if n < 3 * 1440:
        return None
    path = _series_cache_path(symbol, tz_amp)
    if not path.exists():
        series = build_series(ohlc, qvol, tz_amp)
        SERIES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **series)
    return start, n


def load_market(
    coins: list[str],
    series: list[str] | None = None,
    tz_amp: float = 5.6,
    verbose: bool = True,
    workers: int = 3,
) -> Market:
    """Загружает монеты из кеша. series — подмножество меню серий (экономия памяти)."""
    names = list(series) if series else list(SERIES_NAMES)
    unknown = [n for n in names if n not in SERIES_MENU]
    if unknown:
        raise KeyError(unknown)
    tf = np.array([TF_MINUTES[SERIES_MENU[n]] for n in names], dtype=np.int64)
    # недостающие кеши серий строим параллельно в отдельных процессах
    missing = [
        coin_to_symbol(c)
        for c in coins
        if (CACHE_DIR / f"{coin_to_symbol(c)}.npz").exists()
        and not _series_cache_path(coin_to_symbol(c), tz_amp).exists()
    ]
    if missing:
        t = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_ensure_series, missing, [tz_amp] * len(missing)))
        if verbose:
            print(f"  построены серии для {len(missing)} монет ({time.time() - t:.0f}s)", flush=True)
    meta: list[tuple[str, str, int, int]] = []
    for coin in coins:
        symbol = coin_to_symbol(coin)
        t = time.time()
        info = _ensure_series(symbol, tz_amp)
        if info is None:
            if verbose:
                print(f"  {symbol}: нет данных, пропуск")
            continue
        meta.append((coin, symbol, info[0], info[1]))
        if verbose:
            print(f"  {symbol}: {info[1] / 1440:.0f} дн. с {date_of(info[0])} ({time.time() - t:.1f}s)", flush=True)

    total_rows = sum(m[3] for m in meta)
    ser_lens = np.array([[m[3] // k for k in tf] for m in meta], dtype=np.int64)
    ser_total = int(ser_lens.sum())
    px = np.empty((total_rows, 3), dtype=np.float32)
    ser = np.empty(ser_total, dtype=np.float32)
    coin_off = np.zeros(len(meta), dtype=np.int64)
    coin_len = np.zeros(len(meta), dtype=np.int64)
    coin_start = np.zeros(len(meta), dtype=np.int64)
    ser_off = np.zeros((len(meta), len(names)), dtype=np.int64)
    row = 0
    pos = 0
    t_end = 0
    for ci, (_coin, symbol, start, n) in enumerate(meta):
        _, ohlc, _ = _aligned(symbol)
        px[row : row + n] = ohlc[:, 1:4]
        coin_off[ci], coin_len[ci], coin_start[ci] = row, n, start
        row += n
        with np.load(_series_cache_path(symbol, tz_amp)) as z:
            for si, name in enumerate(names):
                arr = z[name]
                ln = ser_lens[ci, si]
                ser[pos : pos + ln] = arr[:ln]
                ser_off[ci, si] = pos
                pos += ln
        t_end = max(t_end, start + n)
    return Market(
        coins=[m[0] for m in meta],
        series_names=names,
        px=px,
        coin_off=coin_off,
        coin_len=coin_len,
        coin_start=coin_start,
        ser=ser,
        ser_off=ser_off,
        ser_len=ser_lens,
        ser_tf=tf,
        t_end=t_end,
    )


# ═══════════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТЫ
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Result:
    stats: np.ndarray
    daily: np.ndarray
    sstats: np.ndarray  # [S, N_SSTATS]
    deposit: float
    t_from: int
    t_to: int
    coin_pnl: np.ndarray | None = None  # [S, C]
    deals: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def months(self) -> float:
        return (self.t_to - self.t_from) / 1440 / 30.4375

    @property
    def liquidated(self) -> bool:
        return bool(self.stats[E.S_LIQUIDATED])

    @property
    def profit_pct(self) -> float:
        return (self.stats[E.S_FINAL_EQ] - self.deposit) / self.deposit * 100

    @property
    def avg_month_pct(self) -> float:
        """Средняя доходность в месяц, % начального депозита (простая, как у автора стратегии)."""
        return self.profit_pct / self.months

    @property
    def cmgr_pct(self) -> float:
        """Среднемесячный сложный рост эквити."""
        fe = self.stats[E.S_FINAL_EQ]
        if fe <= 0:
            return -100.0
        return ((fe / self.deposit) ** (1 / self.months) - 1) * 100

    @property
    def max_dd_pct(self) -> float:
        """Максимальная просадка от пика эквити по худшим ценам минут, % текущего пика."""
        return self.stats[E.S_MAX_DD_LOW] * 100

    @property
    def dd_abs_pct(self) -> float:
        """Максимальная просадка в % НАЧАЛЬНОГО депозита — линейно масштабируется размером бота."""
        return self.stats[E.S_MAX_DD_ABS] / self.deposit * 100

    @property
    def efficiency(self) -> float:
        """Среднемесячная доходность (% депозита) на 1% просадки от депозита."""
        return self.avg_month_pct / max(self.dd_abs_pct, 1.0)

    @property
    def mpu_pct(self) -> float:
        return self.stats[E.S_MPU] / self.deposit * 100

    def monthly_returns(self) -> list[tuple[str, float]]:
        """Доходность по календарным месяцам, % от эквити на начало месяца."""
        out: list[tuple[str, float]] = []
        eq_prev = self.deposit
        cur = None
        last_eq = self.deposit
        for d in range(self.daily.shape[0]):
            label = date_of(self.t_from + d * 1440)[:7]
            if cur is None:
                cur = label
            if label != cur:
                out.append((cur, (last_eq / eq_prev - 1) * 100 if eq_prev > 0 else -100.0))
                eq_prev = last_eq
                cur = label
            last_eq = self.daily[d, 0]
        if cur is not None:
            out.append((cur, (last_eq / eq_prev - 1) * 100 if eq_prev > 0 else -100.0))
        return out

    def concurrent_mae_pct(self) -> float:
        """«Суммарный МПУ» как в Veles Tools: максимум суммы MAE одновременно открытых сделок, % депозита."""
        if not self.deals:
            return float("nan")
        t0, t1, mae = self.deals["t0"], self.deals["t1"], -self.deals["mae"]
        ev_t = np.concatenate([t0, t1])
        ev_v = np.concatenate([mae, -mae])
        order = np.lexsort((-ev_v, ev_t))  # при равном времени сначала открытия
        running = np.cumsum(ev_v[order])
        return float(running.max(initial=0.0) / self.deposit * 100)

    def summary(self) -> dict[str, float]:
        s = self.stats
        mr = [r for _, r in self.monthly_returns()]
        wins, losses = s[E.S_SUM_WIN], s[E.S_SUM_LOSS]
        return {
            "months": round(self.months, 1),
            "profit_pct": round(self.profit_pct, 2),
            "avg_month_pct": round(self.avg_month_pct, 2),
            "cmgr_pct": round(self.cmgr_pct, 2),
            "worst_month_pct": round(min(mr), 2) if mr else 0.0,
            "neg_months": int(sum(1 for r in mr if r < 0)),
            "max_dd_pct": round(self.max_dd_pct, 2),
            "dd_abs_pct": round(self.dd_abs_pct, 2),
            "eff": round(self.efficiency, 4),
            "mpu_pct": round(self.mpu_pct, 2),
            "liquidated": int(self.liquidated),
            "deals": int(s[E.S_DEALS]),
            "tp": int(s[E.S_TP]),
            "sl": int(s[E.S_SL]),
            "full_grid": int(s[E.S_FULL_GRID]),
            "pf": round(wins / losses, 2) if losses > 0 else float("inf"),
            "max_conc": int(s[E.S_MAX_CONC]),
            "max_notional_x": round(s[E.S_MAX_NOTIONAL] / self.deposit, 2),
            "avg_deal_h": round(s[E.S_AVG_DEAL_MIN] / 60, 1),
            "fees_pct": round(s[E.S_FEES] / self.deposit * 100, 2),
            "funding_pct": round(s[E.S_FUNDING] / self.deposit * 100, 2),
            "open_upnl_pct": round(s[E.S_OPEN_UPNL] / self.deposit * 100, 2),
            "open_deals_end": int(s[E.S_OPEN_DEALS_END]),
            "sleeve_pnl_pct": [round(x / self.deposit * 100, 2) for x in self.sstats[:, E.SS_REALIZED]],
        }


def _stack_sleeves(market: Market, sleeves: list[Sleeve], n_slots: int) -> dict[str, np.ndarray]:
    idx = market.series_index
    packs = [pack_sleeve(s, idx) for s in sleeves]
    masks = [market.mask(s.coins) for s in sleeves]
    while len(packs) < n_slots:  # пустые слоты (маска нулевая)
        packs.append(packs[0])
        masks.append(np.zeros(market.n_coins, dtype=np.uint8))
    out = {k: np.ascontiguousarray(np.stack([p[k] for p in packs])) for k in packs[0]}
    out["masks"] = np.ascontiguousarray(np.stack(masks))
    return out


def run_sleeves(
    market: Market,
    sleeves: list[Sleeve],
    pf: PortfolioConfig,
    t_from: int,
    t_to: int,
    record_deals: bool = False,
) -> Result:
    t_to = min(t_to, market.t_end)
    st = _stack_sleeves(market, sleeves, len(sleeves))
    res = E.simulate(
        market.px,
        market.coin_off,
        market.coin_len,
        market.coin_start,
        st["masks"],
        market.ser,
        market.ser_off,
        market.ser_len,
        market.ser_tf,
        st["cser"],
        st["cop"],
        st["cthr"],
        st["indent"],
        st["volume"],
        st["fparams"],
        st["iparams"],
        pack_global(pf),
        t_from,
        t_to,
        3_000_000 if record_deals else 0,
    )
    deals = {}
    if record_deals:
        keys = ["strat", "coin", "t0", "t1", "nf", "pnl", "reason", "maxnot", "mae"]
        deals = dict(zip(keys, res[4:]))
    return Result(res[0], res[1], res[3], pf.deposit, t_from, t_to, res[2], deals)


def run(
    market: Market,
    bot: BotConfig,
    pf: PortfolioConfig,
    t_from: int,
    t_to: int,
    coins: list[str] | None = None,
    record_deals: bool = False,
) -> Result:
    """Одиночный набор ботов: размер и блокировка берутся из PortfolioConfig."""
    return run_sleeves(market, [sleeve_from_portfolio(bot, pf, coins)], pf, t_from, t_to, record_deals)


def run_batch(
    market: Market,
    configs: list[tuple[list[Sleeve], PortfolioConfig]],
    t_from: int,
    t_to: int,
) -> list[Result]:
    """Параллельный прогон набора портфелей (numba prange)."""
    t_to = min(t_to, market.t_end)
    n_slots = max(len(s) for s, _ in configs)
    stacks = [_stack_sleeves(market, s, n_slots) for s, _ in configs]
    b = {k: np.ascontiguousarray(np.stack([st[k] for st in stacks])) for k in stacks[0]}
    gp = np.ascontiguousarray(np.stack([pack_global(pf) for _, pf in configs]))
    stats, daily, sstats = E.simulate_batch(
        market.px,
        market.coin_off,
        market.coin_len,
        market.coin_start,
        b["masks"],
        market.ser,
        market.ser_off,
        market.ser_len,
        market.ser_tf,
        b["cser"],
        b["cop"],
        b["cthr"],
        b["indent"],
        b["volume"],
        b["fparams"],
        b["iparams"],
        gp,
        t_from,
        t_to,
    )
    return [
        Result(stats[i], daily[i], sstats[i][: len(configs[i][0])], configs[i][1].deposit, t_from, t_to)
        for i in range(len(configs))
    ]


def run_batch_single(
    market: Market,
    configs: list[tuple[BotConfig, PortfolioConfig, list[str] | None]],
    t_from: int,
    t_to: int,
) -> list[Result]:
    """Совместимость: батч одиночных ботов (bot, pf, coins)."""
    return run_batch(market, [([sleeve_from_portfolio(b, p, c)], p) for b, p, c in configs], t_from, t_to)
