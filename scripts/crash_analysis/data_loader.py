"""
Модуль загрузки данных с Binance API с кешированием на диск.

Загружает:
- OHLCV (15m) — Futures + Spot
- Funding Rate
- Open Interest
- Long/Short Ratio
- Taker Buy/Sell Volume
- Fear & Greed Index (alternative.me)
- Multi-timeframe: 15m, 1h, 4h, 1d

Кеширование:
- Парquet-файлы в ./cache/
- Инкрементальная загрузка (только новые данные)
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_FUTURES = "https://fapi.binance.com"
BINANCE_SPOT = "https://api.binance.com"
FEAR_GREED_API = "https://api.alternative.me/fng/"

KLINES_LIMIT = 1500
FUNDING_LIMIT = 1000
OI_LIMIT = 500
REQUEST_DELAY = 0.15  # seconds

CACHE_DIR = Path("cache")


def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _safe_get(url: str, params: dict | None = None, retries: int = 5) -> dict | list:
    """GET-запрос с retry и exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(60, 2 ** (attempt + 2))
                print(f"      ⏳ Rate limit, ждём {wait}с...")
                time.sleep(wait)
                continue
            if resp.status_code == 418:
                # IP ban — wait 2 minutes
                print("      🚫 IP ban, ждём 120с...")
                time.sleep(120)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"      ⚠️  Ошибка: {e}, повтор через {wait}с...")
            time.sleep(wait)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# KLINES (OHLCV)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    market: str = "futures",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Загрузить klines с Binance."""
    base = BINANCE_FUTURES if market == "futures" else BINANCE_SPOT
    endpoint = "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"

    all_rows: list[dict] = []
    current = _ts_ms(start)
    end_ms = _ts_ms(end)
    total_ms = end_ms - current

    while current < end_ms:
        data = _safe_get(
            f"{base}{endpoint}",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": KLINES_LIMIT,
            },
        )
        if not data:
            break

        for k in data:
            taker_buy_vol = float(k[9])
            total_vol = float(k[5])
            all_rows.append({
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": total_vol,
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": taker_buy_vol,
                "taker_sell_volume": total_vol - taker_buy_vol,
                "taker_buy_quote": float(k[10]),
            })

        current = data[-1][0] + 1

        if on_progress:
            pct = min(100, int((current - _ts_ms(start)) / total_ms * 100))
            on_progress(len(all_rows), pct)

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDING RATE
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_funding_rates(
    symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Загрузить исторические funding rates."""
    all_rows: list[dict] = []
    current = _ts_ms(start)
    end_ms = _ts_ms(end)

    while current < end_ms:
        data = _safe_get(
            f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": current,
                "endTime": end_ms,
                "limit": FUNDING_LIMIT,
            },
        )
        if not data:
            break

        for fr in data:
            all_rows.append({
                "timestamp": fr["fundingTime"],
                "funding_rate": float(fr["fundingRate"]),
            })

        current = data[-1]["fundingTime"] + 1
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# OPEN INTEREST (limited to ~30 days via public API)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_open_interest(
    symbol: str, interval: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Загрузить open interest (ограничена ~30 днями)."""
    all_rows: list[dict] = []
    current = _ts_ms(start)
    end_ms = _ts_ms(end)

    while current < end_ms:
        try:
            data = _safe_get(
                f"{BINANCE_FUTURES}/futures/data/openInterestHist",
                params={
                    "symbol": symbol,
                    "period": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": OI_LIMIT,
                },
            )
        except Exception:
            break

        if not data:
            break

        for oi in data:
            all_rows.append({
                "timestamp": oi["timestamp"],
                "open_interest": float(oi["sumOpenInterest"]),
                "open_interest_value": float(oi["sumOpenInterestValue"]),
            })

        current = data[-1]["timestamp"] + 1
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LONG/SHORT RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_long_short_ratio(
    symbol: str, interval: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Top Trader Long/Short Ratio."""
    all_rows: list[dict] = []
    current = _ts_ms(start)
    end_ms = _ts_ms(end)

    while current < end_ms:
        try:
            data = _safe_get(
                f"{BINANCE_FUTURES}/futures/data/topLongShortAccountRatio",
                params={
                    "symbol": symbol,
                    "period": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": OI_LIMIT,
                },
            )
        except Exception:
            break
        if not data:
            break

        for item in data:
            all_rows.append({
                "timestamp": item["timestamp"],
                "long_short_ratio": float(item["longShortRatio"]),
                "long_account_pct": float(item["longAccount"]),
                "short_account_pct": float(item["shortAccount"]),
            })

        current = data[-1]["timestamp"] + 1
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TAKER BUY/SELL VOLUME (отдельный endpoint с агрегацией)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_taker_volume(
    symbol: str, interval: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Taker Buy/Sell Volume ratio."""
    all_rows: list[dict] = []
    current = _ts_ms(start)
    end_ms = _ts_ms(end)

    while current < end_ms:
        try:
            data = _safe_get(
                f"{BINANCE_FUTURES}/futures/data/takerlongshortRatio",
                params={
                    "symbol": symbol,
                    "period": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": OI_LIMIT,
                },
            )
        except Exception:
            break
        if not data:
            break

        for item in data:
            all_rows.append({
                "timestamp": item["timestamp"],
                "buy_sell_ratio": float(item["buySellRatio"]),
                "buy_vol": float(item["buyVol"]),
                "sell_vol": float(item["sellVol"]),
            })

        current = data[-1]["timestamp"] + 1
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fear_greed_index(limit: int = 2000) -> pd.DataFrame:
    """Fear & Greed Index от alternative.me."""
    try:
        data = _safe_get(FEAR_GREED_API, params={"limit": limit, "format": "json"})
        rows = [
            {
                "timestamp": int(item["timestamp"]),
                "fear_greed_value": int(item["value"]),
                "fear_greed_class": item["value_classification"],
            }
            for item in data.get("data", [])
        ]
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"      ⚠️  Fear & Greed unavailable: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER (with cache)
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def _load_cached(name: str) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if path.exists():
        df = pd.read_parquet(path)
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        print(f"   💾 Кеш {name}: {len(df):,} строк (возраст: {age_hours:.1f}ч)")
        return df
    return None


def _save_cache(name: str, df: pd.DataFrame) -> None:
    path = _cache_path(name)
    df.to_parquet(path, index=False)


def load_all_data(
    symbol: str = "BTCUSDT",
    years: int = 5,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Загрузить все данные для анализа.

    Returns:
        dict с ключами:
        - futures_15m: OHLCV фьючерсы 15m
        - futures_1h:  OHLCV фьючерсы 1h
        - futures_4h:  OHLCV фьючерсы 4h
        - futures_1d:  OHLCV фьючерсы 1d
        - spot_15m:    OHLCV спот 15m
        - funding:     Funding rate
        - open_interest: Open interest
        - long_short:  Long/Short ratio
        - taker_volume: Taker buy/sell
        - fear_greed:  Fear & Greed index
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)

    result: dict[str, pd.DataFrame] = {}

    # ── OHLCV multi-timeframe ──────────────────────────────────────────
    for interval in ["15m", "1h", "4h", "1d"]:
        cache_key = f"{symbol}_futures_{interval}"
        if use_cache and not force_refresh:
            cached = _load_cached(cache_key)
            if cached is not None and len(cached) > 0:
                result[f"futures_{interval}"] = cached
                continue

        print(f"   📡 Загрузка {symbol} Futures {interval}...")
        df = fetch_klines(
            symbol, interval, start, end, market="futures",
            on_progress=lambda n, pct: print(f"      {n:,} баров ({pct}%)", end="\r"),
        )
        print(f"      ✅ {len(df):,} баров")
        _save_cache(cache_key, df)
        result[f"futures_{interval}"] = df

    # ── Spot 15m (для Basis) ──────────────────────────────────────────
    cache_key = f"{symbol}_spot_15m"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None and len(cached) > 0:
            result["spot_15m"] = cached
        else:
            print(f"   📡 Загрузка {symbol} Spot 15m...")
            df = fetch_klines(
                symbol, "15m", start, end, market="spot",
                on_progress=lambda n, pct: print(f"      {n:,} баров ({pct}%)", end="\r"),
            )
            print(f"      ✅ {len(df):,} баров")
            _save_cache(cache_key, df)
            result["spot_15m"] = df
    else:
        print(f"   📡 Загрузка {symbol} Spot 15m...")
        df = fetch_klines(
            symbol, "15m", start, end, market="spot",
            on_progress=lambda n, pct: print(f"      {n:,} баров ({pct}%)", end="\r"),
        )
        print(f"      ✅ {len(df):,} баров")
        _save_cache(cache_key, df)
        result["spot_15m"] = df

    # ── Funding Rate ──────────────────────────────────────────────────
    cache_key = f"{symbol}_funding"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            result["funding"] = cached
        else:
            print(f"   📡 Загрузка Funding Rate...")
            result["funding"] = fetch_funding_rates(symbol, start, end)
            print(f"      ✅ {len(result['funding']):,} записей")
            _save_cache(cache_key, result["funding"])
    else:
        print(f"   📡 Загрузка Funding Rate...")
        result["funding"] = fetch_funding_rates(symbol, start, end)
        print(f"      ✅ {len(result['funding']):,} записей")
        _save_cache(cache_key, result["funding"])

    # Start for data-API endpoints (OI, LS, Taker) is limited to ~30 days
    data_api_start = end - timedelta(days=29)

    # ── Open Interest ──────────────────────────────────────────────────
    cache_key = f"{symbol}_oi"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            result["open_interest"] = cached
        else:
            print(f"   📡 Загрузка Open Interest (last 30d)...")
            result["open_interest"] = fetch_open_interest(symbol, "5m", data_api_start, end)
            print(f"      ✅ {len(result['open_interest']):,} записей")
            _save_cache(cache_key, result["open_interest"])
    else:
        print(f"   📡 Загрузка Open Interest (last 30d)...")
        result["open_interest"] = fetch_open_interest(symbol, "5m", data_api_start, end)
        print(f"      ✅ {len(result['open_interest']):,} записей")
        _save_cache(cache_key, result["open_interest"])

    # ── Long/Short Ratio ──────────────────────────────────────────────
    cache_key = f"{symbol}_ls_ratio"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            result["long_short"] = cached
        else:
            print(f"   📡 Загрузка Long/Short Ratio (last 30d)...")
            result["long_short"] = fetch_long_short_ratio(symbol, "5m", data_api_start, end)
            print(f"      ✅ {len(result['long_short']):,} записей")
            _save_cache(cache_key, result["long_short"])
    else:
        print(f"   📡 Загрузка Long/Short Ratio (last 30d)...")
        result["long_short"] = fetch_long_short_ratio(symbol, "5m", data_api_start, end)
        print(f"      ✅ {len(result['long_short']):,} записей")
        _save_cache(cache_key, result["long_short"])

    # ── Taker Volume ──────────────────────────────────────────────────
    cache_key = f"{symbol}_taker_vol"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            result["taker_volume"] = cached
        else:
            print(f"   📡 Загрузка Taker Volume Ratio (last 30d)...")
            result["taker_volume"] = fetch_taker_volume(symbol, "5m", data_api_start, end)
            print(f"      ✅ {len(result['taker_volume']):,} записей")
            _save_cache(cache_key, result["taker_volume"])
    else:
        print(f"   📡 Загрузка Taker Volume Ratio (last 30d)...")
        result["taker_volume"] = fetch_taker_volume(symbol, "5m", data_api_start, end)
        print(f"      ✅ {len(result['taker_volume']):,} записей")
        _save_cache(cache_key, result["taker_volume"])

    # ── Fear & Greed ──────────────────────────────────────────────────
    cache_key = "fear_greed"
    if use_cache and not force_refresh:
        cached = _load_cached(cache_key)
        if cached is not None:
            result["fear_greed"] = cached
        else:
            print(f"   📡 Загрузка Fear & Greed Index...")
            result["fear_greed"] = fetch_fear_greed_index()
            print(f"      ✅ {len(result['fear_greed']):,} записей")
            if len(result["fear_greed"]) > 0:
                _save_cache(cache_key, result["fear_greed"])
    else:
        print(f"   📡 Загрузка Fear & Greed Index...")
        result["fear_greed"] = fetch_fear_greed_index()
        print(f"      ✅ {len(result['fear_greed']):,} записей")
        if len(result["fear_greed"]) > 0:
            _save_cache(cache_key, result["fear_greed"])

    return result
