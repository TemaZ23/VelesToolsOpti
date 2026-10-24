"""
Загрузка минутных свечей USDⓈ-M фьючерсов Binance с data.binance.vision.

Месячные архивы кешируются в `cache/dca_lab/<SYMBOL>.npz`:
    ts    — int64, время открытия свечи (мс, UTC)
    ohlc  — float64[N, 4]
    qvol  — float64[N], номинальный объём в USDT (quote volume)

Пропуски в данных заполняются плоскими свечами с нулевым объёмом,
чтобы индекс минуты однозначно соответствовал времени.
"""

from __future__ import annotations

import argparse
import http.client
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache" / "dca_lab"
RAW_DIR = CACHE_DIR / "raw"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
SPOT_URL = "https://data.binance.vision/data/spot/monthly/klines"
SPOT_DIR = CACHE_DIR / "spot"
MINUTE_MS = 60_000

# Список «WORK» из LONG_STRATEGY (Bybit). Тикеры Binance отличаются для SHIB.
WORK_COINS = (
    "1INCH AAVE ADA AKT ALGO ARB ATOM AVAX BAT BSV CAKE CFX CHZ COMP CRO CRV CVX DASH DOGE DOT "
    "EGLD ENS ETC FIL GALA GAS GLM GRAM GRT HBAR ICP IMX INJ IOTA JASMY JTO JUP KAS LDO LINK MANA "
    "MNT NEAR NEO ONDO OP PENDLE POL PYTH RSR S SAND SEI SHIB1000 STX TAO THETA TIA TWT UNI VET "
    "XLM XMR XRP XTZ"
).split()

# Расширенный список «STRESS-TEST» автора (152 монеты, включая «шлак» для стресс-теста)
STRESS_COINS = (
    "1000000BABYDOGE 1000BONK 1000FLOKI 1000LUNC 1000PEPE 1INCH AAVE ACH ADA AGLD AIOZ AKT ALGO ALICE ANKR APE "
    "APT AR ARB ARK ARKM ASTR ATH ATOM AUCTION AVAX AXL AXS BAND BAT BCH BEAM BICO BIGTIME BLUR BNB BSV C98 CAKE "
    "CELO CETUS CFX CHR CHZ CKB COMP COTI CRO CRV CTC CVC CVX CYBER DASH DEXE DOGE DOT DYDX DYM EGLD ENJ ENS ETC "
    "FIDA FIL FLOW FLR GALA GAS GLM GRAM GRT HBAR ICP ICX ID IMX INJ IOTA JASMY JTO JUP KAS KNC KSM LDO LINK LPT "
    "LRC LSK MAGIC MANA MANTA MBOX MINA MNT MOVR NEAR NEO NMR ONDO ONG OP ORDI PENDLE PEOPLE PIXEL POL PYTH QNT "
    "QTUM RARE ROSE RSR RVN S SAFE SAND SEI SHIB1000 SKL SLP SNX SSV STG STORJ STX SUPER SUSHI TAO THETA TIA TWT "
    "UMA UNI VET WAVES WAXP WIF WLD WOO XLM XMR XRP XTZ XVG YFI YGG ZEC ZEN ZIL ZRX"
).split()

BINANCE_ALIASES = {"SHIB1000": "1000SHIB", "1000000BABYDOGE": "1000000BABYDOGE"}


def coin_to_symbol(coin: str) -> str:
    return f"{BINANCE_ALIASES.get(coin, coin)}USDT"


def month_range(start: str, end: str) -> list[str]:
    """Список месяцев 'YYYY-MM' включительно."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _download(url: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            time.sleep(2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
            time.sleep(2 ** (attempt + 1))
    return False


def _parse_zip(path: Path) -> np.ndarray:
    """CSV kline → массив [ts, o, h, l, c, quote_volume]."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    text = raw.decode()
    first_line_end = text.find("\n")
    if not text[:1].isdigit():  # заголовок есть в новых архивах
        text = text[first_line_end + 1 :]
    arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4, 7), dtype=np.float64, ndmin=2)
    # Спотовые архивы с 2025 года пишут время в микросекундах.
    if len(arr) and arr[0, 0] > 1e14:
        arr[:, 0] = np.floor(arr[:, 0] / 1000.0)
    return arr


def build_symbol_cache(
    symbol: str, months: list[str], workers: int = 6, extend: bool = False, market: str = "futures"
) -> Path | None:
    """
    Скачивает месячные архивы и собирает единый npz для символа.
    extend=True — дописывает к существующему кешу только недостающие более ранние месяцы.
    market="spot" — спот Binance, кеш в `cache/dca_lab/spot/`.
    """
    cache_dir = SPOT_DIR if market == "spot" else CACHE_DIR
    base_url = SPOT_URL if market == "spot" else BASE_URL
    raw_dir = RAW_DIR / "spot" if market == "spot" else RAW_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{symbol}.npz"
    existing: np.ndarray | None = None
    if extend and out_path.exists():
        with np.load(out_path) as z:
            first_month = np.datetime64(int(z["ts"][0]), "ms").astype("datetime64[M]").astype(str)
            existing = np.column_stack([z["ts"].astype(np.float64), z["ohlc"], z["qvol"]])
        months = [m for m in months if m < first_month]
        if not months:
            return out_path
    zips: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for month in months:
            fname = f"{symbol}-1m-{month}.zip"
            dest = raw_dir / symbol / fname
            futures[pool.submit(_download, f"{base_url}/{symbol}/1m/{fname}", dest)] = dest
        for fut in as_completed(futures):
            if fut.result():
                zips.append(futures[fut])
    if not zips:
        return out_path if existing is not None else None
    parts = [_parse_zip(p) for p in sorted(zips)]
    if existing is not None:
        parts.append(existing)
    data = np.concatenate(parts)
    data = data[np.argsort(data[:, 0], kind="stable")]
    ts = data[:, 0].astype(np.int64)
    keep = np.concatenate([[True], np.diff(ts) > 0])
    data, ts = data[keep], ts[keep]

    # Сплошная минутная сетка
    t0, t1 = ts[0], ts[-1]
    n = int((t1 - t0) // MINUTE_MS) + 1
    idx = ((ts - t0) // MINUTE_MS).astype(np.int64)
    ohlc = np.full((n, 4), np.nan)
    qvol = np.zeros(n)
    ohlc[idx] = data[:, 1:5]
    qvol[idx] = data[:, 5]
    # forward-fill пропусков плоской свечой по последнему close
    close = ohlc[:, 3]
    mask = np.isnan(close)
    if mask.any():
        last = np.where(~mask, np.arange(n), 0)
        np.maximum.accumulate(last, out=last)
        filled_close = close[last]
        for j in range(4):
            col = ohlc[:, j]
            col[mask] = filled_close[mask]
    full_ts = t0 + np.arange(n, dtype=np.int64) * MINUTE_MS
    np.savez(out_path, ts=full_ts, ohlc=ohlc, qvol=qvol)
    return out_path


def load_symbol(symbol: str, market: str = "futures") -> dict[str, np.ndarray]:
    path = (SPOT_DIR if market == "spot" else CACHE_DIR) / f"{symbol}.npz"
    with np.load(path) as z:
        return {"ts": z["ts"], "ohlc": z["ohlc"], "qvol": z["qvol"]}


def available_symbols() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка 1m свечей Binance Futures для DCA Lab")
    parser.add_argument("--start", default="2023-01")
    today = date.today()
    prev = f"{today.year:04d}-{today.month - 1:02d}" if today.month > 1 else f"{today.year - 1:04d}-12"
    parser.add_argument("--end", default=prev)
    parser.add_argument("--coins", nargs="*", default=WORK_COINS)
    parser.add_argument("--stress", action="store_true", help="скачать расширенный список STRESS-TEST")
    parser.add_argument("--extend", action="store_true", help="дописать более ранние месяцы к уже скачанным монетам")
    parser.add_argument("--keep-raw", action="store_true", help="не удалять zip после сборки")
    parser.add_argument("--market", choices=("futures", "spot"), default="futures", help="рынок Binance")
    args = parser.parse_args()

    months = month_range(args.start, args.end)
    coins = STRESS_COINS if args.stress else args.coins
    for i, coin in enumerate(coins, 1):
        cache_dir = SPOT_DIR if args.market == "spot" else CACHE_DIR
        if (cache_dir / f"{coin_to_symbol(coin)}.npz").exists() and not args.extend:
            continue
        symbol = coin_to_symbol(coin)
        t = time.time()
        path = build_symbol_cache(symbol, months, extend=args.extend, market=args.market)
        status = "нет данных" if path is None else f"{path.stat().st_size / 1e6:.0f} MB"
        print(f"[{i}/{len(coins)}] {symbol}: {status} ({time.time() - t:.0f}s)", flush=True)
        if not args.keep_raw:
            raw = (RAW_DIR / "spot" if args.market == "spot" else RAW_DIR) / symbol
            if raw.exists():
                for f in raw.iterdir():
                    f.unlink()
                raw.rmdir()
    sys.exit(0)


if __name__ == "__main__":
    main()
