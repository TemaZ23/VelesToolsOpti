"""
Дневные свечи спота Binance (data.binance.vision, месячные архивы 1d) → cache/cycle_lab/<SYMBOL>.npz.

    PYTHONPATH=. python3 -m scripts.cycle_lab.data BTC ETH SOL ...
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "cache" / "cycle_lab"
URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1d/{s}-1d-{m}.zip"


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(symbol: str, start: str = "2017-08", end: str | None = None) -> Path:
    if end is None:
        t = date.today()
        end = f"{t.year:04d}-{t.month - 1:02d}" if t.month > 1 else f"{t.year - 1:04d}-12"
    rows = []
    for m in months(start, end):
        try:
            with urllib.request.urlopen(URL.format(s=symbol, m=m), timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text = zf.read(zf.namelist()[0]).decode()
        if not text[:1].isdigit():
            text = text[text.find("\n") + 1 :]
        arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4, 7), ndmin=2)
        if len(arr) and arr[0, 0] > 1e14:  # микросекунды в архивах с 2025 года
            arr[:, 0] = np.floor(arr[:, 0] / 1000)
        rows.append(arr)
    data = np.concatenate(rows)
    data = data[np.argsort(data[:, 0])]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{symbol}.npz"
    np.savez(out, ts=data[:, 0].astype(np.int64), ohlc=data[:, 1:5], qvol=data[:, 5])
    return out


def load(symbol: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"{symbol}.npz") as z:
        return {k: z[k] for k in z.files}



FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"


def fetch_funding(symbol: str, start: str = "2019-09", end: str | None = None) -> Path:
    """Ставки фандинга фьючерса USDT-M → дневная сумма ставок (доля номинала, + платят лонги)."""
    if end is None:
        t = date.today()
        end = f"{t.year:04d}-{t.month - 1:02d}" if t.month > 1 else f"{t.year - 1:04d}-12"
    rows = []
    for m in months(start, end):
        try:
            with urllib.request.urlopen(FUNDING_URL.format(s=symbol, m=m), timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text = zf.read(zf.namelist()[0]).decode()
        if not text[:1].isdigit():
            text = text[text.find("\n") + 1 :]
        arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 2), ndmin=2)
        if len(arr) and arr[0, 0] > 1e14:
            arr[:, 0] = np.floor(arr[:, 0] / 1000)
        rows.append(arr)
    data = np.concatenate(rows)
    day = (data[:, 0] // 86_400_000).astype(np.int64) * 86_400_000
    days, inv = np.unique(day, return_inverse=True)
    rate = np.zeros(len(days))
    np.add.at(rate, inv, data[:, 1])
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{symbol}_funding.npz"
    np.savez(out, ts=days, rate=rate)
    return out


def load_funding(symbol: str) -> dict[str, np.ndarray] | None:
    path = CACHE / f"{symbol}_funding.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    for coin in sys.argv[1:]:
        fetch(f"{coin}USDT")
        fetch_funding(f"{coin}USDT")
        d = load(f"{coin}USDT")
        f = load_funding(f"{coin}USDT")
        print(coin, np.datetime64(int(d["ts"][0]), "ms"), "→", np.datetime64(int(d["ts"][-1]), "ms"), len(d["ts"]), "дней;",
              "фандинг с", np.datetime64(int(f["ts"][0]), "ms"), f"средний {f['rate'].mean() * 365 * 100:.1f}%/год")
