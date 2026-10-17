"""
seasonality — дополнительная независимая выборка «до IS»: спот Binance BTCUSDT/ETHUSDT, 1h,
2017-08 … 2019-12 (data.binance.vision, месячные архивы) → cache/edge_lab/seasonality/spot1h/<SYM>.npz.
Используется ТОЛЬКО для диагностики ячеек (выбор параметров — по IS 2020–2023).

    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality_spot
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from scripts.edge_lab.common import CACHE
from scripts.edge_lab.universe import _get

SDIR = CACHE / "seasonality" / "spot1h"
URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip"


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(sym: str) -> None:
    rows = []
    for m in months("2017-08", "2019-12"):
        data = _get(URL.format(s=sym, m=m))
        if data is None:
            continue
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text = zf.read(zf.namelist()[0]).decode()
        if not text[:1].isdigit():
            text = text[text.find("\n") + 1 :]
        arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4, 7), ndmin=2)
        rows.append(arr)
    d = np.concatenate(rows)
    d = d[np.argsort(d[:, 0])]
    SDIR.mkdir(parents=True, exist_ok=True)
    np.savez(SDIR / f"{sym}.npz", ts=d[:, 0].astype(np.int64), ohlc=d[:, 1:5], qvol=d[:, 5])
    print(sym, len(d), d[0, 0], d[-1, 0], flush=True)


def load(sym: str) -> dict[str, np.ndarray]:
    with np.load(SDIR / f"{sym}.npz") as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    for s in ("BTCUSDT", "ETHUSDT"):
        fetch(s)
