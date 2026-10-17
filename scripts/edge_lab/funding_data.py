"""
Данные для семейства funding (scripts/edge_lab/funding.py).

    PYTHONPATH=. python3 -m scripts.edge_lab.funding_data spot     # дневной спот Binance для фьючерсов, у которых есть спот
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_data mark     # дневные mark price klines (для ликвидаций)
    PYTHONPATH=. python3 -m scripts.edge_lab.funding_data build    # выровненные панели (спот + фандинг со сдвигом)

Кеш: cache/edge_lab/funding/spot_1d/<PERP>.npz (ts, ohlc, qvol, scale) — НЕ пересекается с файлами
ставок cache/edge_lab/funding/<SYM>.npz; производные панели: cache/edge_lab/funding/derived/.

Соглашение по фандингу (важно для отсутствия заглядывания):
- panel()["funding"][t] — выплаты с меткой времени в [open_t, open_{t+1}) (так строит common.py);
- fund_hold[t] (строится здесь) — выплаты в (open_t, open_{t+1}], т.е. то, что получает/платит позиция,
  открытая СРАЗУ ПОСЛЕ выплаты в open_t и закрытая сразу после выплаты в open_{t+1}. Позиция, вошедшая
  по open[t+1], не получает выплату с меткой open[t+1] — консервативно для стратегий «заходим ради фандинга».
- сигнал на баре t строится по panel funding[<=t] (выплата ровно в open_{t+1} в сигнал НЕ входит).
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from scripts.edge_lab.common import CACHE, DAY, load_funding, panel, universe
from scripts.edge_lab.universe import BASE, LAST_MONTH, _get, list_prefix

FAM = CACHE / "funding"
SPOT = FAM / "spot_1d"
MARK = FAM / "mark_1d"  # дневные свечи mark price (по ней Binance ликвидирует) для перпов со спотом
DER = FAM / "derived"
PREFIXES = ("1000000", "100000", "10000", "1000", "1M")


def spot_map() -> dict[str, tuple[str, float]]:
    """perp symbol -> (spot pair, scale), где цена перпа ≈ scale * цена спота."""
    _, dirs = list_prefix("data/spot/monthly/klines/")
    spot = {d.rstrip("/").split("/")[-1] for d in dirs}
    out = {}
    for s in universe():
        if s in spot:
            out[s] = (s, 1.0)
            continue
        for pre in PREFIXES:
            if s.startswith(pre) and s[len(pre):] in spot:
                out[s] = (s[len(pre):], 1e6 if pre == "1M" else float(pre))
                break
    return out


def _read(blob: bytes) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    if text and not text[:1].isdigit():
        text = text[text.find("\n") + 1:]
    if not text.strip():
        return np.empty((0, 6))
    a = np.loadtxt(io.StringIO(text), delimiter=",", usecols=(0, 1, 2, 3, 4, 7), ndmin=2)
    if len(a) and a[0, 0] > 1e14:
        a[:, 0] = np.floor(a[:, 0] / 1000)
    return a


def fetch_spot(perp: str, pair: str, scale: float) -> dict:
    path = SPOT / f"{perp}.npz"
    if path.exists():
        return {"perp": perp, "cached": True}
    keys, _ = list_prefix(f"data/spot/monthly/klines/{pair}/1d/", delimiter=False)
    months = sorted({m for k in keys for m in re.findall(r"(\d{4}-\d{2})\.zip$", k)})
    months = [m for m in months if "2019-12" <= m <= LAST_MONTH]
    rows = []
    for m in months:
        blob = _get(f"{BASE}data/spot/monthly/klines/{pair}/1d/{pair}-1d-{m}.zip")
        if blob:
            r = _read(blob)
            if len(r):
                rows.append(r)
    if not rows:
        return {"perp": perp, "error": "no data"}
    a = np.concatenate(rows)
    a = a[np.argsort(a[:, 0], kind="stable")]
    a = a[np.r_[True, a[1:, 0] != a[:-1, 0]]]
    SPOT.mkdir(parents=True, exist_ok=True)
    np.savez(path, ts=a[:, 0].astype(np.int64), ohlc=a[:, 1:5], qvol=a[:, 5], scale=scale, pair=pair)
    return {"perp": perp, "rows": len(a)}


def fetch_mark(perp: str) -> dict:
    path = MARK / f"{perp}.npz"
    if path.exists():
        return {"perp": perp, "cached": True}
    keys, _ = list_prefix(f"data/futures/um/monthly/markPriceKlines/{perp}/1d/", delimiter=False)
    months = sorted({m for k in keys for m in re.findall(r"(\d{4}-\d{2})\.zip$", k)})
    rows = []
    for m in [m for m in months if "2019-12" <= m <= LAST_MONTH]:
        blob = _get(f"{BASE}data/futures/um/monthly/markPriceKlines/{perp}/1d/{perp}-1d-{m}.zip")
        if blob:
            r = _read(blob)
            if len(r):
                rows.append(r)
    if not rows:
        return {"perp": perp, "error": "no data"}
    a = np.concatenate(rows)
    a = a[np.argsort(a[:, 0], kind="stable")]
    a = a[np.r_[True, a[1:, 0] != a[:-1, 0]]]
    MARK.mkdir(parents=True, exist_ok=True)
    np.savez(path, ts=a[:, 0].astype(np.int64), ohlc=a[:, 1:5])
    return {"perp": perp, "rows": len(a)}


def download_mark(threads: int = 12) -> None:
    mp = json.loads((DER / "spot_map.json").read_text())
    res = []
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fetch_mark, k): k for k in mp}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as exc:  # noqa: BLE001
                res.append({"perp": futs[f], "error": repr(exc)})
            if i % 100 == 0:
                print(i, len(mp), flush=True)
    err = [r for r in res if "error" in r]
    print("mark готово, ошибок:", len(err), err[:10])


def download_spot(threads: int = 12) -> None:
    mp = spot_map()
    print("перпов со спотом:", len(mp), flush=True)
    SPOT.mkdir(parents=True, exist_ok=True)
    (FAM / "derived").mkdir(parents=True, exist_ok=True)
    (DER / "spot_map.json").write_text(json.dumps(mp, indent=1))
    res = []
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(fetch_spot, k, v[0], v[1]): k for k, v in mp.items()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as exc:  # noqa: BLE001
                res.append({"perp": futs[f], "error": repr(exc)})
            if i % 50 == 0:
                print(i, len(mp), flush=True)
    err = [r for r in res if "error" in r]
    print("готово, ошибок:", len(err), err[:10])


def build() -> None:
    """Выровненные по panel('1d') массивы: спот OHLC/объём (в единицах перпа) и fund_hold (сдвиг выплат)."""
    p = panel("1d")
    ts, syms = p["ts"], list(p["symbols"])
    T, N = len(ts), len(syms)
    t0 = int(ts[0])
    so, sh, sl, sc, sv = (np.full((T, N), np.nan) for _ in range(5))
    for j, s in enumerate(syms):
        path = SPOT / f"{s}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        b = (d["ts"] - t0) // DAY
        ok = (b >= 0) & (b < T)
        b = b[ok].astype(np.int64)
        sc_ = float(d["scale"])
        so[b, j], sh[b, j], sl[b, j], sc[b, j] = (d["ohlc"][ok, k] * sc_ for k in range(4))
        sv[b, j] = d["qvol"][ok]
    mo, mh = np.full((T, N), np.nan), np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        path = MARK / f"{s}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            mts, mohlc = z["ts"], z["ohlc"]
        b = (mts - t0) // DAY
        ok = (b >= 0) & (b < T)
        mo[b[ok].astype(np.int64), j] = mohlc[ok, 0]
        mh[b[ok].astype(np.int64), j] = mohlc[ok, 1]
    # фандинг «держим бар» — выплаты в (open_t, open_{t+1}]; сдвиг на 1 минуту назад
    fh = np.zeros((T, N))
    nset = np.zeros((T, N), dtype=np.int16)
    for j, s in enumerate(syms):
        f = load_funding(s)
        if not len(f["ts"]):
            continue
        b = (f["ts"] - 60_000 - t0) // DAY
        ok = (b >= 0) & (b < T)
        np.add.at(fh[:, j], b[ok].astype(np.int64), f["rate"][ok])
        np.add.at(nset[:, j], b[ok].astype(np.int64), 1)
    DER.mkdir(parents=True, exist_ok=True)
    np.savez(DER / "aux_1d.npz", ts=ts, spot_open=so, spot_high=sh, spot_low=sl, spot_close=sc, spot_qvol=sv,
             fund_hold=fh, n_settle=nset, mark_open=mo, mark_high=mh)  # fmt: skip
    have = (~np.isnan(sc)).any(0).sum()
    print("aux_1d сохранён; символов со спотом в панели:", int(have))
    # sanity: базис перп/спот
    pc = p["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        basis = pc / sc - 1
    v = basis[np.isfinite(basis)]
    print("базис close перп/спот, перцентили %:", np.round(np.percentile(v * 100, [0.1, 1, 5, 50, 95, 99, 99.9]), 3))


def load_aux() -> dict[str, np.ndarray]:
    with np.load(DER / "aux_1d.npz") as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "spot":
        download_spot()
    elif cmd == "mark":
        download_mark()
    elif cmd == "build":
        build()
