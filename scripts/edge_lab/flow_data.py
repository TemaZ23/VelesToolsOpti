"""
flow: подготовка данных (один раз).

    PYTHONPATH=. python3 -m scripts.edge_lab.flow_data

Строит cache/edge_lab/flow/:
  trades_1h.npy  float32 [T,N] — число сделок за час, выровнено по panel('1h')['ts'] (NaN — нет данных);
  adv_1h.npy     float32 [T,N] — среднесуточный объём (USDT) за 30 ПОЛНЫХ прошлых суток, известный на
                 момент закрытия часа t (берётся ADV по суткам до вчерашнего дня включительно —
                 без заглядывания вперёд); используется для издержек и фильтров ликвидности.
"""

from __future__ import annotations

import numpy as np

from scripts.edge_lab.common import CACHE, DAY, load_hourly, panel

FCACHE = CACHE / "flow"


def build() -> None:
    FCACHE.mkdir(parents=True, exist_ok=True)
    p1 = panel("1h")
    ts, syms = p1["ts"], p1["symbols"]
    T, N = len(ts), len(syms)
    t0, step = int(ts[0]), int(ts[1] - ts[0])
    tr = np.full((T, N), np.nan, dtype=np.float32)
    for j, s in enumerate(syms):
        d = load_hourly(str(s))
        b = ((d["ts"] - t0) // step).astype(np.int64)
        ok = (b >= 0) & (b < T)
        bars, first = np.unique(b[ok], return_index=True)
        tr[bars, j] = d["trades"][ok][first]
    np.save(FCACHE / "trades_1h.npy", tr)

    # ADV по суточной панели: rolling-среднее qvol за 30 суток (включая сутки d), применяется к часам суток d+1
    p1d = panel("1d")
    qv = p1d["qvol"]
    v = np.nan_to_num(qv)
    cnt = (~np.isnan(qv)).astype(float)
    cs = np.cumsum(np.vstack([np.zeros((1, N)), v]), axis=0)
    cc = np.cumsum(np.vstack([np.zeros((1, N)), cnt]), axis=0)
    n = 30
    adv_d = np.full(qv.shape, np.nan)
    lo = np.maximum(np.arange(1, len(qv) + 1) - n, 0)
    hi = np.arange(1, len(qv) + 1)
    s = cs[hi] - cs[lo]
    k = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        adv_d = np.where(k > 0, s / k, np.nan)
    day_of_hour = (ts - int(p1d["ts"][0])) // DAY  # индекс суток, которым принадлежит час
    src = day_of_hour - 1  # ADV на конец ВЧЕРАШНИХ суток
    adv_h = np.full((T, N), np.nan, dtype=np.float32)
    okd = src >= 0
    adv_h[okd] = adv_d[src[okd]].astype(np.float32)
    np.save(FCACHE / "adv_1h.npy", adv_h)
    print("trades", tr.shape, np.nanmean(tr[:, list(syms).index("BTCUSDT")]), "adv BTC last", adv_h[-1, list(syms).index("BTCUSDT")])


def load_trades() -> np.ndarray:
    return np.load(FCACHE / "trades_1h.npy")


def load_adv() -> np.ndarray:
    return np.load(FCACHE / "adv_1h.npy")


if __name__ == "__main__":
    build()
