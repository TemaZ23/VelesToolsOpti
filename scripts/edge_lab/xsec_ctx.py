"""
xsec: контекст суточной кросс-секции (признаки, вселенная) — всё рассчитывается только по прошлым данным.
"""

from __future__ import annotations

import numpy as np

from scripts.edge_lab.xsec_data import crypto_mask, load_daily
from scripts.edge_lab.xsec_engine import make_exec


def roll_sum(x: np.ndarray, n: int):
    """Скользящие сумма и число валидных значений за n баров, включая текущий (NaN игнорируются)."""
    v = np.nan_to_num(x)
    c = (~np.isnan(x)).astype(float)
    z = np.zeros((1, x.shape[1]))
    cs = np.cumsum(np.vstack([z, v]), axis=0)
    cc = np.cumsum(np.vstack([z, c]), axis=0)
    s = np.full(x.shape, np.nan)
    k = np.zeros(x.shape)
    s[n - 1 :] = cs[n:] - cs[:-n]
    k[n - 1 :] = cc[n:] - cc[:-n]
    return s, k


def roll_mean(x, n, min_cnt=None):
    s, k = roll_sum(x, n)
    min_cnt = min_cnt or max(2, int(0.7 * n))
    with np.errstate(all="ignore"):
        return np.where(k >= min_cnt, s / np.maximum(k, 1), np.nan)


def roll_std(x, n, min_cnt=None):
    m = roll_mean(x, n, min_cnt)
    m2 = roll_mean(x * x, n, min_cnt)
    with np.errstate(all="ignore"):
        return np.sqrt(np.maximum(m2 - m * m, 0))


def roll_beta(x, y, n, min_cnt=None):
    """β of x[:, i] on y (vector) over n bars, pairwise valid."""
    yy = np.where(np.isnan(x), np.nan, y[:, None])
    mx = roll_mean(x, n, min_cnt)
    my = roll_mean(yy, n, min_cnt)
    mxy = roll_mean(x * yy, n, min_cnt)
    myy = roll_mean(yy * yy, n, min_cnt)
    with np.errstate(all="ignore"):
        return (mxy - mx * my) / (myy - my * my)


def roll_max(x, n):
    """Скользящий максимум за n баров (NaN игнорируются; NaN, если валидных < n/2)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view

    xv = np.where(np.isnan(x), -np.inf, x)
    sw = sliding_window_view(xv, n, axis=0)  # [T-n+1, N, n]
    mx = sw.max(axis=2)
    _, k = roll_sum(x, n)
    out[n - 1 :] = np.where(k[n - 1 :] >= n / 2, mx, np.nan)
    return out


class Ctx:
    """Суточная панель + производные признаки на смещении off (часы от 00:00 UTC)."""

    def __init__(self, off: int = 0):
        d = load_daily(off)
        self.d = d
        self.ts = d["ts"]
        self.symbols = d["symbols"]
        self.O, self.C, self.QV = d["open"], d["close"], d["qvol"]
        self.T, self.N = self.C.shape
        self.alive = ~np.isnan(self.C)
        self.crypto = crypto_mask(self.symbols)
        self.age = np.cumsum(self.alive, axis=0)
        self.j_btc = int(np.where(self.symbols == "BTCUSDT")[0][0])
        self.j_eth = int(np.where(self.symbols == "ETHUSDT")[0][0])
        with np.errstate(all="ignore"):
            self.r1 = self.C / np.vstack([np.full((1, self.N), np.nan), self.C[:-1]]) - 1.0
            self.lc = np.log(self.C)
        # ADV: среднее за 30 прошлых дней включая текущий (как liquid_mask в common)
        s, k = roll_sum(self.QV, 30)
        with np.errstate(all="ignore"):
            self.adv = np.where(k > 0, s / np.maximum(k, 1), np.nan)
        self.ex = make_exec(d, self.adv)
        self._cache: dict = {}

    # --- cached features
    def vol(self, n: int = 30):
        key = ("vol", n)
        if key not in self._cache:
            self._cache[key] = roll_std(self.r1, n)
        return self._cache[key]

    def beta(self, n: int = 60):
        key = ("beta", n)
        if key not in self._cache:
            self._cache[key] = roll_beta(self.r1, self.r1[:, self.j_btc], n)
        return self._cache[key]

    def mom(self, L: int, skip: int = 0):
        key = ("mom", L, skip)
        if key not in self._cache:
            lc = self.lc
            a = np.full_like(lc, np.nan)
            a[L + skip :] = lc[L : self.T - skip] - lc[: self.T - L - skip]  # a[t] = lc[t-skip] - lc[t-skip-L]
            self._cache[key] = a
        return self._cache[key]

    def universe(self, min_adv: float = 1e7, min_age: int = 60, top: int = 0):
        key = ("uni", min_adv, min_age, top)
        if key not in self._cache:
            u = self.alive & self.crypto & (self.age >= min_age) & (np.nan_to_num(self.adv) >= min_adv)
            if top > 0:
                a = np.where(u, self.adv, -np.inf)
                rank = np.argsort(np.argsort(-a, axis=1), axis=1)
                u = u & (rank < top)
            self._cache[key] = u
        return self._cache[key]
