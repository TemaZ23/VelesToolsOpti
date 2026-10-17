"""
xsec: подготовка данных для кросс-секционных тестов.

    PYTHONPATH=. python3 -m scripts.edge_lab.xsec_data      # один раз: строит cache/edge_lab/xsec/*.npz

Из часовой панели (загружается один раз) строятся:
  daily_off<k>.npz — суточные бары, начинающиеся в k:00 UTC (k = 0, 1, 8, 16):
      ts (открытие бара), open/close/high/low/qvol/tbqvol [T,N] (float64),
      fund_hold [T,N] — сумма ставок фандинга, которые платит ЛОНГ, удерживаемый с open[t] до open[t+1]
      (выплаты в момент открытия бара относятся к предыдущему периоду удержания: позиция, открытая
      по open[t], не платит выплату в t, но платит выплату в t+1, если держится на этот момент).
  h4.npz — 4-часовые бары с тем же определением fund_hold.

Список не-крипто инструментов (акции, ETF, металлы, индексы, стейблкоины) — EXCLUDE; они исключаются
из кросс-секционной вселенной.
"""

from __future__ import annotations

import numpy as np

from scripts.edge_lab.common import CACHE, DAY, HOUR, panel

XCACHE = CACHE / "xsec"

# Не-крипто перпы Binance (TradFi: акции, ETF, металлы, сырьё, индексы, валюты) и стейблкоины/индексы.
_TRADFI = """
XAU XAG XPT XPD XAUT PAXG COPPER BZ CL NATGAS TSLA HOOD INTC AMZN CRCL MSTR COIN PLTR EWY EWJ EWT EWZ GOOGL
META NVDA AAPL QQQ SPY TSM MU SNDK AVGO BABA MSFT AMD QCOM USAR CSCO DIS HD LITE ORCL UBER MRVL SOXL BRKB CRWV
FLNC JPM V WMT DRAM RKLB CBRS SPCX OPENAI ARM BE COHR NBIS WDC SLX ASTS LLY NOK NVO HYUNDAI SAMSUNG SKHYNIX
ANTHROPIC CRM DELL IBM IREN NOW ONDS AAOI AMAT AXTI BX CRDO CRWD HPE IWM COST EBAY HIMS NFLX URNM DKNG GME RIVN
XLE ZM ADBE ASML BMNR GLW UVXY ALAB CIEN KLAC KORU LRCX SMCI SONY SQQQ TQQQ CAT FLEX STRC TER TTWO TXN FWDI XBI
APP GEV SNOW VRT SKHY MUU SOXS TZA MINIMAX ZHIPU PANW SOFI HK0700 HK1810 TENCENT POPMART BITO TBT TMF GS PYPL
SMH GIGADEV KO RDDT CSOPSAMSUNG2L CSOPSKHYNIX2L KUAISHOU MEITUAN HANMI KODEX200 LGELECTRONICS NAVER SAMSUNGEM
ZHONGJI GDX NET SHOP VST CXMT UNITREE DJT MRNA IONQ MARA MRK PDD TEM HK0625 DDOG GTLB MDB NVDL TEAM TSLL ZS GPRO
BYD HK0992 AMC ANET APLD HUT PATH USDBRL OURA CVNA OKLO RUM TWST XOM ACN CRML MP NKE UNH STXX MVLL SKDD SKUU O
BTCDOM DEFI FOOTBALL BLUEBIRD USDC TUSD BUSD FDUSD USDP EUR GBP AUD
SNXX BSP AGPU BNC BOT SHAZ MOONSHOT INTW KSTR QNTX WEN BBX CYPH RAM PAYP LYTE PENG
"""
# вторая строка снизу — листинги 2026 г. с объёмом в выходные < 30% от будней (признак TradFi-инструмента)
EXCLUDE = {s + "USDT" for s in _TRADFI.split()}


def crypto_mask(symbols: np.ndarray) -> np.ndarray:
    return np.array([s not in EXCLUDE for s in symbols])


def _agg_daily(p1: dict, off: int, step_h: int = 24) -> dict:
    """Агрегирует часовую панель в бары длиной step_h часов, начинающиеся в off:00 UTC."""
    ts1 = p1["ts"]
    T1, N = p1["open"].shape
    start = off  # первый бар начинается в час off от 2020-01-01 00:00
    nb = (T1 - start) // step_h
    idx0 = start + np.arange(nb) * step_h
    sl = slice(start, start + nb * step_h)

    def rs(a):
        return a[sl].reshape(nb, step_h, N)

    o = rs(p1["open"]).astype(np.float64)
    c = rs(p1["close"]).astype(np.float64)
    h = rs(p1["high"]).astype(np.float64)
    lo = rs(p1["low"]).astype(np.float64)
    alive = ~np.isnan(o)
    any_alive = alive.any(axis=1)
    first = np.argmax(alive, axis=1)
    last = step_h - 1 - np.argmax(alive[:, ::-1, :], axis=1)
    ii = np.arange(nb)[:, None]
    jj = np.arange(N)[None, :]
    op = np.where(any_alive, o[ii, first, jj], np.nan)
    cl = np.where(any_alive, c[ii, last, jj], np.nan)
    with np.errstate(all="ignore"):
        hi = np.where(any_alive, np.nanmax(h, axis=1), np.nan)
        lw = np.where(any_alive, np.nanmin(lo, axis=1), np.nan)
    qv = np.where(any_alive, np.nansum(rs(p1["qvol"]).astype(np.float64), axis=1), np.nan)
    tb = np.where(any_alive, np.nansum(rs(p1["taker_buy_qvol"]).astype(np.float64), axis=1), np.nan)
    # funding settled at hour h (+ms) is in hourly bar h; the holding period [open[t], open[t+1])
    # pays settlements at hours (H0, H0+step_h], i.e. hourly bars H0+1 .. H0+step_h.
    f1 = np.nan_to_num(p1["funding"].astype(np.float64))
    f1s = np.vstack([f1[1:], np.zeros((1, N))])  # f1s[h] = funding settled at hour h+1
    fh = f1s[sl].reshape(nb, step_h, N).sum(axis=1)
    return {"ts": ts1[idx0], "open": op, "close": cl, "high": hi, "low": lw, "qvol": qv, "tbqvol": tb,
            "fund_hold": fh, "symbols": p1["symbols"]}  # fmt: skip


def build() -> None:
    XCACHE.mkdir(parents=True, exist_ok=True)
    p1 = panel("1h")
    for off in (0, 1, 8, 16):
        d = _agg_daily(p1, off, 24)
        np.savez(XCACHE / f"daily_off{off}.npz", **d)
        print("daily off", off, d["open"].shape)
    d4 = _agg_daily(p1, 0, 4)
    np.savez(XCACHE / "h4.npz", **d4)
    print("4h", d4["open"].shape)
    # sanity: off0 vs panel_1d
    p = panel("1d")
    d0 = dict(np.load(XCACHE / "daily_off0.npz"))
    n = min(len(p["ts"]), len(d0["ts"]))
    assert (p["ts"][:n] == d0["ts"][:n]).all()
    for k in ("open", "close", "qvol"):
        a, b = p[k][:n], d0[k][:n]
        m = ~np.isnan(a)
        print(k, "nan-agree", (np.isnan(a) == np.isnan(b)).mean(), "max rel diff", np.nanmax(np.abs(a[m] / b[m] - 1)))
    # weekend volume share of non-crypto vs crypto (sanity check of EXCLUDE)
    print("excluded present:", sum(s in EXCLUDE for s in p["symbols"]))


def load_daily(off: int = 0) -> dict:
    with np.load(XCACHE / f"daily_off{off}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_h4() -> dict:
    with np.load(XCACHE / "h4.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build()
