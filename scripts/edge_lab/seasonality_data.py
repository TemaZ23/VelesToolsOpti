"""
seasonality — подготовка часовых рядов для поиска календарных/внутридневных эффектов.

    PYTHONPATH=. python3 -m scripts.edge_lab.seasonality_data

Читает часовую панель (cache/edge_lab/panel_1h.npz) ОДИН раз и сохраняет компактный кеш
cache/edge_lab/seasonality/base_1h.npz:

    ts                      открытие часа (мс UTC)
    BTC_*, ETH_*, <ALT>_*   open/high/low/close/qvol/funding/adv по отдельным перпам
    ALT_ret                 равновзвешенный индекс ликвидных альтов: доходность часа t
    ALT_ret_pess            то же, но монета, переставшая торговаться, даёт −50%
    ALT_cost                средняя cost_per_side() по составу индекса (taker)
    ALT_fund                средняя сумма фандинга, рассчитанного в момент ts[t] (начало часа t)
    ALT_n                   число монет в индексе
    ALT30_*                 то же для топ-30 альтов по ADV

Правила (без заглядывания вперёд):
- доходность часа t: r[t] = close[t] / close[t-1] - 1 (вход по close[t-1] ≈ open[t]);
- состав индекса на час t определяется ТОЛЬКО по данным до часа t-1 включительно: монета торговалась
  на t-1, возраст ≥ 30 дней, средний объём за 720 прошлых часов (до t-1) × 24 ≥ MIN_ADV;
- если монета из состава не имеет close[t] (делистинг / остановка) — выход по последнему close,
  вклад 0 (в пессимистичном варианте −50%);
- исключены BTC, ETH, не-крипто (акции/ETF/металлы/индексы) и стейблкоины.
- фандинг: в панели 1h ставка, рассчитанная в HH:00:00, попадает в бар с открытием HH:00, т.е.
  расчёт происходит на ГРАНИЦЕ между барами t-1 и t.
"""

from __future__ import annotations

import time

import numpy as np

from scripts.edge_lab.common import CACHE, cost_per_side, panel

SCACHE = CACHE / "seasonality"
BASE = SCACHE / "base_1h.npz"
MIN_ADV = 10e6
MIN_AGE_H = 30 * 24
ADV_H = 720
SINGLE = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "DOT", "BCH", "TRX"]

# Не-крипто перпы Binance (TradFi) и стейблкоины/индексы — та же логика, что и в других семьях edge_lab.
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
EXCLUDE = {s + "USDT" for s in _TRADFI.split()}


def _past_adv(qv: np.ndarray, n: int) -> np.ndarray:
    """ADV (USD/сутки) по n часам, заканчивающимся НА t (включительно), float64. NaN → 0 в сумме,
    делитель — число валидных часов."""
    T, N = qv.shape
    out = np.full((T, N), np.nan)
    v = np.nan_to_num(qv.astype(np.float64))
    cnt = (~np.isnan(qv)).astype(np.float64)
    cs = np.zeros((T + 1, N))
    np.cumsum(v, axis=0, out=cs[1:])
    s = cs[n:] - cs[:-n]
    np.cumsum(cnt, axis=0, out=cs[1:])
    k = cs[n:] - cs[:-n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1 :] = np.where(k > 0, s / k * 24.0, np.nan)
    return out


def build() -> None:
    t0 = time.time()
    p = panel("1h")
    ts = p["ts"]
    syms = p["symbols"]
    T, N = p["close"].shape
    print(f"panel loaded {T}x{N} in {time.time() - t0:.0f}s", flush=True)
    out: dict[str, np.ndarray] = {"ts": ts}
    close = p["close"]
    adv = _past_adv(p["qvol"], ADV_H)  # adv[t] — по часам ≤ t
    for s in SINGLE:
        j = int(np.where(syms == s + "USDT")[0][0])
        for k in ("open", "high", "low", "close", "qvol", "funding"):
            out[f"{s}_{k}"] = p[k][:, j].astype(np.float64)
        out[f"{s}_adv"] = adv[:, j]
    print("singles done", flush=True)

    alive = ~np.isnan(close)
    age = np.cumsum(alive, axis=0, dtype=np.int32)
    crypto = np.array([(s not in EXCLUDE) and s not in ("BTCUSDT", "ETHUSDT") for s in syms])
    # состав на час t — по данным до t-1
    elig_prev = alive & (age >= MIN_AGE_H) & (np.nan_to_num(adv) >= MIN_ADV) & crypto[None, :]
    elig = np.zeros_like(elig_prev)
    elig[1:] = elig_prev[:-1]
    del age
    r = np.full((T, N), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = close[1:].astype(np.float64) / close[:-1].astype(np.float64) - 1.0
    # был в составе, но close[t] нет → выход по последнему close (вклад 0). Пессимистичный вариант (−50%)
    # — только для окончательной остановки торгов (после t монета больше не торгуется); разовые
    # пропуски архива (напр. 2022-02-26 00:00, 46 монет) — не делистинг.
    last_alive = T - 1 - np.argmax(alive[::-1], axis=0)
    stopped = elig & np.isnan(r) & (np.arange(T)[:, None] > last_alive[None, :])
    print(f"gap events in index: {int((elig & np.isnan(r)).sum())}", flush=True)
    print(f"stopped events in index: {int(stopped.sum())}", flush=True)
    adv_prev = np.full((T, N), np.nan)
    adv_prev[1:] = adv[:-1]
    cps = cost_per_side(np.nan_to_num(adv_prev))
    fund = p["funding"].astype(np.float64)

    def _index(mask: np.ndarray, pre: str) -> None:
        n = mask.sum(axis=1)
        rr = np.where(mask, np.nan_to_num(r), 0.0)
        rp = np.where(mask & stopped, -0.5, rr)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{pre}_ret"] = np.where(n > 0, rr.sum(1) / n, 0.0)
            out[f"{pre}_ret_pess"] = np.where(n > 0, rp.sum(1) / n, 0.0)
            out[f"{pre}_cost"] = np.where(n > 0, np.where(mask, cps, 0).sum(1) / n, np.nan)
            out[f"{pre}_fund"] = np.where(n > 0, np.where(mask, fund, 0).sum(1) / n, 0.0)
            # ADV медианной монеты состава (для оценки ёмкости) — по данным до t-1
            a = np.where(mask, adv_prev, np.nan)
            out[f"{pre}_adv_med"] = np.where(n > 0, np.nanmedian(np.where(n[:, None] > 0, a, 0), axis=1), np.nan)
        out[f"{pre}_n"] = n.astype(np.int32)
        out[f"{pre}_stopped"] = (mask & stopped).sum(1).astype(np.int32)

    _index(elig, "ALT")
    print("ALT index done", flush=True)
    # топ-30 по ADV на t-1 среди допустимых
    a = np.where(elig, np.nan_to_num(adv_prev), -1.0)
    kth = np.sort(a, axis=1)[:, -30][:, None]
    top = elig & (a >= kth) & (a > 0)
    _index(top, "ALT30")
    print("ALT30 index done", flush=True)

    # Купить-и-держать внутри блока L часов (состав фиксирован на старте блока t0: elig[t0], т.е. по
    # данным до t0-1; вход по close[t0-1]; выход по close[t0+L-1], при пропуске — по последнему
    # известному close; окончательная остановка торгов внутри блока → −50% в пессимистичном варианте).
    ff = close.astype(np.float64)
    for t in range(1, T):  # forward-fill по времени
        m = np.isnan(ff[t])
        if m.any():
            ff[t, m] = ff[t - 1, m]
    for L in (4, 24):
        nb = T // L
        st = np.arange(nb) * L  # старт блока (час открытия)
        st1 = np.maximum(st - 1, 0)
        en = st + L - 1
        for mask, pre in ((elig, "ALT"), (top, "ALT30")):
            mb = mask[st].copy()
            mb[0] = False
            entry = close[st1].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                rb = ff[en] / entry - 1.0
            mb &= np.isfinite(rb)
            term = (last_alive[None, :] >= st1[:, None]) & (last_alive[None, :] < en[:, None])
            n = mb.sum(1)
            rr = np.where(mb, np.nan_to_num(rb), 0.0)
            rp = np.where(mb & term, -0.5, rr)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{pre}_bh{L}"] = np.where(n > 0, rr.sum(1) / n, 0.0)
                out[f"{pre}_bh{L}_pess"] = np.where(n > 0, rp.sum(1) / n, 0.0)
            out[f"{pre}_bh{L}_n"] = n.astype(np.int32)
    print("block buy-and-hold done", flush=True)
    SCACHE.mkdir(parents=True, exist_ok=True)
    np.savez(BASE, **out)
    print(f"saved {BASE} in {time.time() - t0:.0f}s", flush=True)


def load_base() -> dict[str, np.ndarray]:
    with np.load(BASE) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build()
