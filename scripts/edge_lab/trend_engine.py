"""
Семейство «trend»: общие данные и движок суточного бэктеста портфеля фьючерсов.

Соглашения (важно для отсутствия заглядывания вперёд):
- бар t = сутки, ts[t] — открытие (00:00 UTC). Всё, что вычислено по close[t] (сигналы, ранги ADV,
  волатильность, режим BTC) — это целевые веса TW[t], которые исполняются по open[t+1];
- позиция, открытая по open[t+1], держится в течение бара t+1 и зарабатывает
  R[t+1] = open[t+2] / open[t+1] - 1 (если open[t+2] нет — монета перестала торговаться — то
  close[t+1] / open[t+1] - 1, т. е. выход по последнему close);
- фандинг: выплаты внутри первого часа бара (00:00 UTC) платит СТАРАЯ позиция (ребалансировка идёт
  сразу после расчёта в 00:00), остальные выплаты бара (08:00, 16:00 и т. д.) — новая позиция;
  знак: + ставка — лонг платит;
- издержки: |сделка| × cost_per_side(ADV на момент t) на каждую сделку, включая ребалансировки;
- делистинг: если open[t+1] нет, позиция принудительно закрывается (цена — последний close[t]),
  пессимистичный вариант — дополнительно −50 % номинала этой позиции.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from scripts.edge_lab.common import CACHE, DAY, HOUR, adv_usd, cost_per_side, load_funding, panel

TCACHE = CACHE / "trend"

# Не-крипто перпы Binance (TradFi: акции, ETF, металлы, сырьё, индексы, валюты) и стейблкоины/индексы.
# Список совпадает со списком семейства xsec (скопирован, чтобы не зависеть от чужого кода).
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


def _fund_split(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Фандинг бара делится на выплату в первый час (00:00 UTC; её платит позиция ДО ребалансировки)
    и остальные выплаты бара (их платит позиция ПОСЛЕ ребалансировки)."""
    path = TCACHE / "fund0_1d.npy"
    if path.exists():
        f0 = np.load(path)
    else:
        TCACHE.mkdir(parents=True, exist_ok=True)
        ts, syms = p["ts"], p["symbols"]
        T, N = len(ts), len(syms)
        f0 = np.zeros((T, N))
        t0 = int(ts[0])
        for j, s in enumerate(syms):
            f = load_funding(str(s))
            if not len(f["ts"]):
                continue
            b = (f["ts"] - t0) // DAY
            off = (f["ts"] - t0) % DAY
            ok = (b >= 0) & (b < T) & (off < HOUR)
            np.add.at(f0[:, j], b[ok].astype(np.int64), f["rate"][ok])
        np.save(path, f0)
    f1 = np.nan_to_num(p["funding"]) - f0
    return f0, f1


def daily_data() -> dict:
    """Суточные массивы для движка. Только крипто-перпы (TradFi/стейблкоины исключены)."""
    p = panel("1d")
    keep = np.array([s not in EXCLUDE for s in p["symbols"]])
    f0, f1 = _fund_split(p)
    O = p["open"][:, keep].astype(np.float64)
    C = p["close"][:, keep].astype(np.float64)
    Q = p["qvol"][:, keep].astype(np.float64)
    syms = p["symbols"][keep]
    T, N = O.shape
    alive = ~np.isnan(C)
    # доходность удержания в баре t: open[t] -> open[t+1], иначе -> close[t] (последний close)
    R = np.zeros((T, N))
    with np.errstate(invalid="ignore", divide="ignore"):
        nxt = np.vstack([O[1:], np.full((1, N), np.nan)])
        r_oo = nxt / O - 1
        r_oc = C / O - 1
    R = np.where(alive & ~np.isnan(nxt), r_oo, np.where(alive, r_oc, 0.0))
    R = np.nan_to_num(R)
    # «торгуется на следующем открытии» — можно держать позицию в баре t+1
    tradable_next = np.vstack([~np.isnan(O[1:]), np.zeros((1, N), bool)])
    with np.errstate(invalid="ignore", divide="ignore"):
        rcc = np.vstack([np.full((1, N), np.nan), C[1:] / C[:-1] - 1])
    adv = np.nan_to_num(adv_usd({"ts": p["ts"], "qvol": Q}))  # 30 дней, включая t
    cost = cost_per_side(np.maximum(adv, 1.0))
    age = np.cumsum(alive, axis=0)
    return {
        "ts": p["ts"], "symbols": syms, "O": O, "C": C, "Q": Q, "R": R, "rcc": rcc, "alive": alive,
        "tradable_next": tradable_next, "adv": adv, "cost": cost, "age": age,
        "F0": f0[:, keep].astype(np.float64), "F1": f1[:, keep].astype(np.float64),
    }  # fmt: skip


# ----------------------------------------------------------------------------------------------
# индикаторы (по каждой монете, только прошлые данные)


@njit(cache=True)
def ema_cols(x: np.ndarray, span: int) -> np.ndarray:
    """EMA по столбцам с NaN-пропусками; NaN, пока не набралось span наблюдений."""
    T, N = x.shape
    a = 2.0 / (span + 1.0)
    out = np.full((T, N), np.nan)
    for j in range(N):
        e = np.nan
        n = 0
        for t in range(T):
            v = x[t, j]
            if not np.isnan(v):
                if n == 0:
                    e = v
                else:
                    e = a * v + (1 - a) * e
                n += 1
            if n >= span and not np.isnan(x[t, j]):
                out[t, j] = e
    return out


@njit(cache=True)
def ewm_vol(r: np.ndarray, halflife: float, min_obs: int) -> np.ndarray:
    """EWMA-волатильность дневных доходностей (годовая, ×sqrt(365)), только прошлые данные до t вкл."""
    T, N = r.shape
    a = 1.0 - np.exp(np.log(0.5) / halflife)
    out = np.full((T, N), np.nan)
    for j in range(N):
        v = 0.0
        n = 0
        for t in range(T):
            x = r[t, j]
            if not np.isnan(x):
                x = min(max(x, -0.6), 1.5)  # грубая защита от сбойных принтов
                if n == 0:
                    v = x * x
                else:
                    v = a * x * x + (1 - a) * v
                n += 1
                if n >= min_obs:
                    out[t, j] = np.sqrt(v * 365.0)
    return out


@njit(cache=True)
def donchian_state(C: np.ndarray, n_in: int, n_out: int, allow_short: bool) -> np.ndarray:
    """Пробой канала: лонг, когда close[t] >= max(close[t-n_in..t-1]); выход из лонга, когда
    close[t] <= min(close[t-n_out..t-1]); шорт симметрично. Состояние на t по данным до t."""
    T, N = C.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        st = 0.0
        cnt = 0
        for t in range(T):
            c = C[t, j]
            if np.isnan(c):
                cnt = 0
                st = 0.0
                continue
            cnt += 1
            if cnt <= n_in:
                continue
            hi_in = -1e300
            lo_in = 1e300
            for k in range(1, n_in + 1):
                v = C[t - k, j]
                hi_in = max(hi_in, v)
                lo_in = min(lo_in, v)
            hi_out = -1e300
            lo_out = 1e300
            for k in range(1, n_out + 1):
                v = C[t - k, j]
                hi_out = max(hi_out, v)
                lo_out = min(lo_out, v)
            if st > 0 and c <= lo_out:
                st = 0.0
            elif st < 0 and c >= hi_out:
                st = 0.0
            if st == 0.0:
                if c >= hi_in:
                    st = 1.0
                elif allow_short and c <= lo_in:
                    st = -1.0
            out[t, j] = st
    return out


def topk_mask(d: dict, k: int, min_age: int = 60) -> np.ndarray:
    """Вселенная на t: крипто, торгуется на t и на следующем открытии неизвестно (не используем),
    возраст ≥ min_age дней, top-k по 30-дневному ADV (включая t)."""
    ok = d["alive"] & (d["age"] >= min_age)
    score = np.where(ok, d["adv"], -1.0)
    T, N = score.shape
    m = np.zeros((T, N), bool)
    kk = min(k, N)
    idx = np.argpartition(-score, kk - 1, axis=1)[:, :kk]
    rows = np.repeat(np.arange(T), kk)
    m[rows, idx.ravel()] = True
    return m & ok


@njit(cache=True)
def hist_port_vol(W: np.ndarray, rcc: np.ndarray, win: int) -> np.ndarray:
    """Ex-ante волатильность портфеля с весами W[t] по историческим доходностям rcc[t-win+1..t]
    (историческое моделирование, ковариации учитываются неявно). Годовая."""
    T, N = W.shape
    out = np.full(T, np.nan)
    pr = np.empty(win)
    for t in range(win, T):
        g = 0.0
        for j in range(N):
            g += abs(W[t, j])
        if g == 0.0:
            out[t] = 0.0
            continue
        for k in range(win):
            s = 0.0
            for j in range(N):
                w = W[t, j]
                if w != 0.0:
                    x = rcc[t - k, j]
                    if not np.isnan(x):
                        s += w * x
            pr[k] = s
        m = pr.mean()
        out[t] = np.sqrt(((pr - m) ** 2).sum() / (win - 1) * 365.0)
    return out


# ----------------------------------------------------------------------------------------------
# движок


@njit(cache=True)
def simulate(TW, reb, R, F0, F1, cost, tradable_next, buf, delist_pen, cap0, min_usd):
    """TW[t] — целевые веса (доля капитала, со знаком), решённые по close[t]; reb[t] — ребалансировать
    ли на open[t+1]. Возвращает дневные чистые доходности капитала и разбивку P&L.
    buf — не торговать монету, если |цель − текущий вес| < buf·|цель| (и знак тот же);
    min_usd — минимальный размер ордера (кроме полного закрытия), cap0 — стартовый счёт в $."""
    T, N = TW.shape
    w = np.zeros(N)
    ret = np.zeros(T)
    pr_pnl = np.zeros(T)
    fu_pnl = np.zeros(T)
    co_pnl = np.zeros(T)
    gross = np.zeros(T)
    net = np.zeros(T)
    turn = np.zeros(T)
    entries = 0
    fills = 0
    delists = 0
    eq = cap0
    for t in range(T - 1):
        tt = t + 1  # бар, в котором держим позицию после исполнения на open[tt]
        f = 0.0
        for j in range(N):
            if w[j] != 0.0:
                f -= w[j] * F0[tt, j]
        c = 0.0
        pen = 0.0
        tv = 0.0
        for j in range(N):
            if not tradable_next[t, j]:
                if w[j] != 0.0:  # монета перестала торговаться: закрытие по последнему close
                    c += abs(w[j]) * cost[t, j]
                    tv += abs(w[j])
                    pen += delist_pen * abs(w[j])
                    delists += 1
                    fills += 1
                    w[j] = 0.0
                continue
            if not reb[t]:
                continue
            tgt = TW[t, j]
            dw = tgt - w[j]
            if dw == 0.0:
                continue
            if tgt != 0.0 and w[j] * tgt > 0.0 and abs(dw) < buf * abs(tgt):
                continue
            if tgt != 0.0 and min_usd > 0.0 and abs(dw) * eq < min_usd:
                continue
            if w[j] == 0.0 or w[j] * tgt < 0.0:
                if tgt != 0.0:
                    entries += 1
            c += abs(dw) * cost[t, j]
            tv += abs(dw)
            fills += 1
            w[j] = tgt
        p = 0.0
        g = 0.0
        nn = 0.0
        for j in range(N):
            if w[j] != 0.0:
                p += w[j] * R[tt, j]
                f -= w[j] * F1[tt, j]
                g += abs(w[j])
                nn += w[j]
        r = p + f - c - pen
        if r < -0.999:
            r = -0.999
        ret[tt] = r
        pr_pnl[tt] = p - pen
        fu_pnl[tt] = f
        co_pnl[tt] = -c
        gross[tt] = g
        net[tt] = nn
        turn[tt] = tv
        eq *= 1.0 + r
        if r != -1.0:
            for j in range(N):
                if w[j] != 0.0:
                    w[j] = w[j] * (1.0 + R[tt, j]) / (1.0 + r)
    return ret, pr_pnl, fu_pnl, co_pnl, gross, net, turn, entries, fills, delists


def weekday(ts: np.ndarray) -> np.ndarray:
    """0 = понедельник ... 6 = воскресенье (ts — открытие бара)."""
    return ((ts // DAY + 3) % 7).astype(np.int64)


def run(d: dict, TW: np.ndarray, reb: np.ndarray, buf: float = 0.0, delist_pen: float = 0.0,
        cap0: float = 1e6, min_usd: float = 0.0) -> dict:  # fmt: skip
    out = simulate(np.ascontiguousarray(TW), reb, d["R"], d["F0"], d["F1"], d["cost"], d["tradable_next"],
                   buf, delist_pen, cap0, min_usd)  # fmt: skip
    keys = ("ret", "price", "funding", "cost", "gross", "net", "turn")
    res = {k: v for k, v in zip(keys, out[:7])}
    res["entries"], res["fills"], res["delists"] = int(out[7]), int(out[8]), int(out[9])
    return res
