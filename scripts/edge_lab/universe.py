"""
Вселенная без «ошибки выжившего»: все USDT-M бессрочные фьючерсы Binance, которые когда-либо
торговались (включая делистинги), часовые свечи + история фандинга.

    PYTHONPATH=. python3 -m scripts.edge_lab.universe          # скачать / докачать
    PYTHONPATH=. python3 -m scripts.edge_lab.universe --check  # сводка по кешу

Кеш (cache/edge_lab/):
    1h/<SYM>.npz      ts (мс, открытие часа, int64), ohlc float64[N,4], qvol (объём в USDT),
                      taker_buy_qvol (объём агрессивных покупок в USDT), trades (число сделок)
                      — пропуски НЕ заполняются, часы идут как в архиве.
    funding/<SYM>.npz ts (мс), rate (доля номинала за период выплаты; + платят лонги)
    universe.json     символ → первый/последний час, число часов, делистинг (нет данных
                      в последнем месяце архива)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "edge_lab"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/"
LAST_MONTH = "2026-09"


def _get(url: str, retries: int = 5) -> bytes | None:
    url = urllib.parse.quote(url, safe=":/?&=%")  # есть символы с иероглифами
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if k == retries - 1:
                raise
        except Exception:  # noqa: BLE001 — сеть: повторяем
            if k == retries - 1:
                raise
        time.sleep(2**k)
    return None


def list_prefix(prefix: str, delimiter: bool = True) -> tuple[list[str], list[str]]:
    """Ключи и подпапки S3 под prefix (с пагинацией)."""
    keys, dirs, marker = [], [], ""
    while True:
        url = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        text = _get(url).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", text)
        dirs += re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>", text)
        if "<IsTruncated>true</IsTruncated>" not in text:
            return keys, dirs
        nxt = re.findall(r"<NextMarker>([^<]+)</NextMarker>", text)
        marker = nxt[0] if nxt else (keys[-1] if keys else dirs[-1])


def all_symbols() -> list[str]:
    _, dirs = list_prefix("data/futures/um/monthly/klines/")
    syms = [d.rstrip("/").split("/")[-1] for d in dirs]
    return sorted(s for s in syms if s.endswith("USDT") and "_" not in s)


def _read_csv(blob: bytes, cols: tuple[int, ...]) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode()
    if text and not text[:1].isdigit():
        text = text[text.find("\n") + 1 :]
    if not text.strip():
        return np.empty((0, len(cols)))
    arr = np.loadtxt(io.StringIO(text), delimiter=",", usecols=cols, ndmin=2)
    if len(arr) and arr[0, 0] > 1e14:  # микросекунды в архивах с 2025 года
        arr[:, 0] = np.floor(arr[:, 0] / 1000)
    return arr


def _months(keys: list[str]) -> list[str]:
    return sorted({m for k in keys if k.endswith(".zip") for m in re.findall(r"(\d{4}-\d{2})\.zip$", k)})


def fetch_symbol(sym: str) -> dict:
    out = {"symbol": sym}
    kpath = CACHE / "1h" / f"{sym}.npz"
    if not kpath.exists():
        keys, _ = list_prefix(f"data/futures/um/monthly/klines/{sym}/1h/", delimiter=False)
        months = [m for m in _months(keys) if m <= LAST_MONTH]
        rows = []
        for m in months:
            blob = _get(f"{BASE}data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{m}.zip")
            if blob:
                rows.append(_read_csv(blob, (0, 1, 2, 3, 4, 7, 8, 10)))
        rows = [r for r in rows if len(r)]
        if not rows:
            out["error"] = "no klines"
            return out
        a = np.concatenate(rows)
        a = a[np.argsort(a[:, 0], kind="stable")]
        a = a[np.r_[True, a[1:, 0] != a[:-1, 0]]]
        kpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(kpath, ts=a[:, 0].astype(np.int64), ohlc=a[:, 1:5], qvol=a[:, 5], trades=a[:, 6], taker_buy_qvol=a[:, 7])
    fpath = CACHE / "funding" / f"{sym}.npz"
    if not fpath.exists():
        keys, _ = list_prefix(f"data/futures/um/monthly/fundingRate/{sym}/", delimiter=False)
        rows = []
        for m in [m for m in _months(keys) if m <= LAST_MONTH]:
            blob = _get(f"{BASE}data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
            if blob:
                rows.append(_read_csv(blob, (0, 2)))
        rows = [r for r in rows if len(r)]
        a = np.concatenate(rows) if rows else np.empty((0, 2))
        a = a[np.argsort(a[:, 0], kind="stable")]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fpath, ts=a[:, 0].astype(np.int64), rate=a[:, 1])
    with np.load(kpath) as z:
        ts = z["ts"]
    out.update(first=int(ts[0]), last=int(ts[-1]), hours=int(len(ts)))
    return out


def summarize(results: list[dict]) -> dict:
    last_cut = int(np.datetime64(f"{LAST_MONTH}-25", "ms").astype(np.int64))
    uni = {}
    for r in results:
        if "first" not in r:
            continue
        uni[r["symbol"]] = {
            "first": str(np.datetime64(r["first"], "ms")),
            "last": str(np.datetime64(r["last"], "ms")),
            "hours": r["hours"],
            "delisted": bool(r["last"] < last_cut),
        }
    return uni


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    if args.check:
        uni = json.loads((CACHE / "universe.json").read_text())
        dl = sum(v["delisted"] for v in uni.values())
        print(f"символов {len(uni)}, из них делистинг {dl}")
        return
    syms = all_symbols()
    print("символов USDT-M:", len(syms), flush=True)
    results, t0 = [], time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch_symbol, s): s for s in syms}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": futs[f], "error": repr(exc)})
            if k % 25 == 0:
                print(f"{k}/{len(syms)} за {time.time() - t0:.0f} с", flush=True)
    uni = summarize(results)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "universe.json").write_text(json.dumps(uni, indent=1))
    errs = [r for r in results if "error" in r]
    print(f"готово: {len(uni)} символов, делистинг {sum(v['delisted'] for v in uni.values())}, ошибок {len(errs)}")
    for e in errs[:20]:
        print("  ", e)


if __name__ == "__main__":
    sys.exit(main())
