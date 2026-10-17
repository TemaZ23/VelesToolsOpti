"""Daily exchange rates from the Bank of Russia, so ruble budgets and payments are valued right."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET

import httpx2 as httpx

from .config import Economics
from .db import Store

log = logging.getLogger(__name__)

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


def parse_cbr(content: bytes) -> dict[str, float]:
    """USD per one unit of each currency, from the CBR daily XML (rubles per unit)."""
    root = ET.fromstring(content)
    rub_per_unit: dict[str, float] = {}
    for v in root.iter("Valute"):
        code = (v.findtext("CharCode") or "").strip()
        unit = v.findtext("VunitRate")
        if unit is None:
            value = float((v.findtext("Value") or "0").replace(",", "."))
            nominal = float(v.findtext("Nominal") or "1")
            rate = value / nominal
        else:
            rate = float(unit.replace(",", "."))
        if code and rate > 0:
            rub_per_unit[code] = rate
    usd = rub_per_unit["USD"]
    rates = {code: rate / usd for code, rate in rub_per_unit.items()}
    rates["RUB"] = 1.0 / usd
    rates["USD"] = 1.0
    return rates


def apply_cached(store: Store, econ: Economics) -> None:
    cached = store.kv_get("fx_rates")
    if cached:
        econ.fx_to_usd.update(json.loads(cached))


async def refresh_daily(http: httpx.AsyncClient, store: Store, econ: Economics) -> bool:
    today = store.clock().date().isoformat()
    if store.kv_get("fx_date") == today:
        return False
    try:
        r = await http.get(CBR_URL)
        r.raise_for_status()
        rates = parse_cbr(r.content)
    except (httpx.HTTPError, ET.ParseError, KeyError, ValueError) as e:
        log.warning("CBR rates unavailable, keeping previous ones: %s", e)
        return False
    econ.fx_to_usd.update(rates)
    store.kv_set("fx_rates", json.dumps(rates))
    store.kv_set("fx_date", today)
    log.info("CBR rates updated: 1 USD = %.2f RUB", 1 / rates["RUB"])
    return True
