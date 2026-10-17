"""FL.ru public RSS feeds. FL.ru has no bidding API: proposals are sent by you (one tap + paste)."""

from __future__ import annotations

import calendar
import html
import re
from datetime import UTC, datetime

import feedparser
import httpx2 as httpx

from ..models import Job

_ID = re.compile(r"/projects/(\d+)/")
_BUDGET = re.compile(r"Бюджет:\s*([\d\s ]+)\s*([^,)]*)")
_TAIL = re.compile(r"\s*\((?:Бюджет:[^)]*|для всех)\)\s*$")
_CURRENCY = {"₽": "RUB", "руб": "RUB", "$": "USD", "€": "EUR", "usd": "USD", "eur": "EUR"}


def parse_budget(title: str) -> tuple[float | None, str | None]:
    m = _BUDGET.search(title)
    if not m:
        return None, None
    digits = re.sub(r"\D", "", m.group(1))
    if not digits:
        return None, None
    symbol = m.group(2).strip().lower().rstrip(".")
    currency = next((code for key, code in _CURRENCY.items() if symbol.startswith(key)), "RUB")
    return float(digits), currency


def clean_title(title: str) -> str:
    return _TAIL.sub("", title).strip()


def parse_feed(content: bytes) -> list[Job]:
    feed = feedparser.parse(content)
    jobs: list[Job] = []
    for e in feed.entries:
        link = e.get("link", "")
        m = _ID.search(link)
        if not m:
            continue
        raw_title = html.unescape(e.get("title", ""))
        amount, currency = parse_budget(raw_title)
        posted = None
        if e.get("published_parsed"):
            posted = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=UTC)
        jobs.append(
            Job(
                source="flru",
                external_id=m.group(1),
                url=link,
                title=clean_title(raw_title),
                description=html.unescape(e.get("summary", "")).strip(),
                budget_min=None,
                budget_max=amount,
                currency=currency,
                posted_at=posted,
                skills=[t.get("term", "").strip() for t in e.get("tags", []) if t.get("term")],
                # Without a PRO account FL.ru lets you answer only projects marked "для всех".
                raw={"category": e.get("category"), "open_to_all": "для всех" in raw_title},
            )
        )
    return jobs


class FlruSource:
    name = "flru"

    def __init__(self, http: httpx.AsyncClient, feeds: list[str], pro_account: bool) -> None:
        self.http = http
        self.feeds = feeds
        self.pro_account = pro_account

    async def fetch(self) -> list[Job]:
        seen: dict[str, Job] = {}
        for url in self.feeds:
            r = await self.http.get(url)
            r.raise_for_status()
            for job in parse_feed(r.content):
                if self.pro_account or job.raw.get("open_to_all"):
                    seen.setdefault(job.uid, job)
        return list(seen.values())
