"""Any RSS/Atom feed of job posts (e.g. WeWorkRemotely categories). Responses are manual."""

from __future__ import annotations

import calendar
import hashlib
import html
import re
from datetime import UTC, datetime

import feedparser
import httpx2 as httpx

from ..models import Job

_TAGS = re.compile(r"<[^>]+>")


def parse_feed(content: bytes, source: str) -> list[Job]:
    feed = feedparser.parse(content)
    jobs: list[Job] = []
    for e in feed.entries:
        link = e.get("link", "")
        key = e.get("id") or link
        if not key:
            continue
        posted = None
        if e.get("published_parsed"):
            posted = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=UTC)
        text = _TAGS.sub(" ", html.unescape(e.get("summary", "")))
        jobs.append(
            Job(
                source=source,
                external_id=hashlib.sha1(key.encode()).hexdigest()[:16],
                url=link,
                title=html.unescape(e.get("title", "")).strip(),
                description=re.sub(r"[ \t]+", " ", text).strip(),
                posted_at=posted,
                skills=[t.get("term", "") for t in e.get("tags", []) if t.get("term")],
            )
        )
    return jobs


class RssSource:
    def __init__(self, http: httpx.AsyncClient, name: str, url: str) -> None:
        self.http = http
        self.name = name
        self.url = url

    async def fetch(self) -> list[Job]:
        r = await self.http.get(self.url)
        r.raise_for_status()
        return parse_feed(r.content, self.name)
