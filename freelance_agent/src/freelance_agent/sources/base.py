"""Job source interface and shared HTTP settings."""

from __future__ import annotations

from typing import Protocol

import httpx2 as httpx

from ..models import Job

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) freelance-agent/0.1"


def make_http(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )


class Source(Protocol):
    name: str

    async def fetch(self) -> list[Job]: ...
