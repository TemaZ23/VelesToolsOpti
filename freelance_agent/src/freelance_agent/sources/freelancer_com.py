"""Freelancer.com official REST API: project search (public) plus bids, messages and milestones
(needs an OAuth token from https://developers.freelancer.com)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2 as httpx

from ..models import Job

log = logging.getLogger(__name__)

API = "https://www.freelancer.com/api"
SITE = "https://www.freelancer.com"


class FreelancerError(RuntimeError):
    pass


@dataclass
class ThreadMessage:
    id: str
    from_user: int | None
    text: str
    time_created: int


def project_to_job(p: dict[str, Any]) -> Job:
    currency = p.get("currency") or {}
    budget = p.get("budget") or {}
    stats = p.get("bid_stats") or {}
    submitted = p.get("time_submitted") or p.get("submitdate")
    return Job(
        source="freelancer",
        external_id=str(p["id"]),
        url=f"{SITE}/projects/{p.get('seo_url') or p['id']}",
        title=p.get("title", "").strip(),
        description=(p.get("description") or p.get("preview_description") or "").strip(),
        budget_min=budget.get("minimum"),
        budget_max=budget.get("maximum"),
        currency=currency.get("code"),
        usd_rate=currency.get("exchange_rate"),
        bid_count=stats.get("bid_count"),
        posted_at=datetime.fromtimestamp(submitted, tz=UTC) if submitted else None,
        skills=[j["name"] for j in p.get("jobs") or [] if j.get("name")],
        raw={
            "type": p.get("type"),
            "owner_id": p.get("owner_id"),
            "language": p.get("language"),
            "bid_avg": stats.get("bid_avg"),
            "status": p.get("status"),
        },
    )


class FreelancerClient:
    def __init__(self, http: httpx.AsyncClient, oauth_token: str | None) -> None:
        self.http = http
        self.token = oauth_token
        self._self_id: int | None = None

    @property
    def can_write(self) -> bool:
        return bool(self.token)

    async def _call(self, method: str, path: str, *, auth: bool = True, **kw: Any) -> Any:
        headers = dict(kw.pop("headers", {}))
        if auth:
            if not self.token:
                raise FreelancerError("FREELANCER_OAUTH_TOKEN is not set")
            headers["Freelancer-OAuth-V1"] = self.token
        r = await self.http.request(method, f"{API}{path}", headers=headers, **kw)
        try:
            data = r.json()
        except ValueError as e:
            raise FreelancerError(f"{method} {path}: HTTP {r.status_code}, non-JSON body") from e
        if r.status_code != 200 or data.get("status") != "success":
            raise FreelancerError(
                f"{method} {path}: HTTP {r.status_code} {data.get('error_code')}: {data.get('message')}"
            )
        return data.get("result")

    # ---- search (no auth needed) -----------------------------------------------------

    async def search_active(self, query: str, limit: int = 30,
                            job_ids: list[int] | None = None) -> list[Job]:
        params: list[tuple[str, str | int]] = [
            ("query", query),
            ("limit", limit),
            ("full_description", "true"),
            ("job_details", "true"),
            ("compact", "true"),
            ("project_types[]", "fixed"),
        ]
        params += [("jobs[]", j) for j in job_ids or []]
        result = await self._call("GET", "/projects/0.1/projects/active/", auth=False,
                                  params=params)
        return [project_to_job(p) for p in result.get("projects", [])]

    # ---- bidding ---------------------------------------------------------------------

    async def self_id(self) -> int:
        if self._self_id is None:
            result = await self._call("GET", "/users/0.1/self/")
            self._self_id = int(result["id"])
        return self._self_id

    async def place_bid(self, project_id: int, amount: float, period_days: int,
                        description: str) -> int:
        result = await self._call(
            "POST",
            "/projects/0.1/bids/",
            json={
                "project_id": project_id,
                "bidder_id": await self.self_id(),
                "amount": amount,
                "period": period_days,
                "milestone_percentage": 100,
                "description": description,
            },
        )
        return int(result["id"])

    async def get_bids(self, bid_ids: list[int]) -> list[dict[str, Any]]:
        if not bid_ids:
            return []
        params = [("bids[]", b) for b in bid_ids]
        result = await self._call("GET", "/projects/0.1/bids/", params=params)
        return list(result.get("bids", []))

    async def accept_bid(self, bid_id: int) -> None:
        await self._call(
            "PUT",
            f"/projects/0.1/bids/{bid_id}/",
            params={"action": "accept"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    # ---- messages --------------------------------------------------------------------

    async def project_thread_id(self, project_id: int) -> int | None:
        result = await self._call(
            "GET",
            "/messages/0.1/threads/",
            params=[("contexts[]", project_id), ("context_type", "project")],
        )
        threads = result.get("threads", [])
        return int(threads[0]["id"]) if threads else None

    async def thread_messages(self, thread_id: int) -> list[ThreadMessage]:
        result = await self._call(
            "GET", "/messages/0.1/messages/", params=[("threads[]", thread_id), ("limit", 100)]
        )
        out = [
            ThreadMessage(
                id=str(m["id"]),
                from_user=m.get("from_user"),
                text=m.get("message") or "",
                time_created=int(m.get("time_created") or 0),
            )
            for m in result.get("messages", [])
        ]
        return sorted(out, key=lambda m: (m.time_created, m.id))

    async def post_message(self, thread_id: int, text: str) -> None:
        await self._call(
            "POST", f"/messages/0.1/threads/{thread_id}/messages/", data={"message": text}
        )

    async def post_attachment(self, thread_id: int, path: Path) -> None:
        with path.open("rb") as fh:
            await self._call(
                "POST",
                f"/messages/0.1/threads/{thread_id}/messages/",
                data={"attachments[]": path.name},
                files=[("files[]", (path.name, fh.read()))],
            )

    # ---- payments --------------------------------------------------------------------

    async def milestones(self, project_id: int) -> list[dict[str, Any]]:
        result = await self._call(
            "GET", "/projects/0.1/milestones/", params=[("projects[]", project_id)]
        )
        milestones = result.get("milestones", {})
        return list(milestones.values()) if isinstance(milestones, dict) else list(milestones)

    async def request_release(self, milestone_id: int) -> None:
        await self._call(
            "PUT", f"/projects/0.1/milestones/{milestone_id}/", params={"action": "request_release"}
        )


class FreelancerSource:
    name = "freelancer"

    def __init__(self, client: FreelancerClient, queries: list[str], limit: int,
                 job_ids: list[int]) -> None:
        self.client = client
        self.queries = queries
        self.limit = limit
        self.job_ids = job_ids

    async def fetch(self) -> list[Job]:
        seen: dict[str, Job] = {}
        for q in self.queries:
            try:
                jobs = await self.client.search_active(q, self.limit, self.job_ids)
            except (FreelancerError, httpx.HTTPError) as e:
                log.warning("freelancer search %r failed: %s", q, e)
                continue
            for job in jobs:
                seen.setdefault(job.uid, job)
        return list(seen.values())
