"""Domain types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    NEW = "new"
    FILTERED = "filtered"  # rejected by the free prefilter
    SKIPPED = "skipped"  # rejected after scoring / economics, or by you
    AWAITING_APPROVAL = "awaiting_approval"  # proposal drafted, waiting for your tap
    SUBMITTED = "submitted"  # bid placed / you confirmed you responded
    LOST = "lost"
    WON = "won"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"  # deliverable ready, waiting for your tap to send
    DELIVERED = "delivered"
    PAID = "paid"
    CANCELLED = "cancelled"
    FAILED = "failed"


ACTIVE_WORK = (JobStatus.WON, JobStatus.IN_PROGRESS, JobStatus.REVIEW, JobStatus.DELIVERED)


@dataclass
class Job:
    source: str
    external_id: str
    url: str
    title: str
    description: str
    budget_min: float | None = None
    budget_max: float | None = None
    currency: str | None = None
    # USD per one unit of `currency`, when the platform tells us.
    usd_rate: float | None = None
    bid_count: int | None = None
    posted_at: datetime | None = None
    skills: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.external_id}"
