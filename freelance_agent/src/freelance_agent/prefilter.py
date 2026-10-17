"""Free, deterministic filtering before any paid LLM call."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .config import Economics, Filters
from .economics import to_usd
from .models import Job

# High-precision stop words only: jobs an AI agent cannot do (body, voice, phone) or that are
# unethical. Everything subtler is left to the triage model, which costs fractions of a cent.
DEFAULT_EXCLUDE = [
    # unethical / platform-banned
    "отзыв", "накрут", "курсовую", "курсовая", "дипломную", "дипломная", "реферат",
    "контрольную", "казино", "букмекер", "fake review", "write reviews", "homework",
    "assignment help", "buy followers", "upvote", "casino", "betting", "aged account",
    # needs a human body / voice / phone
    "курьер", "устный перевод", "устный переводчик", "респондент", "озвучк", "озвучить",
    "диктор", "видеосъемк", "фотосъемк", "видеосъёмк", "фотосъёмк", "сними видео", "снять видео",
    "фотосесс", "обзвон", "холодные звонки", "менеджер по продаж", "voice over", "voiceover",
    "cold call", "call center", "onsite", "on-site", "in person", "photographer", "videographer",
    # employment, not a gig
    "вакансия", "full-time", "full time position",
]

_WORD = re.compile(r"\w+", re.UNICODE)


def _contains(haystack: str, needle: str) -> bool:
    needle = needle.lower()
    if " " in needle or not needle.isalnum():
        return needle in haystack
    # Prefix match on word starts: "отзыв" matches "отзывы", but "exam" does not match "example".
    return any(w.startswith(needle) and (len(w) - len(needle) <= 3) for w in _WORD.findall(haystack))


def check(job: Job, f: Filters, econ: Economics, now: datetime) -> tuple[bool, str]:
    text = f"{job.title}\n{job.description}".lower()

    for kw in [*DEFAULT_EXCLUDE, *f.exclude_keywords]:
        if _contains(text, kw):
            return False, f"стоп-слово «{kw}»"

    if f.include_keywords and not any(_contains(text, kw) for kw in f.include_keywords):
        return False, "нет ключевых слов из filters.include_keywords"

    if job.raw.get("type") == "hourly":
        return False, "почасовой проект (поддерживаются только фиксированные)"

    hi = to_usd(job.budget_max or job.budget_min, job.currency, job.usd_rate, econ.fx_to_usd)
    if hi is not None and hi < f.min_budget_usd:
        return False, f"бюджет ~${hi:.0f} < ${f.min_budget_usd:.0f}"

    if job.bid_count is not None and job.bid_count > f.max_existing_bids:
        return False, f"уже {job.bid_count} откликов"

    if job.posted_at and now - job.posted_at > timedelta(hours=f.max_age_hours):
        return False, "заказ слишком старый"

    if len(job.description.strip()) < 40:
        return False, "слишком короткое описание"

    return True, ""
