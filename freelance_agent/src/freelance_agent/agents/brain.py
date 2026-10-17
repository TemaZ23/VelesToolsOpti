"""The thinking steps of the pipeline: triage, proposal, client replies, delivery review."""

from __future__ import annotations

from ..config import Settings
from ..db import Message
from ..economics import BidDecision, usd_rate_for
from ..llm import LLM
from ..models import Job
from . import prompts
from .schemas import Assessment, ClientReply, ProposalDraft, ReviewVerdict

PERIOD_DAYS = {"trivial": 1, "small": 2, "medium": 5, "large": 10}

PLATFORM_LIMITS = {
    "freelancer": (100, 1500, "100-1500 characters, no contact details"),
    "flru": (50, 2000, "up to 2000 characters, no contact details"),
}
DEFAULT_LIMITS = (50, 2000, "up to 2000 characters")


class Brain:
    def __init__(self, llm: LLM, settings: Settings) -> None:
        self.llm = llm
        self.s = settings

    async def assess(self, job: Job, job_id: int) -> Assessment:
        rate = usd_rate_for(job, self.s.economics.fx_to_usd)
        budget_usd = (
            job.budget_min * rate if rate and job.budget_min else None,
            job.budget_max * rate if rate and job.budget_max else None,
        )
        a = await self.llm.structured(
            purpose="triage",
            model=self.s.models.triage,
            effort=self.s.models.triage_effort,
            system=prompts.TRIAGE_SYSTEM,
            user=prompts.triage_user(job, self.s.profile, budget_usd),
            schema=Assessment,
            max_tokens=4000,
            job_id=job_id,
        )
        a.fit_score = max(0, min(100, a.fit_score))
        a.suggested_price_usd = max(0.0, a.suggested_price_usd)
        return a

    async def write_proposal(self, job: Job, job_id: int, a: Assessment,
                             d: BidDecision) -> ProposalDraft:
        lo, hi, limits = PLATFORM_LIMITS.get(job.source, DEFAULT_LIMITS)
        auto = self.s.autonomy
        draft = await self.llm.structured(
            purpose="proposal",
            model=self.s.models.writer,
            effort=self.s.models.writer_effort,
            system=prompts.PROPOSAL_SYSTEM,
            user=prompts.proposal_user(
                job,
                self.s.profile,
                a.model_dump_json(indent=1),
                d.bid_amount,
                d.currency,
                PERIOD_DAYS[a.complexity],
                limits,
                auto.disclosure_text if auto.disclose_ai else None,
            ),
            schema=ProposalDraft,
            max_tokens=6000,
            job_id=job_id,
        )
        draft.text = draft.text.strip()
        if len(draft.text) > hi:
            draft.text = _shorten(draft.text, hi)
        if len(draft.text) < lo:
            raise ValueError(f"proposal too short for {job.source}: {len(draft.text)} chars")
        draft.period_days = max(1, draft.period_days)
        return draft

    async def draft_reply(self, job: Job, job_id: int, requirements: str, bid: str,
                          history: list[Message], new_message: str) -> ClientReply:
        return await self.llm.structured(
            purpose="reply",
            model=self.s.models.writer,
            effort=self.s.models.writer_effort,
            system=prompts.REPLY_SYSTEM,
            user=prompts.reply_user(job, self.s.profile, requirements, bid, history, new_message),
            schema=ClientReply,
            max_tokens=6000,
            job_id=job_id,
        )

    async def review(self, job: Job, job_id: int, requirements: str,
                     files_listing: str, files_content: str, agent_report: str) -> ReviewVerdict:
        user = (
            f"{prompts.job_block(job)}\n\n<agreed_requirements>\n{requirements}\n"
            f"</agreed_requirements>\n\n<agent_report>\n{agent_report}\n</agent_report>\n\n"
            f"<files>\n{files_listing}\n</files>\n\n<file_contents>\n{files_content}\n"
            "</file_contents>\n\nReview the deliverable."
        )
        return await self.llm.structured(
            purpose="review",
            model=self.s.models.writer,
            effort=self.s.models.writer_effort,
            system=prompts.REVIEW_SYSTEM,
            user=user,
            schema=ReviewVerdict,
            max_tokens=8000,
            job_id=job_id,
        )


def _shorten(text: str, limit: int) -> str:
    """Cut at the last sentence/paragraph boundary that fits."""
    cut = text[:limit]
    for sep in ("\n\n", "\n", ". ", "! ", "? "):
        pos = cut.rfind(sep)
        if pos > limit * 0.6:
            return cut[: pos + 1].rstrip()
    return cut.rstrip()
