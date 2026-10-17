"""Structured outputs requested from Claude (validated by the SDK's messages.parse)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Assessment(BaseModel):
    category: Literal["software", "data", "writing", "translation", "design", "marketing", "other"]
    ai_deliverable: bool = Field(
        description=(
            "True only if an AI agent with a code sandbox and text tools can fully deliver the "
            "result as files/text: no calls, meetings, physical presence, client's accounts, "
            "paid third-party services, manual design tools or human identity required."
        )
    )
    ethical: bool = Field(
        description="False for fake reviews, academic cheating, spam, account farming, scraping "
        "personal data, gambling promotion, deception or anything illegal."
    )
    fit_score: int = Field(description="0-100: how well the job matches the freelancer profile.")
    complexity: Literal["trivial", "small", "medium", "large"] = Field(
        description="trivial < 1h of senior human work, small 1-4h, medium 4-16h, large > 16h"
    )
    estimated_human_hours: float
    suggested_price_usd: float = Field(
        description="Fair market fixed price for this job on this platform, in USD."
    )
    language: Literal["ru", "en", "other"]
    requirements_summary: str = Field(description="What exactly must be delivered, 2-6 lines.")
    clarifying_questions: list[str] = Field(description="At most 3 questions that block the work.")
    red_flags: list[str] = Field(
        description="Scam signals: payment outside the platform, free test work, crypto/upfront "
        "fees, requests for personal data or credentials, vague 'long-term partner' bait."
    )
    reasoning: str = Field(description="One or two sentences why.")


class ProposalDraft(BaseModel):
    text: str = Field(description="The proposal exactly as it will be sent to the client.")
    period_days: int = Field(description="Delivery time in days you commit to.")


class ClientReply(BaseModel):
    intent: Literal[
        "question",
        "requirements",
        "negotiation",
        "award",
        "revision_request",
        "acceptance",
        "payment",
        "smalltalk",
        "scam_risk",
        "other",
    ]
    reply: str = Field(description="Reply to send to the client, in the client's language.")
    requirements_update: str = Field(
        description="New or changed requirements stated in this message; empty if none."
    )
    acceptance_criteria: list[str] = Field(
        description="Concrete checks that tell the work is done, as currently agreed."
    )
    needs_human: bool = Field(
        description="True for price changes, deadlines beyond the bid, calls, contracts, "
        "refunds, disputes, legal or anything the freelancer must decide personally."
    )
    human_reason: str = Field(description="Why a human must decide; empty if needs_human is false.")


class ReviewVerdict(BaseModel):
    passed: bool
    issues: list[str] = Field(description="Concrete problems that must be fixed before delivery.")
    delivery_message: str = Field(
        description="Message to the client that accompanies the files: what is delivered, how "
        "to use it, what to check. In the client's language. No false claims."
    )
