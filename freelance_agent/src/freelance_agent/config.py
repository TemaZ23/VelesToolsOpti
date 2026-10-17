"""Configuration: YAML file for settings, environment variables for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Complexity = Literal["trivial", "small", "medium", "large"]


class Profile(BaseModel):
    """Who the agent represents. Everything here must be true: proposals quote it."""

    name: str
    headline: str
    about: str
    skills: list[str]
    portfolio: list[str] = Field(default_factory=list)
    hourly_rate_usd: float = 25.0
    min_project_usd: float = 30.0


class Models(BaseModel):
    # Bulk triage of every job that survives the free prefilter: cheapest model.
    triage: str = "claude-haiku-5-5"
    triage_effort: Effort = "low"
    # Proposals, client replies, delivery review: quality matters, volume is low.
    writer: str = "claude-opus-5-5"
    writer_effort: Effort = "medium"
    # The agent that actually does the job (Claude Agent SDK).
    executor: str = "claude-opus-5-5"
    executor_effort: Effort = "high"
    server_side_fallback: bool = True


class Budget(BaseModel):
    # Per day, only for finding work (triage + proposals).
    daily_llm_limit_usd: float = 2.0
    # Per calendar month, all Claude API spend including doing the work.
    monthly_llm_limit_usd: float = 40.0
    # Lifetime LLM spend may exceed lifetime net revenue by at most this much.
    # When tripped, the agent stops bidding (already won jobs are still delivered).
    stop_loss_usd: float = 15.0
    # Subscriptions/VPS/platform plans. Only reported (break-even), never auto-spent.
    fixed_monthly_costs_usd: dict[str, float] = Field(default_factory=lambda: {"claude_subscription": 20.0})


class PlatformFee(BaseModel):
    percent: float = 10.0
    min_usd: float = 0.0


def _execution_costs() -> dict[Complexity, float]:
    return {"trivial": 0.5, "small": 1.5, "medium": 4.0, "large": 10.0}


class Economics(BaseModel):
    # Bayesian prior for the proposal win rate before we have our own statistics.
    prior_win_rate: float = 0.05
    prior_strength: int = 20
    min_profit_per_job_usd: float = 15.0
    # A bid is placed only if expected profit >= this multiple of what the bid costs us.
    min_ev_to_bid_cost_ratio: float = 3.0
    proposal_cost_estimate_usd: float = 0.06
    max_execution_share: float = 0.25
    max_execution_usd: float = 8.0
    execution_cost_usd: dict[Complexity, float] = Field(default_factory=_execution_costs)
    platform_fees: dict[str, PlatformFee] = Field(
        default_factory=lambda: {
            "freelancer": PlatformFee(percent=10.0, min_usd=5.0),
            "flru": PlatformFee(percent=0.0),
        }
    )
    # Units of USD per one unit of currency. Freelancer.com supplies live rates itself.
    fx_to_usd: dict[str, float] = Field(
        default_factory=lambda: {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "RUB": 0.0125, "INR": 0.0118}
    )


class Autonomy(BaseModel):
    # manual: every outgoing proposal / message waits for your tap in Telegram.
    # semi:   proposals inside the limits below go out automatically; messages wait.
    # auto:   proposals and replies go out automatically; won jobs start automatically.
    mode: Literal["manual", "semi", "auto"] = "manual"
    auto_bid_min_score: int = 80
    auto_bid_max_usd: float = 300.0
    # Appended to proposals when true.
    disclose_ai: bool = False
    disclosure_text: str = "I use AI tools under my supervision to deliver faster."


class Filters(BaseModel):
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    min_budget_usd: float = 30.0
    min_fit_score: int = 55
    max_existing_bids: int = 60
    max_age_hours: float = 48.0


class FlruSource(BaseModel):
    enabled: bool = True
    feeds: list[str] = Field(default_factory=lambda: ["https://www.fl.ru/rss/all.xml"])
    # Without PRO only projects marked "для всех" accept your responses.
    pro_account: bool = False


class FreelancerSource(BaseModel):
    enabled: bool = True
    queries: list[str] = Field(default_factory=lambda: ["python", "scraping", "telegram bot"])
    job_ids: list[int] = Field(default_factory=list)
    per_query_limit: int = 30
    max_bids_per_day: int = 10


class RssSource(BaseModel):
    name: str
    url: str


class Sources(BaseModel):
    flru: FlruSource = Field(default_factory=FlruSource)
    freelancer: FreelancerSource = Field(default_factory=FreelancerSource)
    rss: list[RssSource] = Field(default_factory=list)


class Executor(BaseModel):
    max_turns: int = 60
    sandbox: bool = True
    # Running without the OS sandbox lets job text (untrusted) drive Bash on your host.
    allow_unsandboxed: bool = False
    allow_web: bool = False
    # Hosts sandboxed Bash may reach (package registries for installing dependencies).
    allowed_domains: list[str] = Field(
        default_factory=lambda: ["pypi.org", "files.pythonhosted.org", "registry.npmjs.org"]
    )
    # Needed inside unprivileged Docker, where bubblewrap cannot create full namespaces.
    # The container is then the outer isolation layer. FREELANCE_AGENT_DOCKER=1 turns it on.
    weaker_nested_sandbox: bool = Field(
        default_factory=lambda: os.environ.get("FREELANCE_AGENT_DOCKER") == "1"
    )
    review_retries: int = 1


class Telegram(BaseModel):
    enabled: bool = True
    chat_id: int | None = None


class Settings(BaseModel):
    data_dir: Path = Path("data")
    poll_interval_minutes: int = 10
    max_new_jobs_per_cycle: int = 40
    profile: Profile
    models: Models = Field(default_factory=Models)
    budget: Budget = Field(default_factory=Budget)
    economics: Economics = Field(default_factory=Economics)
    autonomy: Autonomy = Field(default_factory=Autonomy)
    filters: Filters = Field(default_factory=Filters)
    sources: Sources = Field(default_factory=Sources)
    executor: Executor = Field(default_factory=Executor)
    telegram: Telegram = Field(default_factory=Telegram)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.budget.daily_llm_limit_usd > self.budget.monthly_llm_limit_usd:
            raise ValueError("budget.daily_llm_limit_usd must not exceed budget.monthly_llm_limit_usd")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.sqlite3"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"


class Secrets(BaseModel):
    anthropic_api_key: str | None = None
    telegram_bot_token: str | None = None
    freelancer_oauth_token: str | None = None

    @classmethod
    def from_env(cls) -> Secrets:
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            freelancer_oauth_token=os.environ.get("FREELANCER_OAUTH_TOKEN") or None,
        )


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_settings(path: Path) -> Settings:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(data)
    if not settings.data_dir.is_absolute():
        settings.data_dir = (path.parent / settings.data_dir).resolve()
    return settings
