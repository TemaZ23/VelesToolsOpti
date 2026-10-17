"""Wires settings and secrets into a running agent."""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
import httpx2 as httpx

from . import fx
from .agents.brain import Brain
from .agents.executor import Executor, QueryFn
from .budget import BudgetGuard
from .config import Secrets, Settings
from .db import Store
from .llm import LLM
from .notify.telegram import Telegram
from .orchestrator import Agent
from .sources.base import Source, make_http
from .sources.flru import FlruSource
from .sources.freelancer_com import FreelancerClient, FreelancerSource
from .sources.rss import RssSource


@dataclass
class App:
    settings: Settings
    store: Store
    guard: BudgetGuard
    llm: LLM
    brain: Brain
    agent: Agent
    http: httpx.AsyncClient
    anthropic: anthropic.AsyncAnthropic

    async def aclose(self) -> None:
        await self.http.aclose()
        await self.anthropic.close()
        self.store.close()


def build_sources(settings: Settings, http: httpx.AsyncClient,
                  freelancer: FreelancerClient) -> list[Source]:
    src = settings.sources
    sources: list[Source] = []
    if src.flru.enabled:
        sources.append(FlruSource(http, src.flru.feeds, src.flru.pro_account))
    if src.freelancer.enabled:
        sources.append(FreelancerSource(freelancer, src.freelancer.queries,
                                        src.freelancer.per_query_limit, src.freelancer.job_ids))
    sources += [RssSource(http, f.name, f.url) for f in src.rss]
    return sources


def build(
    settings: Settings,
    secrets: Secrets,
    *,
    store: Store | None = None,
    anthropic_client: anthropic.AsyncAnthropic | None = None,
    http: httpx.AsyncClient | None = None,
    query_fn: QueryFn | None = None,
) -> App:
    store = store or Store(settings.db_path)
    http = http or make_http()
    client = anthropic_client or anthropic.AsyncAnthropic(api_key=secrets.anthropic_api_key)
    guard = BudgetGuard(store, settings.budget)
    llm = LLM(client, store, guard, settings.models.server_side_fallback)
    brain = Brain(llm, settings)
    executor = Executor(settings, store, guard, query_fn) if query_fn else Executor(settings, store, guard)
    freelancer = FreelancerClient(http, secrets.freelancer_oauth_token)
    telegram = None
    if settings.telegram.enabled and secrets.telegram_bot_token:
        telegram = Telegram(http, secrets.telegram_bot_token, settings.telegram.chat_id)
    if settings.economics.live_fx:
        fx.apply_cached(store, settings.economics)
    agent = Agent(settings, store, guard, brain, executor, build_sources(settings, http, freelancer),
                  telegram, freelancer, fx_http=http if settings.economics.live_fx else None)
    return App(settings, store, guard, llm, brain, agent, http, client)
