"""Command line: init, check, scan, once, run, stats, paid, estimate, smoke-llm."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from . import prefilter
from .agents.executor import sandbox_problem
from .app import build, build_sources
from .budget import BudgetGuard, format_report
from .config import Secrets, Settings, load_dotenv, load_settings
from .db import Store
from .economics import platform_fee, to_usd
from .models import Job
from .sources.base import make_http
from .sources.freelancer_com import FreelancerClient

HERE = Path(__file__).resolve().parent


def _load(args: argparse.Namespace) -> tuple[Settings, Secrets]:
    config = Path(args.config)
    load_dotenv(config.parent / ".env")
    if not config.exists():
        sys.exit(f"{config} not found. Run: freelance-agent init")
    return load_settings(config), Secrets.from_env()


def cmd_init(args: argparse.Namespace) -> None:
    target = Path(args.config)
    for name, dest in (("config.example.yaml", target), ("env.example", target.parent / ".env")):
        src = HERE / "templates" / name
        if dest.exists():
            print(f"exists: {dest}")
        else:
            shutil.copy(src, dest)
            print(f"created: {dest}")
    print("Fill in the profile in config.yaml and the keys in .env, then: freelance-agent check")


async def cmd_check(args: argparse.Namespace) -> None:
    settings, secrets = _load(args)
    ok = True

    def report(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        print(("OK   " if good else "FAIL ") + text)

    report(bool(secrets.anthropic_api_key), "ANTHROPIC_API_KEY is set")
    if secrets.anthropic_api_key:
        client = anthropic.AsyncAnthropic(api_key=secrets.anthropic_api_key)
        for model in sorted({settings.models.triage, settings.models.writer, settings.models.executor}):
            try:
                await client.models.retrieve(model)  # free call
                report(True, f"model {model} available")
            except anthropic.APIStatusError as e:
                report(False, f"model {model}: {e.status_code} {e.message}")
        await client.close()

    http = make_http()
    try:
        fl = FreelancerClient(http, secrets.freelancer_oauth_token)
        for src in build_sources(settings, http, fl):
            try:
                jobs = await src.fetch()
                report(len(jobs) > 0, f"source {src.name}: {len(jobs)} jobs")
            except Exception as e:  # noqa: BLE001 - report any failure of a source
                report(False, f"source {src.name}: {e}")
        if secrets.freelancer_oauth_token:
            try:
                report(True, f"freelancer.com token OK, user id {await fl.self_id()}")
            except Exception as e:  # noqa: BLE001
                report(False, f"freelancer.com token: {e}")
        else:
            print("INFO FREELANCER_OAUTH_TOKEN not set: Freelancer.com bids will be manual")

        if settings.telegram.enabled:
            if not secrets.telegram_bot_token:
                report(False, "TELEGRAM_BOT_TOKEN is not set")
            else:
                from .notify.telegram import Telegram

                me = await Telegram(http, secrets.telegram_bot_token, None).get_me()
                report(True, f"telegram bot @{me.get('username')}")
                report(settings.telegram.chat_id is not None,
                       "telegram.chat_id is set (send /start to the bot while `run` is active to learn it)")
    finally:
        await http.aclose()

    if settings.executor.sandbox:
        problem = sandbox_problem()
        report(problem is None, "OS sandbox for the executor" + (f": {problem}" if problem else ""))
    else:
        print("WARN executor.sandbox is off")
    sys.exit(0 if ok else 1)


async def cmd_scan(args: argparse.Namespace) -> None:
    """Free market scan: fetch + prefilter, no LLM, nothing stored."""
    settings, secrets = _load(args)
    http = make_http()
    now = datetime.now(UTC)
    passed: list[tuple[Job, float | None]] = []
    reasons: dict[str, int] = {}
    total = 0
    try:
        fl = FreelancerClient(http, secrets.freelancer_oauth_token)
        for src in build_sources(settings, http, fl):
            try:
                jobs = await src.fetch()
            except Exception as e:  # noqa: BLE001
                print(f"{src.name}: {e}")
                continue
            total += len(jobs)
            for job in jobs:
                good, why = prefilter.check(job, settings.filters, settings.economics, now)
                if good:
                    usd = to_usd(job.budget_max or job.budget_min, job.currency, job.usd_rate,
                                 settings.economics.fx_to_usd)
                    passed.append((job, usd))
                else:
                    key = re.sub(r"«[^»]*»|[~$]?\d[\d.,]*", "N", why)
                    reasons[key] = reasons.get(key, 0) + 1
    finally:
        await http.aclose()
    print(f"Fetched {total}, passed free prefilter {len(passed)}")
    for job, usd in passed:
        budget = f"${usd:.0f}" if usd is not None else "  ?"
        print(f"  {budget:>6}  {job.source:<10} {job.title[:70]}  {job.url}")
    if reasons:
        print("Filtered out:", ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))


async def _with_app(args: argparse.Namespace, forever: bool) -> None:
    settings, secrets = _load(args)
    if not secrets.anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY is not set (.env)")
    app = build(settings, secrets)
    try:
        if forever:
            await app.agent.run_forever()
        else:
            stats = await app.agent.cycle()
            print(stats)
            print(format_report(app.guard.report()))
    finally:
        await app.aclose()


def cmd_stats(args: argparse.Namespace) -> None:
    settings, _ = _load(args)
    store = Store(settings.db_path)
    print(format_report(BudgetGuard(store, settings.budget).report()))
    print("Funnel:", ", ".join(f"{k} {v}" for k, v in sorted(store.status_counts().items())))


def cmd_paid(args: argparse.Namespace) -> None:
    settings, _ = _load(args)
    store = Store(settings.db_path)
    rec = store.get_job(args.job_id)
    if rec is None:
        sys.exit(f"job #{args.job_id} not found")
    currency = (args.currency or rec.currency or "USD").upper()
    rate = rec.usd_rate if currency == (rec.currency or "").upper() and rec.usd_rate else \
        settings.economics.fx_to_usd.get(currency)
    if rate is None:
        sys.exit(f"no rate for {currency} in economics.fx_to_usd")
    usd = args.amount * rate
    store.add_ledger("revenue", usd, rec.id, f"{args.amount:g} {currency} via CLI")
    fee = platform_fee(usd, rec.source, settings.economics)
    if fee:
        store.add_ledger("fee", fee, rec.id, f"{rec.source} fee")
    from .models import JobStatus

    store.set_status(rec.id, JobStatus.PAID)
    print(f"recorded ${usd:.2f} revenue (fee ${fee:.2f}) for #{rec.id}")


def cmd_estimate(args: argparse.Namespace) -> None:
    """Break-even arithmetic for the current settings."""
    settings, _ = _load(args)
    e, b = settings.economics, settings.budget
    fixed = sum(b.fixed_monthly_costs_usd.values())
    price = args.price
    fee = platform_fee(price, args.source, e)
    exec_cost = e.execution_cost_usd[args.complexity]
    per_win = price - fee - exec_cost
    win_rate = e.prior_win_rate
    bids_per_win = 1 / win_rate
    acquisition = bids_per_win * e.proposal_cost_estimate_usd
    margin = per_win - acquisition
    print(f"Job ${price:.0f} on {args.source}, complexity {args.complexity}")
    print(f"  platform fee           -{fee:.2f}")
    print(f"  AI execution (budget)  -{exec_cost:.2f}")
    print(f"  proposals per win      {bids_per_win:.0f} x ${e.proposal_cost_estimate_usd:.2f} = -{acquisition:.2f}")
    print(f"  margin per won job     {margin:.2f}")
    print(f"Fixed monthly costs ${fixed:.2f}: break-even at {fixed / margin:.1f} such jobs/month"
          if margin > 0 else "This job size never pays off with these settings.")


async def cmd_smoke_llm(args: argparse.Namespace) -> None:
    """One real triage call (fractions of a cent) to verify the key, model and cost accounting."""
    settings, secrets = _load(args)
    store = Store(":memory:")
    app = build(settings, secrets, store=store)
    try:
        job = Job(source="flru", external_id="smoke", url="https://www.fl.ru/projects/0/",
                  title="Написать парсер цен с сайта в CSV",
                  description="Нужен скрипт на Python, который собирает названия и цены товаров "
                              "из каталога (около 500 позиций, 20 страниц) и сохраняет в CSV.",
                  budget_max=5000, currency="RUB")
        job_id, _ = store.upsert_job(job)
        a = await app.brain.assess(job, job_id)
        print(a.model_dump_json(indent=2))
        print(f"cost: ${store.llm_spend():.5f}")
    finally:
        await app.aclose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="freelance-agent")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create config.yaml and .env from templates")
    sub.add_parser("check", help="verify keys, models, sources, Telegram, sandbox")
    sub.add_parser("scan", help="free dry run: fetch and prefilter jobs, no LLM")
    sub.add_parser("once", help="one full cycle")
    sub.add_parser("run", help="run forever")
    sub.add_parser("stats", help="profit report")
    sub.add_parser("smoke-llm", help="one real cheap triage call")
    p = sub.add_parser("paid", help="record a payment")
    p.add_argument("job_id", type=int)
    p.add_argument("amount", type=float)
    p.add_argument("currency", nargs="?")
    p = sub.add_parser("estimate", help="break-even arithmetic")
    p.add_argument("--price", type=float, default=100.0)
    p.add_argument("--source", default="freelancer")
    p.add_argument("--complexity", default="small", choices=["trivial", "small", "medium", "large"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    handlers = {
        "init": cmd_init,
        "stats": cmd_stats,
        "paid": cmd_paid,
        "estimate": cmd_estimate,
    }
    async_handlers = {
        "check": cmd_check,
        "scan": cmd_scan,
        "smoke-llm": cmd_smoke_llm,
    }
    if args.cmd in handlers:
        handlers[args.cmd](args)
    elif args.cmd in async_handlers:
        asyncio.run(async_handlers[args.cmd](args))
    else:
        try:
            asyncio.run(_with_app(args, forever=args.cmd == "run"))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
