"""The full cycle: find -> triage -> bid -> talk -> do the work -> review -> deliver -> get paid.

Every outgoing action goes through `route_*`, which either performs it (autonomy allows it and
the platform has an API for it) or asks you in Telegram with a one-tap card.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2 as httpx

from . import prefilter
from .agents.brain import Brain
from .agents.executor import Executor, files_for_review, pack
from .agents.schemas import Assessment
from .budget import BudgetExceeded, BudgetGuard, format_report
from .config import Settings
from .db import JobRecord, Store
from .economics import decide, platform_fee, usd_rate_for, win_probability
from .llm import LLMError
from .models import JobStatus
from .notify.telegram import Buttons, Telegram, TelegramError, callback, esc, parse_callback
from .sources.base import Source
from .sources.freelancer_com import FreelancerClient, FreelancerError

log = logging.getLogger(__name__)

# Freelancer.com bid award statuses.
AWARDED = {"awarded"}
ACCEPTED = {"accepted"}
LOST_STATUSES = {"rejected", "revoked", "canceled", "cancelled", "retracted"}
BID_TIMEOUT = timedelta(days=21)


@dataclass
class CycleStats:
    fetched: int = 0
    new: int = 0
    filtered: int = 0
    assessed: int = 0
    proposals: int = 0
    errors: int = 0


class Agent:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        guard: BudgetGuard,
        brain: Brain,
        executor: Executor,
        sources: list[Source],
        telegram: Telegram | None,
        freelancer: FreelancerClient | None,
    ) -> None:
        self.s = settings
        self.store = store
        self.guard = guard
        self.brain = brain
        self.executor = executor
        self.sources = sources
        self.tg = telegram
        self.fl = freelancer
        self._busy: set[int] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()

    # =========================================================================== finding work

    async def scan(self, stats: CycleStats) -> None:
        for src in self.sources:
            try:
                jobs = await src.fetch()
            except (httpx.HTTPError, FreelancerError) as e:
                log.warning("source %s failed: %s", src.name, e)
                stats.errors += 1
                continue
            stats.fetched += len(jobs)
            for job in jobs:
                _, is_new = self.store.upsert_job(job)
                stats.new += is_new

    async def triage_new(self, stats: CycleStats) -> None:
        now = self.store.clock()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        budget_left = self.s.max_new_jobs_per_cycle
        for rec in self.store.jobs_by_status(JobStatus.NEW):
            job = rec.to_job()
            ok, reason = prefilter.check(job, self.s.filters, self.s.economics, now)
            if not ok:
                self.store.set_status(rec.id, JobStatus.FILTERED, reason)
                stats.filtered += 1
                continue
            allowed, why = self.guard.bidding_allowed()
            if not allowed:
                log.info("bidding stopped: %s", why)
                return
            if (rec.source == "freelancer" and self.store.count_submitted_since(
                    "freelancer", day_start) >= self.s.sources.freelancer.max_bids_per_day):
                continue  # stays NEW, retried tomorrow while still fresh
            if budget_left <= 0:
                return
            budget_left -= 1
            try:
                await self.evaluate(rec)
                stats.assessed += 1
                status = self._job(rec.id).status
                stats.proposals += status in (JobStatus.AWAITING_APPROVAL, JobStatus.SUBMITTED)
            except BudgetExceeded as e:
                await self.alert_once("budget", f"⛔ {esc(str(e))}")
                return
            except (LLMError, ValueError) as e:
                log.warning("job %s: %s", rec.id, e)
                self.store.set_status(rec.id, JobStatus.FAILED, str(e)[:300])
                stats.errors += 1

    async def evaluate(self, rec: JobRecord) -> None:
        job = rec.to_job()
        a = await self.brain.assess(job, rec.id)
        bids, wins = self.store.win_stats(rec.source)
        p = win_probability(bids, wins, job.bid_count, self.s.economics)
        d = decide(job, a, self.s.profile, self.s.economics, p)
        self.store.update_job(rec.id, score=a.fit_score, assessment=a.model_dump(),
                              decision=d.to_dict(), requirements=a.requirements_summary)
        if not d.bid:
            self.store.set_status(rec.id, JobStatus.SKIPPED, d.reason)
            return
        if a.fit_score < self.s.filters.min_fit_score:
            self.store.set_status(rec.id, JobStatus.SKIPPED,
                                  f"соответствие профилю {a.fit_score} < {self.s.filters.min_fit_score}")
            return
        draft = await self.brain.write_proposal(job, rec.id, a, d)
        self.store.update_job(rec.id, proposal=draft.text, bid_amount=d.bid_amount,
                              bid_period_days=draft.period_days)
        await self.route_proposal(rec.id)

    # ================================================================================ bidding

    def _can_auto_bid(self, rec: JobRecord) -> bool:
        auto = self.s.autonomy
        if rec.source != "freelancer" or not (self.fl and self.fl.can_write):
            return False
        if auto.mode == "auto":
            return True
        price = (rec.decision or {}).get("price_usd", 0.0)
        return (auto.mode == "semi" and (rec.score or 0) >= auto.auto_bid_min_score
                and price <= auto.auto_bid_max_usd)

    async def route_proposal(self, job_id: int) -> None:
        rec = self._job(job_id)
        if self._can_auto_bid(rec):
            await self.submit_bid(rec, rec.proposal or "")
            await self.notify(f"🤖 Ставка отправлена автоматически\n{self._card_header(rec)}")
            return
        self.store.set_status(job_id, JobStatus.AWAITING_APPROVAL)
        pid = self.store.create_pending("proposal", job_id, {"text": rec.proposal})
        await self._send_proposal_card(pid, rec, rec.proposal or "")

    async def _send_proposal_card(self, pid: int, rec: JobRecord, text: str) -> None:
        api = rec.source == "freelancer" and self.fl is not None and self.fl.can_write
        action = "✅ Отправить ставку" if api else "✅ Я откликнулся"
        hint = ("" if api else "\nОткройте ссылку, вставьте текст отклика и нажмите «Я откликнулся».")
        body = (
            f"🆕 {self._card_header(rec)}\n\n"
            f"<b>Отклик</b> (ответьте на это сообщение своим текстом, чтобы заменить):\n"
            f"<pre>{esc(text)}</pre>{hint}"
        )
        buttons: Buttons = [[(action, callback("send", pid)), ("❌ Пропустить", callback("skip", pid))]]
        await self._card(pid, body, buttons)

    def _card_header(self, rec: JobRecord) -> str:
        d = rec.decision or {}
        a = rec.assessment or {}
        budget = "не указан"
        if rec.budget_min or rec.budget_max:
            lo = f"{rec.budget_min:g}" if rec.budget_min else ""
            hi = f"{rec.budget_max:g}" if rec.budget_max else ""
            budget = f"{lo}–{hi}".strip("–") + f" {rec.currency or ''}"
        lines = [
            f"<b>{esc(rec.title)}</b> · #{rec.id}",
            f"{rec.source} · бюджет {esc(budget)} · откликов {rec.bid_count if rec.bid_count is not None else '?'}",
        ]
        if a:
            lines.append(f"Оценка {rec.score}/100 · сложность {a.get('complexity')} · "
                         f"~{a.get('estimated_human_hours')} ч работы человека")
        if d.get("bid_amount"):
            lines.append(
                f"Цена {d['bid_amount']:g} {d.get('currency')} (~${d.get('price_usd', 0):.0f}) · "
                f"срок {rec.bid_period_days or '?'} дн. · бюджет ИИ ${d.get('execution_budget_usd', 0):.2f}"
            )
            lines.append(
                f"Прибыль при победе ~${d.get('profit_if_won_usd', 0):.0f} · шанс "
                f"{d.get('win_probability', 0):.0%} · ожидаемая ${d.get('expected_profit_usd', 0):.2f}"
            )
        lines.append(esc(rec.url))
        if a.get("requirements_summary"):
            lines.append(f"\n<b>Что нужно</b>: {esc(a['requirements_summary'])}")
        return "\n".join(lines)

    async def submit_bid(self, rec: JobRecord, text: str) -> None:
        assert self.fl is not None
        bid_id = await self.fl.place_bid(int(rec.external_id), float(rec.bid_amount or 0),
                                         int(rec.bid_period_days or 3), text)
        self.store.update_job(rec.id, external_bid_id=str(bid_id), proposal=text)
        self.store.set_status(rec.id, JobStatus.SUBMITTED)

    # ========================================================================== platform sync

    async def sync_freelancer(self) -> None:
        if not (self.fl and self.fl.can_write):
            return
        await self._sync_bids()
        for rec in self.store.jobs_by_status(JobStatus.SUBMITTED, JobStatus.WON,
                                             JobStatus.IN_PROGRESS, JobStatus.REVIEW,
                                             JobStatus.DELIVERED):
            if rec.source != "freelancer":
                continue
            try:
                await self._sync_messages(rec)
                if rec.status in (JobStatus.IN_PROGRESS, JobStatus.REVIEW, JobStatus.DELIVERED):
                    await self._sync_payments(rec)
            except FreelancerError as e:
                log.warning("freelancer sync job %s: %s", rec.id, e)

    async def _sync_bids(self) -> None:
        assert self.fl is not None
        submitted = [r for r in self.store.jobs_by_status(JobStatus.SUBMITTED)
                     if r.source == "freelancer" and r.external_bid_id]
        if not submitted:
            return
        try:
            bids = await self.fl.get_bids([int(r.external_bid_id or 0) for r in submitted])
        except FreelancerError as e:
            log.warning("bid sync failed: %s", e)
            return
        by_id = {str(b.get("id")): b for b in bids}
        now = self.store.clock()
        for rec in submitted:
            status = str((by_id.get(rec.external_bid_id or "") or {}).get("award_status") or "").lower()
            if status in AWARDED:
                if not any(p.job_id == rec.id for p in self.store.open_pending("award")):
                    await self.route_award(rec)
            elif status in ACCEPTED:
                self.store.set_status(rec.id, JobStatus.WON)
                self.start_work(rec.id)
            elif status in LOST_STATUSES:
                self.store.set_status(rec.id, JobStatus.LOST, f"ставка: {status}")
            elif rec.submitted_at and now - datetime.fromisoformat(rec.submitted_at) > BID_TIMEOUT:
                self.store.set_status(rec.id, JobStatus.LOST, "нет ответа 3 недели")

    async def route_award(self, rec: JobRecord) -> None:
        if self.s.autonomy.mode == "auto":
            await self.accept_award(rec)
            await self.notify(f"🏆 Заказ #{rec.id} выигран и принят автоматически. Начинаю работу.")
            return
        pid = self.store.create_pending("award", rec.id, {})
        await self._card(pid, f"🏆 <b>Вас выбрали!</b>\n{self._card_header(rec)}\n\n"
                              "Принять заказ и запустить выполнение?",
                         [[("▶️ Принять и начать", callback("accept", pid)),
                           ("❌ Отказаться", callback("decline", pid))]])

    async def accept_award(self, rec: JobRecord) -> None:
        if rec.source == "freelancer" and self.fl and rec.external_bid_id:
            await self.fl.accept_bid(int(rec.external_bid_id))
        self.store.set_status(rec.id, JobStatus.WON)
        self.start_work(rec.id)

    async def _sync_messages(self, rec: JobRecord) -> None:
        assert self.fl is not None
        thread_id = rec.thread_id
        if not thread_id:
            found = await self.fl.project_thread_id(int(rec.external_id))
            if not found:
                return
            thread_id = str(found)
            self.store.update_job(rec.id, thread_id=thread_id)
        me = await self.fl.self_id()
        fresh_incoming: list[str] = []
        for m in await self.fl.thread_messages(int(thread_id)):
            if self.store.has_message(rec.id, m.id):
                continue
            direction = "out" if m.from_user == me else "in"
            if direction == "out" and self.store.claim_outgoing(rec.id, m.text, m.id):
                continue  # sent by the agent itself, already in the history
            self.store.add_message(rec.id, direction, m.text, external_id=m.id)
            if direction == "in":
                fresh_incoming.append(m.text)
            else:
                fresh_incoming.clear()  # you already answered on the website
        if fresh_incoming:
            await self.on_client_message(rec.id, "\n\n".join(fresh_incoming), stored=True)

    async def _sync_payments(self, rec: JobRecord) -> None:
        assert self.fl is not None
        milestones = await self.fl.milestones(int(rec.external_id))
        me = await self.fl.self_id()
        released = sum(float(m.get("amount") or 0) for m in milestones
                       if str(m.get("status", "")).lower() == "cleared"
                       and m.get("bidder_id") in (None, me))
        if released <= 0:
            return
        rate = usd_rate_for(rec.to_job(), self.s.economics.fx_to_usd) or 1.0
        already = float(self.store.kv_get(f"paid:{rec.id}", "0") or 0)
        if released > already + 1e-6:
            self.record_payment(rec, (released - already) * rate,
                                note=f"milestones {released:g} {rec.currency}")
            self.store.kv_set(f"paid:{rec.id}", str(released))
            if released >= float(rec.bid_amount or 0) * 0.99:
                self.store.set_status(rec.id, JobStatus.PAID)
            await self.notify(f"💰 Оплата по #{rec.id}: {released - already:g} {rec.currency}\n"
                              f"<pre>{esc(format_report(self.guard.report()))}</pre>")

    def record_payment(self, rec: JobRecord, amount_usd: float, note: str = "") -> None:
        fee = platform_fee(amount_usd, rec.source, self.s.economics)
        self.store.add_ledger("revenue", amount_usd, rec.id, note)
        if fee:
            self.store.add_ledger("fee", fee, rec.id, f"{rec.source} fee")

    # ============================================================================= talking

    async def on_client_message(self, job_id: int, text: str, stored: bool = False) -> None:
        rec = self._job(job_id)
        if not stored:
            self.store.add_message(job_id, "in", text)
        history = self.store.messages_for(job_id)[:-1]
        bid = (f"{rec.bid_amount:g} {rec.currency}, {rec.bid_period_days} days\n{rec.proposal or ''}"
               if rec.bid_amount else rec.proposal or "(no bid recorded)")
        try:
            r = await self.brain.draft_reply(rec.to_job(), job_id, rec.requirements or "", bid,
                                             history, text)
        except (BudgetExceeded, LLMError) as e:
            await self.notify(f"⚠️ #{job_id}: не смог подготовить ответ клиенту: {esc(str(e))}")
            return
        if r.requirements_update.strip() or r.acceptance_criteria:
            req = (rec.requirements or "").strip()
            if r.requirements_update.strip():
                req += f"\n\nОбновление от клиента:\n{r.requirements_update.strip()}"
            if r.acceptance_criteria:
                req += "\n\nКритерии приёмки:\n" + "\n".join(f"- {c}" for c in r.acceptance_criteria)
            self.store.update_job(job_id, requirements=req.strip())

        api = rec.source == "freelancer" and self.fl is not None and self.fl.can_write
        if api and self.s.autonomy.mode == "auto" and not r.needs_human and r.intent != "scam_risk":
            await self.send_client_message(rec, r.reply)
            await self.notify(f"🤖 #{job_id} клиенту отправлен ответ:\n<pre>{esc(r.reply)}</pre>")
            return
        pid = self.store.create_pending("reply", job_id, {"text": r.reply, "intent": r.intent})
        warn = f"\n⚠️ <b>Нужно ваше решение</b>: {esc(r.human_reason)}" if r.needs_human else ""
        award_hint = ""
        buttons: Buttons = [[("✅ Отправить" if api else "✅ Я отправил", callback("reply_send", pid)),
                             ("❌ Не отвечать", callback("reply_skip", pid))]]
        if not api and rec.status == JobStatus.SUBMITTED:
            buttons.append([("🏆 Меня выбрали", callback("won", pid)),
                            ("👎 Отказ", callback("lost", pid))])
            if r.intent == "award":
                award_hint = "\nПохоже, клиент выбрал вас — нажмите «Меня выбрали», чтобы начать."
        body = (f"💬 <b>#{job_id} {esc(rec.title)}</b> · {r.intent}{warn}\n\n"
                f"<b>Клиент</b>:\n{esc(text)}\n\n<b>Черновик ответа</b> (ответьте на это "
                f"сообщение, чтобы заменить):\n<pre>{esc(r.reply)}</pre>{award_hint}")
        await self._card(pid, body, buttons)

    async def send_client_message(self, rec: JobRecord, text: str) -> None:
        if rec.source == "freelancer" and self.fl and self.fl.can_write:
            thread_id = rec.thread_id or str(await self.fl.project_thread_id(int(rec.external_id)) or "")
            if not thread_id:
                raise FreelancerError("no message thread for this project yet")
            await self.fl.post_message(int(thread_id), text)
            self.store.update_job(rec.id, thread_id=thread_id)
        self.store.add_message(rec.id, "out", text)

    # ============================================================================== working

    def start_work(self, job_id: int, notes: list[str] | None = None) -> None:
        if job_id in self._busy:
            return
        self._busy.add(job_id)
        task = asyncio.create_task(self._work(job_id, notes))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _work(self, job_id: int, notes: list[str] | None) -> None:
        try:
            await self.do_work(job_id, notes)
        except Exception as e:  # never let a crash in one job kill the agent
            log.exception("work on job %s crashed", job_id)
            rec = self.store.get_job(job_id)
            if rec and rec.status == JobStatus.IN_PROGRESS:
                # Not resumed automatically on restart: you decide with /work.
                self.store.set_status(job_id, JobStatus.WON, f"сбой: {e}"[:300])
            await self.notify(f"❌ #{job_id}: сбой выполнения: {esc(str(e))}. Повторить: /work {job_id}")
        finally:
            self._busy.discard(job_id)

    def execution_budget(self, rec: JobRecord, rework: bool) -> float:
        planned = float((rec.decision or {}).get("execution_budget_usd") or 0.0)
        if planned <= 0:
            a = Assessment.model_validate(rec.assessment) if rec.assessment else None
            complexity = a.complexity if a else "small"
            planned = self.s.economics.execution_cost_usd[complexity]
        left = planned - self.store.llm_spend(job_id=rec.id, purposes=("execute",))
        if rework:
            # A requested rework may use up to half the original budget again.
            return round(max(left, planned * 0.5), 2)
        return round(left, 2)

    async def do_work(self, job_id: int, notes: list[str] | None = None) -> None:
        rec = self._job(job_id)
        budget = self.execution_budget(rec, rework=bool(notes))
        if budget <= 0.05:
            await self.notify(f"⛔ #{job_id}: бюджет на выполнение исчерпан. "
                              f"Отправьте /work {job_id} после увеличения economics.max_execution_usd.")
            return
        self.store.set_status(job_id, JobStatus.IN_PROGRESS)
        await self.notify(f"🛠 #{job_id} {esc(rec.title)}: начинаю работу, лимит ${budget:.2f}")
        history = self.store.messages_for(job_id)
        job = rec.to_job()
        requirements = rec.requirements or ""
        session = self.store.kv_get(f"session:{job_id}")
        try:
            res = await self.executor.run(job_id, job, requirements, history, budget,
                                          fix_notes=notes, resume=session if notes else None)
        except BudgetExceeded as e:
            self.store.set_status(job_id, JobStatus.WON, str(e))
            await self.notify(f"⛔ #{job_id}: {esc(str(e))}")
            return
        if res.session_id:
            self.store.kv_set(f"session:{job_id}", res.session_id)
        if not res.ok:
            self.store.set_status(job_id, JobStatus.WON, res.error)
            await self.notify(f"❌ #{job_id}: {esc(res.error or 'ошибка')} (потрачено ${res.cost_usd:.2f}). "
                              f"Повторить: /work {job_id}")
            return

        ws = self.executor.workspace(job_id)
        verdict = None
        for attempt in range(self.s.executor.review_retries + 1):
            listing, contents = files_for_review(ws)
            try:
                verdict = await self.brain.review(job, job_id, requirements, listing, contents,
                                                  res.report)
            except (BudgetExceeded, LLMError) as e:
                await self.notify(f"⚠️ #{job_id}: проверка не выполнена: {esc(str(e))}")
                break
            if verdict.passed or attempt == self.s.executor.review_retries:
                break
            fix_budget = self.execution_budget(self._job(job_id), rework=False)
            if fix_budget <= 0.05:
                break
            try:
                res = await self.executor.run(job_id, job, requirements, history, fix_budget,
                                              fix_notes=verdict.issues, resume=res.session_id)
            except BudgetExceeded as e:
                await self.notify(f"⚠️ #{job_id}: доработка по замечаниям не запущена: {esc(str(e))}")
                break
            if res.session_id:
                self.store.kv_set(f"session:{job_id}", res.session_id)
            if not res.ok:
                break

        archive = pack(ws, f"job-{job_id}")
        message = verdict.delivery_message if verdict else ""
        self.store.set_status(job_id, JobStatus.REVIEW)
        spent = self.store.llm_spend(job_id=job_id)
        issues = ""
        if verdict and not verdict.passed:
            issues = "\n⚠️ <b>Замечания проверки</b>:\n" + "\n".join(f"• {esc(i)}" for i in verdict.issues)
        api = rec.source == "freelancer" and self.fl is not None and self.fl.can_write
        if api and self.s.autonomy.mode == "auto" and verdict and verdict.passed:
            try:
                await self.deliver(self._job(job_id), archive, message)
            except FreelancerError as e:
                issues += f"\n⚠️ Автосдача не удалась: {esc(str(e))}"
            else:
                await self.notify(f"📦 #{job_id} сдан клиенту автоматически. Затраты ИИ ${spent:.2f}")
                return
        pid = self.store.create_pending("delivery", job_id, {"text": message, "archive": str(archive)})
        if self.tg:
            try:
                await self.tg.send_document(archive, f"📦 #{job_id} результат")
            except TelegramError as e:
                log.warning("send_document failed: %s", e)
        body = (f"📦 <b>#{job_id} готово</b> · {esc(rec.title)}\nЗатраты ИИ на заказ: ${spent:.2f}{issues}\n\n"
                f"<b>Сообщение клиенту</b> (ответьте на это сообщение текстом правок, "
                f"чтобы отправить на доработку):\n<pre>{esc(message)}</pre>")
        buttons: Buttons = [[("✅ Отправить клиенту" if api else "✅ Я отправил клиенту",
                              callback("deliver", pid)),
                             ("🔁 Доработать по замечаниям", callback("rework", pid))]]
        await self._card(pid, body, buttons)

    async def deliver(self, rec: JobRecord, archive_path: str | Path, message: str) -> None:
        archive = Path(archive_path)
        if rec.source == "freelancer" and self.fl and self.fl.can_write:
            thread_id = rec.thread_id or str(await self.fl.project_thread_id(int(rec.external_id)) or "")
            if not thread_id:
                raise FreelancerError("no message thread for this project yet")
            await self.fl.post_attachment(int(thread_id), archive)
            for m in await self.fl.milestones(int(rec.external_id)):
                if str(m.get("status", "")).lower() == "frozen":
                    try:
                        await self.fl.request_release(int(m["id"]))
                    except FreelancerError as e:
                        log.warning("request_release %s: %s", m.get("id"), e)
        if message:
            await self.send_client_message(self._job(rec.id), message)
        self.store.set_status(rec.id, JobStatus.DELIVERED)

    # ============================================================================= telegram

    async def _card(self, pid: int, body: str, buttons: Buttons) -> None:
        if not self.tg:
            log.info("approval needed (pending %s); Telegram is off", pid)
            return
        try:
            mid = await self.tg.send(body, buttons)
        except TelegramError as e:
            log.warning("telegram send failed: %s", e)
            return
        self.store.update_pending(pid, tg_message_id=mid)

    async def notify(self, text_html: str) -> None:
        if not self.tg:
            log.info("%s", text_html)
            return
        try:
            await self.tg.send(text_html)
        except TelegramError as e:
            log.warning("telegram notify failed: %s", e)

    async def alert_once(self, key: str, text_html: str) -> None:
        today = self.store.clock().date().isoformat()
        if self.store.kv_get(f"alert:{key}") != today:
            self.store.kv_set(f"alert:{key}", today)
            await self.notify(text_html)

    async def handle_update(self, update: dict[str, Any]) -> None:
        assert self.tg is not None
        if cq := update.get("callback_query"):
            chat = (cq.get("message") or {}).get("chat", {}).get("id")
            if chat != self.tg.chat_id:
                return
            parsed = parse_callback(cq.get("data", ""))
            if parsed:
                answer = await self.on_button(*parsed)
                await self.tg.answer_callback(cq["id"], answer)
                if answer and (msg := cq.get("message")):
                    await self.tg.clear_buttons(msg["message_id"])
            return
        msg = update.get("message") or {}
        chat = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        if self.tg.chat_id is None:
            if text.startswith("/start"):
                log.warning("Telegram chat_id is %s: put it into config.yaml (telegram.chat_id)", chat)
                await self.tg.http.post(f"{self.tg.base}/sendMessage", json={
                    "chat_id": chat, "text": f"Ваш chat_id: {chat}. Впишите его в config.yaml "
                                            "(telegram.chat_id) и перезапустите агента."})
            return
        if chat != self.tg.chat_id or not text:
            return
        if reply_to := (msg.get("reply_to_message") or {}).get("message_id"):
            pending = self.store.pending_by_tg_message(int(reply_to))
            if pending:
                await self.on_reply(pending.id, text)
                return
        await self.on_command(text)

    async def on_button(self, action: str, pid: int) -> str:
        p = self.store.get_pending(pid)
        if not p or p.resolved_at:
            return "Уже обработано"
        rec = self._job(p.job_id)
        text = str(p.payload.get("text") or "")
        api = rec.source == "freelancer" and self.fl is not None and self.fl.can_write
        try:
            if action == "send" and p.kind == "proposal":
                if not self.store.resolve_pending(pid, action):
                    return "Уже обработано"
                if api:
                    await self.submit_bid(rec, text)
                    return "Ставка отправлена"
                self.store.update_job(rec.id, proposal=text)
                self.store.set_status(rec.id, JobStatus.SUBMITTED)
                await self._send_followup(rec.id)
                return "Отмечено как отправленное"
            if action == "skip":
                self.store.resolve_pending(pid, action)
                self.store.set_status(rec.id, JobStatus.SKIPPED, "отклонено вами")
                return "Пропущено"
            if action == "reply_send":
                self.store.resolve_pending(pid, action)
                await self.send_client_message(rec, text)
                return "Ответ отправлен" if api else "Отмечено"
            if action == "reply_skip":
                self.store.resolve_pending(pid, action)
                return "Без ответа"
            if action in ("won", "accept"):
                self.store.resolve_pending(pid, action)
                self.store.resolve_pending_for_job(rec.id, ("followup",), "won")
                await self.accept_award(rec)
                return "Принято, начинаю работу"
            if action in ("lost", "decline"):
                self.store.resolve_pending(pid, action)
                self.store.resolve_pending_for_job(rec.id, ("followup",), action)
                self.store.set_status(rec.id, JobStatus.LOST, "отказ")
                return "Отмечено"
            if action == "deliver":
                self.store.resolve_pending(pid, action)
                await self.deliver(rec, p.payload["archive"], text)
                if not api:
                    await self._send_paid_card(rec.id)
                return "Сдано"
            if action == "rework":
                self.store.resolve_pending(pid, action)
                notes = ["Improve the deliverable against the requirements and fix all issues "
                         "listed in DELIVERY_NOTES.md or found while re-checking."]
                self.start_work(rec.id, notes)
                return "Отправил на доработку"
            if action == "paid":
                self.store.resolve_pending(pid, action)
                rate = usd_rate_for(rec.to_job(), self.s.economics.fx_to_usd) or 1.0
                self.record_payment(rec, float(rec.bid_amount or 0) * rate, "confirmed in Telegram")
                self.store.set_status(rec.id, JobStatus.PAID)
                return "Оплата учтена"
        except (FreelancerError, BudgetExceeded, LLMError) as e:
            self.store.reopen_pending(pid)
            await self.notify(f"⚠️ #{rec.id}: {esc(str(e))}")
            return "Ошибка, см. сообщение"
        return ""

    async def _send_followup(self, job_id: int) -> None:
        rec = self._job(job_id)
        pid = self.store.create_pending("followup", job_id, {})
        await self._card(pid, f"📨 #{job_id} {esc(rec.title)}: отклик отмечен.\n"
                              "Когда клиент напишет — ответьте на это сообщение его текстом, "
                              "я подготовлю ответ.",
                         [[("🏆 Меня выбрали", callback("won", pid)), ("👎 Отказ", callback("lost", pid))]])

    async def _send_paid_card(self, job_id: int) -> None:
        rec = self._job(job_id)
        pid = self.store.create_pending("delivered", job_id, {})
        await self._card(pid, f"📬 #{job_id} сдан. Когда клиент оплатит — нажмите кнопку "
                              f"(или /paid {job_id} сумма валюта). Сообщения клиента присылайте "
                              "ответом на это сообщение.",
                         [[(f"💰 Оплачено {rec.bid_amount:g} {rec.currency}", callback("paid", pid))]])

    async def on_reply(self, pid: int, text: str) -> None:
        p = self.store.get_pending(pid)
        if not p:
            return
        rec = self._job(p.job_id)
        if p.kind in ("proposal", "reply"):
            payload = {**p.payload, "text": text}
            self.store.update_pending(pid, payload=payload)
            if p.kind == "proposal":
                self.store.update_job(rec.id, proposal=text)
            old = p.tg_message_id
            if p.kind == "proposal":
                await self._send_proposal_card(pid, rec, text)
            else:
                api = rec.source == "freelancer" and self.fl is not None and self.fl.can_write
                await self._card(pid, f"✏️ #{rec.id} новый ответ клиенту:\n<pre>{esc(text)}</pre>",
                                 [[("✅ Отправить" if api else "✅ Я отправил",
                                    callback("reply_send", pid)),
                                   ("❌ Не отвечать", callback("reply_skip", pid))]])
            if old and self.tg:
                await self.tg.clear_buttons(old)
        elif p.kind in ("followup", "delivered"):
            await self.on_client_message(rec.id, text)
        elif p.kind == "delivery":
            self.store.resolve_pending(pid, "rework")
            if p.tg_message_id and self.tg:
                await self.tg.clear_buttons(p.tg_message_id)
            self.start_work(rec.id, [text])
            await self.notify(f"🔁 #{rec.id}: отправил на доработку")

    async def on_command(self, text: str) -> None:
        cmd, _, rest = text.partition(" ")
        cmd = cmd.split("@")[0].lower()
        args = rest.split()
        if cmd in ("/start", "/help"):
            await self.notify(HELP)
        elif cmd == "/stats":
            await self.notify(f"<pre>{esc(format_report(self.guard.report()))}</pre>")
        elif cmd == "/jobs":
            await self.notify(self.jobs_summary())
        elif cmd == "/pause":
            self.store.kv_set("paused", "1")
            await self.notify("⏸ Пауза: новые отклики не готовятся. Текущие заказы продолжаются.")
        elif cmd == "/resume":
            self.store.kv_set("paused", "0")
            await self.notify("▶️ Продолжаю искать заказы")
            self._wake.set()
        elif cmd == "/scan":
            self._wake.set()
            await self.notify("🔎 Запускаю поиск")
        elif cmd == "/work" and args and args[0].isdigit():
            self.start_work(int(args[0]))
        elif cmd == "/client" and len(args) >= 2 and args[0].isdigit():
            await self.on_client_message(int(args[0]), rest.split(" ", 1)[1])
        elif cmd == "/won" and args and args[0].isdigit():
            await self.accept_award(self._job(int(args[0])))
        elif cmd == "/paid" and len(args) >= 2 and args[0].isdigit():
            rec = self._job(int(args[0]))
            amount = float(args[1].replace(",", "."))
            currency = (args[2] if len(args) > 2 else rec.currency or "USD").upper()
            rate = (rec.usd_rate if currency == (rec.currency or "").upper() and rec.usd_rate
                    else self.s.economics.fx_to_usd.get(currency))
            if rate is None:
                await self.notify(f"Нет курса для {esc(currency)} в economics.fx_to_usd")
                return
            self.record_payment(rec, amount * rate, f"{amount:g} {currency} via /paid")
            self.store.set_status(rec.id, JobStatus.PAID)
            self.store.resolve_pending_for_job(rec.id, ("delivered",), "paid")
            await self.notify(f"💰 Учтено: {amount:g} {esc(currency)} (~${amount * rate:.2f})")
        else:
            await self.notify(HELP)

    def jobs_summary(self) -> str:
        counts = self.store.status_counts()
        lines = ["<b>Воронка</b>: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
        active = self.store.jobs_by_status(JobStatus.AWAITING_APPROVAL, JobStatus.SUBMITTED,
                                           JobStatus.WON, JobStatus.IN_PROGRESS, JobStatus.REVIEW,
                                           JobStatus.DELIVERED)
        for r in active[-25:]:
            lines.append(f"#{r.id} [{r.status}] {esc(r.title[:60])}")
        return "\n".join(lines)

    # ============================================================================= main loop

    async def cycle(self) -> CycleStats:
        stats = CycleStats()
        await self.scan(stats)
        await self.triage_new(stats)
        await self.sync_freelancer()
        await self.daily_digest()
        log.info("cycle: %s", stats)
        return stats

    async def daily_digest(self) -> None:
        today = self.store.clock().date().isoformat()
        if self.store.kv_get("digest") == today:
            return
        self.store.kv_set("digest", today)
        await self.notify(f"📊 Сводка\n<pre>{esc(format_report(self.guard.report()))}</pre>")

    async def poll_telegram(self) -> None:
        assert self.tg is not None
        offset = int(self.store.kv_get("tg_offset", "0") or 0) or None
        while True:
            try:
                updates = await self.tg.get_updates(offset)
            except (TelegramError, httpx.HTTPError) as e:
                log.warning("telegram poll: %s", e)
                await asyncio.sleep(5)
                continue
            for u in updates:
                offset = int(u["update_id"]) + 1
                self.store.kv_set("tg_offset", str(offset))
                try:
                    await self.handle_update(u)
                except Exception:
                    log.exception("telegram update failed")

    async def run_forever(self) -> None:
        if self.tg:
            poller = asyncio.create_task(self.poll_telegram())
            self._tasks.add(poller)
        for rec in self.store.jobs_by_status(JobStatus.IN_PROGRESS):
            self.start_work(rec.id)  # resume work interrupted by a restart
        while True:
            try:
                await self.cycle()
            except Exception:
                log.exception("cycle failed")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.s.poll_interval_minutes * 60)
            except TimeoutError:
                pass

    def _job(self, job_id: int) -> JobRecord:
        rec = self.store.get_job(job_id)
        if rec is None:
            raise KeyError(f"job #{job_id} not found")
        return rec


HELP = (
    "<b>Команды</b>\n"
    "/stats — экономика: выручка, расходы, прибыль\n"
    "/jobs — воронка заказов\n"
    "/scan — искать заказы сейчас\n"
    "/pause, /resume — остановить/возобновить поиск\n"
    "/client ID текст — сообщение клиента (для площадок без API)\n"
    "/won ID — вас выбрали, начать работу\n"
    "/work ID — запустить/повторить выполнение\n"
    "/paid ID сумма [валюта] — учесть оплату\n\n"
    "Ответ (reply) на карточку заменяет текст отклика/ответа или отправляет правки."
)

