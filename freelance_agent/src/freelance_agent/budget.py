"""Spend limits and the profit report. The guard runs before every paid LLM call."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import Budget
from .db import Store

# Spend on finding work. The daily limit caps only this; doing paid work (execute, review,
# client replies) is limited by the monthly cap and by each job's own execution budget.
ACQUISITION = ("triage", "proposal")


class BudgetExceeded(RuntimeError):
    pass


def _day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(now: datetime) -> datetime:
    return _day_start(now).replace(day=1)


@dataclass
class Report:
    llm_today: float
    llm_month: float
    llm_total: float
    revenue_month: float
    fees_month: float
    expenses_month: float
    fixed_month: float
    net_revenue_total: float
    bids: int
    wins: int
    stop_loss_tripped: bool
    paused: bool
    llm_by_purpose: dict[str, float] = field(default_factory=dict)

    @property
    def net_revenue_month(self) -> float:
        return self.revenue_month - self.fees_month - self.expenses_month

    @property
    def profit_month(self) -> float:
        """What is left this month after LLM spend and all fixed costs (subscription etc.)."""
        return self.net_revenue_month - self.llm_month - self.fixed_month

    @property
    def to_break_even(self) -> float:
        return max(0.0, -self.profit_month)

    @property
    def lifetime_variable_profit(self) -> float:
        return self.net_revenue_total - self.llm_total

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.bids if self.bids else None


class BudgetGuard:
    def __init__(self, store: Store, budget: Budget) -> None:
        self.store = store
        self.budget = budget

    def check(self, estimated_usd: float, purpose: str) -> None:
        now = self.store.clock()
        if purpose in ACQUISITION:
            today = self.store.llm_spend(since=_day_start(now), purposes=ACQUISITION)
            if today + estimated_usd > self.budget.daily_llm_limit_usd:
                raise BudgetExceeded(
                    f"{purpose}: дневной лимит на поиск заказов ${self.budget.daily_llm_limit_usd:.2f} "
                    f"(потрачено ${today:.2f}, вызов до ${estimated_usd:.2f})"
                )
        month = self.store.llm_spend(since=_month_start(now))
        if month + estimated_usd > self.budget.monthly_llm_limit_usd:
            raise BudgetExceeded(
                f"{purpose}: месячный лимит LLM ${self.budget.monthly_llm_limit_usd:.2f} "
                f"(потрачено ${month:.2f}, вызов до ${estimated_usd:.2f})"
            )

    def net_revenue_total(self) -> float:
        s = self.store
        return s.ledger_sum("revenue") - s.ledger_sum("fee") - s.ledger_sum("expense")

    def stop_loss_tripped(self) -> bool:
        return self.store.llm_spend() - self.net_revenue_total() > self.budget.stop_loss_usd

    def bidding_allowed(self) -> tuple[bool, str]:
        if self.store.kv_get("paused") == "1":
            return False, "пауза включена вами (/resume чтобы продолжить)"
        if self.stop_loss_tripped():
            return False, (
                f"стоп-лосс: расходы на LLM превысили заработок более чем на "
                f"${self.budget.stop_loss_usd:.2f}; ставки остановлены до первой оплаты "
                f"(или увеличьте budget.stop_loss_usd)"
            )
        return True, ""

    def report(self) -> Report:
        s = self.store
        now = s.clock()
        month = _month_start(now)
        bids, wins = s.win_stats()
        return Report(
            llm_today=s.llm_spend(since=_day_start(now)),
            llm_month=s.llm_spend(since=month),
            llm_total=s.llm_spend(),
            revenue_month=s.ledger_sum("revenue", since=month),
            fees_month=s.ledger_sum("fee", since=month),
            expenses_month=s.ledger_sum("expense", since=month),
            fixed_month=sum(self.budget.fixed_monthly_costs_usd.values()),
            net_revenue_total=self.net_revenue_total(),
            bids=bids,
            wins=wins,
            stop_loss_tripped=self.stop_loss_tripped(),
            paused=s.kv_get("paused") == "1",
            llm_by_purpose=s.llm_spend_by_purpose(since=month),
        )


def format_report(r: Report) -> str:
    win = f"{r.win_rate:.0%}" if r.win_rate is not None else "нет данных"
    lines = [
        "Экономика (USD, текущий месяц)",
        f"  выручка             {r.revenue_month:9.2f}",
        f"  комиссии площадок  -{r.fees_month:9.2f}",
        f"  налоги и прочее    -{r.expenses_month:9.2f}",
        f"  LLM (API)          -{r.llm_month:9.2f}   (сегодня {r.llm_today:.2f})",
        f"  фикс. расходы      -{r.fixed_month:9.2f}   (подписки и т.п.)",
        f"  ПРИБЫЛЬ             {r.profit_month:9.2f}",
    ]
    if r.to_break_even > 0:
        lines.append(f"  до окупаемости      {r.to_break_even:9.2f}")
    lines += [
        "",
        f"За всё время: чистая выручка {r.net_revenue_total:.2f}, LLM {r.llm_total:.2f}, "
        f"маржа {r.lifetime_variable_profit:.2f}",
        f"Откликов {r.bids}, выиграно {r.wins}, конверсия {win}",
    ]
    if r.llm_by_purpose:
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r.llm_by_purpose.items()))
        lines.append(f"LLM по задачам за месяц: {parts}")
    if r.paused:
        lines.append("СТАТУС: пауза")
    if r.stop_loss_tripped:
        lines.append("СТАТУС: сработал стоп-лосс, новые отклики не отправляются")
    return "\n".join(lines)
