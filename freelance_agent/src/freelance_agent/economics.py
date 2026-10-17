"""Bid / no-bid decision: a job is worth a proposal only if its expected profit is positive.

expected profit = P(win) * (price - platform fee - execution budget) - cost of the proposal
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .agents.schemas import Assessment
from .config import Economics, Profile
from .db import JobRecord, Store
from .models import Job


def to_usd(amount: float | None, currency: str | None, usd_rate: float | None,
           fx: dict[str, float]) -> float | None:
    if amount is None:
        return None
    if usd_rate:
        return amount * usd_rate
    rate = fx.get((currency or "USD").upper())
    return amount * rate if rate is not None else None


def usd_rate_for(job: Job, fx: dict[str, float]) -> float | None:
    if job.usd_rate:
        return job.usd_rate
    return fx.get((job.currency or "USD").upper())


def platform_fee(price_usd: float, source: str, econ: Economics) -> float:
    fee = econ.platform_fees.get(source)
    if fee is None:
        return 0.0
    return max(price_usd * fee.percent / 100.0, fee.min_usd)


def income_tax(amount_usd: float, econ: Economics) -> float:
    return amount_usd * econ.tax_percent / 100.0


def win_probability(bids: int, wins: int, bid_count: int | None, econ: Economics) -> float:
    """Beta-prior estimate of our win rate, discounted by how crowded the job already is."""
    base = (wins + econ.prior_win_rate * econ.prior_strength) / (bids + econ.prior_strength)
    if bid_count is None:
        crowd = 1.0
    elif bid_count < 10:
        crowd = 1.5
    elif bid_count < 30:
        crowd = 1.0
    elif bid_count < 60:
        crowd = 0.6
    else:
        crowd = 0.3
    return min(0.9, base * crowd)


@dataclass
class BidDecision:
    bid: bool
    reason: str
    price_usd: float = 0.0
    bid_amount: float = 0.0  # in the job's currency
    currency: str = "USD"
    fee_usd: float = 0.0
    execution_budget_usd: float = 0.0
    win_probability: float = 0.0
    profit_if_won_usd: float = 0.0
    expected_profit_usd: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(job: Job, a: Assessment, profile: Profile, econ: Economics,
           win_prob: float) -> BidDecision:
    if not a.ethical:
        return BidDecision(False, "неэтичный/запрещённый заказ")
    if not a.ai_deliverable:
        return BidDecision(False, "ИИ не может выполнить заказ целиком")
    if a.red_flags:
        return BidDecision(False, "признаки мошенничества: " + "; ".join(a.red_flags[:3]))

    rate = usd_rate_for(job, econ.fx_to_usd)
    if rate is None:
        return BidDecision(False, f"нет курса валюты {job.currency} (economics.fx_to_usd)")
    bmin = job.budget_min * rate if job.budget_min else None
    bmax = job.budget_max * rate if job.budget_max else None
    if bmax is None and a.stated_budget_amount:
        # Budget only in the text ("оплата 5000 рублей"): treat it as the ceiling.
        bmax = to_usd(a.stated_budget_amount, a.stated_budget_currency or job.currency,
                      None, econ.fx_to_usd)

    floor = profile.min_project_usd
    if bmax is not None and bmax < floor:
        return BidDecision(False, f"бюджет ${bmax:.0f} ниже вашего минимума ${floor:.0f}")
    price = max(a.suggested_price_usd, floor, bmin or 0.0)
    if bmax is not None:
        price = min(price, bmax)

    fee = platform_fee(price, job.source, econ) + income_tax(price, econ)
    net = price - fee
    needed = econ.execution_cost_usd[a.complexity]
    cap = min(econ.max_execution_usd, net * econ.max_execution_share)
    if needed > cap:
        return BidDecision(
            False,
            f"выполнение ИИ (~${needed:.2f}) дороже допустимой доли бюджета (${cap:.2f})",
            price_usd=price,
            fee_usd=fee,
        )
    budget = min(cap, needed * 1.5)
    profit = net - budget
    ev = win_prob * profit - econ.proposal_cost_estimate_usd
    d = BidDecision(
        True,
        "ставка выгодна",
        price_usd=round(price, 2),
        currency=(job.currency or "USD").upper(),
        bid_amount=_round_amount(price / rate),
        fee_usd=round(fee, 2),
        execution_budget_usd=round(budget, 2),
        win_probability=round(win_prob, 4),
        profit_if_won_usd=round(profit, 2),
        expected_profit_usd=round(ev, 2),
    )
    if profit < econ.min_profit_per_job_usd:
        return replace(d, bid=False, reason=f"прибыль при победе ${profit:.2f} меньше минимума "
                                            f"${econ.min_profit_per_job_usd:.2f}")
    if ev < econ.min_ev_to_bid_cost_ratio * econ.proposal_cost_estimate_usd:
        return replace(d, bid=False, reason=f"ожидаемая прибыль ${ev:.2f} не окупает отклик")
    return d


def _round_amount(amount: float) -> float:
    return float(round(amount)) if amount >= 20 else round(amount, 2)


def record_payment(store: Store, econ: Economics, rec: JobRecord, amount_usd: float,
                   note: str = "") -> None:
    """Book a received payment with the platform fee and income tax it implies."""
    store.add_ledger("revenue", amount_usd, rec.id, note)
    fee = platform_fee(amount_usd, rec.source, econ)
    if fee:
        store.add_ledger("fee", fee, rec.id, f"{rec.source} fee")
    tax = income_tax(amount_usd, econ)
    if tax:
        store.add_ledger("expense", tax, rec.id, f"tax {econ.tax_percent:g}%")
