"""SQLite persistence: jobs pipeline, LLM spend, money ledger, conversations, approvals."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Job, JobStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    budget_min REAL,
    budget_max REAL,
    currency TEXT,
    usd_rate REAL,
    bid_count INTEGER,
    posted_at TEXT,
    skills TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    status_reason TEXT,
    score INTEGER,
    assessment TEXT,
    decision TEXT,
    proposal TEXT,
    bid_amount REAL,
    bid_period_days INTEGER,
    external_bid_id TEXT,
    thread_id TEXT,
    requirements TEXT,
    workspace TEXT,
    submitted_at TEXT,
    won_at TEXT,
    paid_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    job_id INTEGER,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_ts ON llm_calls(ts);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    job_id INTEGER,
    kind TEXT NOT NULL CHECK (kind IN ('revenue', 'fee', 'expense')),
    amount_usd REAL NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    external_id TEXT,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    UNIQUE (job_id, external_id)
);

CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    tg_message_id INTEGER,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(UTC).isoformat() if dt else None


@dataclass
class JobRecord:
    id: int
    uid: str
    source: str
    external_id: str
    url: str
    title: str
    description: str
    budget_min: float | None
    budget_max: float | None
    currency: str | None
    usd_rate: float | None
    bid_count: int | None
    posted_at: str | None
    skills: list[str]
    raw: dict[str, Any]
    status: JobStatus
    status_reason: str | None
    score: int | None
    assessment: dict[str, Any] | None
    decision: dict[str, Any] | None
    proposal: str | None
    bid_amount: float | None
    bid_period_days: int | None
    external_bid_id: str | None
    thread_id: str | None
    requirements: str | None
    workspace: str | None
    submitted_at: str | None
    won_at: str | None
    paid_at: str | None
    created_at: str
    updated_at: str

    def to_job(self) -> Job:
        return Job(
            source=self.source,
            external_id=self.external_id,
            url=self.url,
            title=self.title,
            description=self.description,
            budget_min=self.budget_min,
            budget_max=self.budget_max,
            currency=self.currency,
            usd_rate=self.usd_rate,
            bid_count=self.bid_count,
            posted_at=datetime.fromisoformat(self.posted_at) if self.posted_at else None,
            skills=list(self.skills),
            raw=dict(self.raw),
        )


_JSON_COLUMNS = {"skills", "raw", "assessment", "decision"}
_JOB_COLUMNS = {f.name for f in fields(JobRecord)}


@dataclass
class Pending:
    id: int
    kind: str
    job_id: int
    payload: dict[str, Any]
    tg_message_id: int | None
    resolved_at: str | None


@dataclass
class Message:
    id: int
    direction: str
    text: str
    ts: str


class Store:
    def __init__(self, path: Path | str, clock: Callable[[], datetime] = utcnow) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.clock = clock

    def now(self) -> str:
        return _iso(self.clock()) or ""

    def close(self) -> None:
        self.conn.close()

    # ---- jobs -------------------------------------------------------------------------

    def upsert_job(self, job: Job) -> tuple[int, bool]:
        """Insert a newly seen job. Returns (id, is_new); existing jobs only refresh bid_count."""
        row = self.conn.execute("SELECT id FROM jobs WHERE uid = ?", (job.uid,)).fetchone()
        if row:
            if job.bid_count is not None:
                self.conn.execute(
                    "UPDATE jobs SET bid_count = ? WHERE id = ?", (job.bid_count, row["id"])
                )
                self.conn.commit()
            return int(row["id"]), False
        now = self.now()
        cur = self.conn.execute(
            """INSERT INTO jobs (uid, source, external_id, url, title, description, budget_min,
                   budget_max, currency, usd_rate, bid_count, posted_at, skills, raw, status,
                   created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.uid,
                job.source,
                job.external_id,
                job.url,
                job.title,
                job.description,
                job.budget_min,
                job.budget_max,
                job.currency,
                job.usd_rate,
                job.bid_count,
                _iso(job.posted_at),
                json.dumps(job.skills, ensure_ascii=False),
                json.dumps(job.raw, ensure_ascii=False, default=str),
                JobStatus.NEW,
                now,
                now,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0), True

    def _record(self, row: sqlite3.Row) -> JobRecord:
        data = dict(row)
        for col in _JSON_COLUMNS:
            if data[col] is not None:
                data[col] = json.loads(data[col])
        data["status"] = JobStatus(data["status"])
        return JobRecord(**data)

    def get_job(self, job_id: int) -> JobRecord | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._record(row) if row else None

    def get_job_by_uid(self, uid: str) -> JobRecord | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE uid = ?", (uid,)).fetchone()
        return self._record(row) if row else None

    def jobs_by_status(self, *statuses: JobStatus, limit: int | None = None) -> list[JobRecord]:
        marks = ",".join("?" for _ in statuses)
        sql = f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self._record(r) for r in self.conn.execute(sql, tuple(statuses))]

    def update_job(self, job_id: int, **values: Any) -> None:
        unknown = set(values) - _JOB_COLUMNS
        if unknown:
            raise KeyError(f"unknown job columns: {sorted(unknown)}")
        for col in _JSON_COLUMNS & set(values):
            values[col] = json.dumps(values[col], ensure_ascii=False, default=str)
        values["updated_at"] = self.now()
        assignments = ", ".join(f"{col} = ?" for col in values)
        self.conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?", (*values.values(), job_id)
        )
        self.conn.commit()

    def set_status(self, job_id: int, status: JobStatus, reason: str | None = None) -> None:
        extra: dict[str, Any] = {}
        if status == JobStatus.SUBMITTED:
            extra["submitted_at"] = self.now()
        elif status == JobStatus.WON:
            extra["won_at"] = self.now()
        elif status == JobStatus.PAID:
            extra["paid_at"] = self.now()
        self.update_job(job_id, status=status, status_reason=reason, **extra)

    def count_submitted_since(self, source: str, since: datetime) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE source = ? AND submitted_at >= ?",
            (source, _iso(since)),
        ).fetchone()
        return int(row[0])

    def win_stats(self, source: str | None = None) -> tuple[int, int]:
        """(bids placed, bids won) over the whole history, optionally per source."""
        where, args = ("WHERE source = ?", (source,)) if source else ("", ())
        row = self.conn.execute(
            f"""SELECT COUNT(submitted_at), COUNT(CASE WHEN submitted_at IS NOT NULL
                       THEN won_at END) FROM jobs {where}""",
            args,
        ).fetchone()
        return int(row[0]), int(row[1])

    def status_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        return {r[0]: int(r[1]) for r in rows}

    # ---- money --------------------------------------------------------------------------

    def add_llm_call(
        self,
        *,
        purpose: str,
        model: str,
        cost_usd: float,
        job_id: int | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        self.conn.execute(
            """INSERT INTO llm_calls (ts, job_id, purpose, model, input_tokens, output_tokens,
                   cache_write_tokens, cache_read_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.now(),
                job_id,
                purpose,
                model,
                input_tokens,
                output_tokens,
                cache_write_tokens,
                cache_read_tokens,
                cost_usd,
            ),
        )
        self.conn.commit()

    def llm_spend(self, since: datetime | None = None, job_id: int | None = None,
                  purposes: Iterable[str] | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE 1=1"
        args: list[Any] = []
        if since is not None:
            sql += " AND ts >= ?"
            args.append(_iso(since))
        if job_id is not None:
            sql += " AND job_id = ?"
            args.append(job_id)
        if purposes is not None:
            purposes = list(purposes)
            sql += f" AND purpose IN ({','.join('?' for _ in purposes)})"
            args.extend(purposes)
        return float(self.conn.execute(sql, args).fetchone()[0])

    def llm_spend_by_purpose(self, since: datetime | None = None) -> dict[str, float]:
        sql, args = "SELECT purpose, SUM(cost_usd) FROM llm_calls", []
        if since is not None:
            sql += " WHERE ts >= ?"
            args.append(_iso(since))
        sql += " GROUP BY purpose"
        return {r[0]: float(r[1]) for r in self.conn.execute(sql, args)}

    def add_ledger(self, kind: str, amount_usd: float, job_id: int | None = None, note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO ledger (ts, job_id, kind, amount_usd, note) VALUES (?, ?, ?, ?, ?)",
            (self.now(), job_id, kind, amount_usd, note),
        )
        self.conn.commit()

    def ledger_sum(self, kind: str, since: datetime | None = None) -> float:
        sql = "SELECT COALESCE(SUM(amount_usd), 0) FROM ledger WHERE kind = ?"
        args: list[Any] = [kind]
        if since is not None:
            sql += " AND ts >= ?"
            args.append(_iso(since))
        return float(self.conn.execute(sql, args).fetchone()[0])

    # ---- conversations ------------------------------------------------------------------

    def add_message(
        self, job_id: int, direction: str, text: str, external_id: str | None = None
    ) -> int | None:
        """Store a message; returns None when this external message was already stored."""
        try:
            cur = self.conn.execute(
                "INSERT INTO messages (job_id, direction, external_id, text, ts) VALUES (?, ?, ?, ?, ?)",
                (job_id, direction, external_id, text, self.now()),
            )
        except sqlite3.IntegrityError:
            return None
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def claim_outgoing(self, job_id: int, text: str, external_id: str) -> bool:
        """Link a platform message id to the copy we stored when we sent it ourselves."""
        cur = self.conn.execute(
            """UPDATE messages SET external_id = ? WHERE id = (
                   SELECT id FROM messages WHERE job_id = ? AND direction = 'out'
                   AND external_id IS NULL AND text = ? ORDER BY id LIMIT 1)""",
            (external_id, job_id, text),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def has_message(self, job_id: int, external_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM messages WHERE job_id = ? AND external_id = ?", (job_id, external_id)
        ).fetchone()
        return row is not None

    def messages_for(self, job_id: int) -> list[Message]:
        rows = self.conn.execute(
            "SELECT id, direction, text, ts FROM messages WHERE job_id = ? ORDER BY id", (job_id,)
        )
        return [Message(**dict(r)) for r in rows]

    # ---- approvals ----------------------------------------------------------------------

    def create_pending(self, kind: str, job_id: int, payload: dict[str, Any]) -> int:
        cur = self.conn.execute(
            "INSERT INTO pending (kind, job_id, payload, created_at) VALUES (?, ?, ?, ?)",
            (kind, job_id, json.dumps(payload, ensure_ascii=False), self.now()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def _pending(self, row: sqlite3.Row | None) -> Pending | None:
        if row is None:
            return None
        return Pending(
            id=row["id"],
            kind=row["kind"],
            job_id=row["job_id"],
            payload=json.loads(row["payload"]),
            tg_message_id=row["tg_message_id"],
            resolved_at=row["resolved_at"],
        )

    def get_pending(self, pending_id: int) -> Pending | None:
        row = self.conn.execute("SELECT * FROM pending WHERE id = ?", (pending_id,)).fetchone()
        return self._pending(row)

    def pending_by_tg_message(self, tg_message_id: int) -> Pending | None:
        row = self.conn.execute(
            "SELECT * FROM pending WHERE tg_message_id = ? AND resolved_at IS NULL", (tg_message_id,)
        ).fetchone()
        return self._pending(row)

    def open_pending(self, kind: str | None = None) -> list[Pending]:
        sql = "SELECT * FROM pending WHERE resolved_at IS NULL"
        args: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        rows = self.conn.execute(sql + " ORDER BY id", args)
        return [p for p in (self._pending(r) for r in rows) if p]

    def update_pending(self, pending_id: int, payload: dict[str, Any] | None = None,
                       tg_message_id: int | None = None) -> None:
        if payload is not None:
            self.conn.execute(
                "UPDATE pending SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), pending_id),
            )
        if tg_message_id is not None:
            self.conn.execute(
                "UPDATE pending SET tg_message_id = ? WHERE id = ?", (tg_message_id, pending_id)
            )
        self.conn.commit()

    def resolve_pending(self, pending_id: int, resolution: str) -> bool:
        """Mark resolved; returns False if somebody already resolved it (double tap)."""
        cur = self.conn.execute(
            "UPDATE pending SET resolved_at = ?, resolution = ? WHERE id = ? AND resolved_at IS NULL",
            (self.now(), resolution, pending_id),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def reopen_pending(self, pending_id: int) -> None:
        """Undo a resolution when the action behind the tap failed, so you can tap again."""
        self.conn.execute(
            "UPDATE pending SET resolved_at = NULL, resolution = NULL WHERE id = ?", (pending_id,)
        )
        self.conn.commit()

    def resolve_pending_for_job(self, job_id: int, kinds: Iterable[str], resolution: str) -> None:
        kinds = list(kinds)
        marks = ",".join("?" for _ in kinds)
        self.conn.execute(
            f"""UPDATE pending SET resolved_at = ?, resolution = ?
                WHERE job_id = ? AND kind IN ({marks}) AND resolved_at IS NULL""",
            (self.now(), resolution, job_id, *kinds),
        )
        self.conn.commit()

    # ---- key/value ----------------------------------------------------------------------

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()
