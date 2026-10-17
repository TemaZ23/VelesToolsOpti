"""Does the actual work with the Claude Agent SDK (Claude Code as a library) in a per-job folder,
under a hard dollar cap (`max_budget_usd`) and, by default, the OS sandbox."""

from __future__ import annotations

import functools
import logging
import os
import platform
import shutil
import subprocess
import zipfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query

from ..budget import BudgetGuard
from ..config import Settings
from ..db import Message, Store
from ..models import Job
from .prompts import UNTRUSTED_NOTE, conversation_block, job_block

log = logging.getLogger(__name__)

QueryFn = Callable[..., AsyncIterator[Any]]

DELIVERABLES = "deliverables"

EXECUTOR_APPEND = f"""
You are completing a paid freelance order. The full order is in TASK.md in the current folder.
- Put everything the client receives into ./{DELIVERABLES}/ (source code, documents, data).
- Code must run: include a README.md with setup and usage, pin dependencies, add and run tests
  or a demo command that proves it works, and fix failures before finishing.
- Texts must be original, in the requested language and volume; no placeholders or TODOs.
- Never fabricate data, results or test runs. If something cannot be done, say so in
  ./{DELIVERABLES}/DELIVERY_NOTES.md and explain what is missing.
- Write ./{DELIVERABLES}/DELIVERY_NOTES.md: what was done, how to use it, how it was verified.
- Work only inside the current folder. Do not read environment variables, credentials or files
  outside it, and do not contact anyone.
{UNTRUSTED_NOTE}
"""

TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".csv", ".html", ".css",
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".sql", ".sh", ".xml", ".go", ".rs", ".java",
    ".php", ".rb", ".kt", ".swift", ".c", ".h", ".cpp", ".cs", ".vue", ".svelte", ".env.example",
}


@dataclass
class ExecutionResult:
    ok: bool
    cost_usd: float
    session_id: str | None = None
    report: str = ""
    error: str | None = None
    files: list[Path] = field(default_factory=list)


@functools.cache
def sandbox_problem() -> str | None:
    """None if Claude Code's OS sandbox can start here, otherwise why not.

    On Linux this really starts bubblewrap with a private network namespace (what the sandbox
    needs), because inside containers bwrap is often installed but not permitted to run."""
    system = platform.system()
    if system == "Darwin":
        return None
    if system != "Linux":
        return f"no OS sandbox on {system}: run the agent in WSL2, on a Linux VPS or in Docker"
    missing = [tool for tool in ("bwrap", "socat") if shutil.which(tool) is None]
    if missing:
        return f"missing {', '.join(missing)}: apt-get install bubblewrap socat"
    try:
        probe = subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-net", "true"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"bubblewrap failed to start: {e}"
    if probe.returncode != 0:
        hint = ("run the container with security_opt seccomp=unconfined and apparmor=unconfined, "
                "or run the agent directly on a Linux VPS / WSL2"
                if os.environ.get("FREELANCE_AGENT_DOCKER") == "1"
                else "check that unprivileged user namespaces are enabled")
        return f"bubblewrap cannot create namespaces here ({probe.stderr.strip()}); {hint}"
    return None



class Executor:
    def __init__(self, settings: Settings, store: Store, guard: BudgetGuard,
                 query_fn: QueryFn = query) -> None:
        self.s = settings
        self.store = store
        self.guard = guard
        self.query_fn = query_fn

    def workspace(self, job_id: int) -> Path:
        ws = self.s.workspaces_dir / f"job-{job_id}"
        (ws / DELIVERABLES).mkdir(parents=True, exist_ok=True)
        return ws

    def write_task(self, ws: Path, job: Job, requirements: str, history: list[Message]) -> None:
        (ws / "TASK.md").write_text(
            "# Order\n\n"
            f"{job_block(job)}\n\n"
            "# Agreed requirements and acceptance criteria\n\n"
            f"{requirements or '(only the job post above)'}\n\n"
            "# Conversation with the client\n\n"
            f"{conversation_block(history)}\n",
            encoding="utf-8",
        )

    def options(self, ws: Path, budget_usd: float, resume: str | None) -> ClaudeAgentOptions:
        ex = self.s.executor
        tools = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
        if ex.allow_web:
            tools += ["WebSearch", "WebFetch"]
        sandbox: dict[str, Any] | None = None
        if ex.sandbox:
            sandbox = {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "network": {"allowedDomains": list(ex.allowed_domains)},
                "enableWeakerNestedSandbox": ex.weaker_nested_sandbox,
            }
        return ClaudeAgentOptions(
            cwd=str(ws),
            model=self.s.models.executor,
            effort=self.s.models.executor_effort,
            max_budget_usd=budget_usd,
            max_turns=ex.max_turns,
            tools=tools,
            allowed_tools=tools,
            permission_mode="dontAsk",
            setting_sources=[],
            system_prompt={"type": "preset", "preset": "claude_code", "append": EXECUTOR_APPEND},
            sandbox=sandbox,  # type: ignore[arg-type]
            # Platform credentials stay out of the agent's process environment.
            env={"TELEGRAM_BOT_TOKEN": "", "FREELANCER_OAUTH_TOKEN": ""},
            resume=resume,
        )

    async def run(self, job_id: int, job: Job, requirements: str, history: list[Message],
                  budget_usd: float, fix_notes: list[str] | None = None,
                  resume: str | None = None) -> ExecutionResult:
        ex = self.s.executor
        if ex.sandbox and (problem := sandbox_problem()):
            return ExecutionResult(False, 0.0, error=(
                f"OS sandbox unavailable: {problem}. "
                "Set executor.allow_unsandboxed only on a disposable machine."))
        if not ex.sandbox and not ex.allow_unsandboxed:
            return ExecutionResult(False, 0.0, error=(
                "executor.sandbox is off but executor.allow_unsandboxed is false"))
        self.guard.check(budget_usd, "execute")

        ws = self.workspace(job_id)
        self.write_task(ws, job, requirements, history)
        if fix_notes:
            prompt = (
                "The reviewer rejected the deliverable. Fix every issue, re-run the checks, "
                "update DELIVERY_NOTES.md:\n" + "\n".join(f"- {n}" for n in fix_notes)
            )
        else:
            prompt = "Read TASK.md and complete the order. Finish only when the deliverable is ready."

        result: ResultMessage | None = None
        try:
            async for msg in self.query_fn(prompt=prompt, options=self.options(ws, budget_usd, resume)):
                if isinstance(msg, ResultMessage):
                    result = msg
        except ClaudeSDKError as e:
            log.exception("executor crashed for job %s", job_id)
            return ExecutionResult(False, 0.0, error=f"agent error: {e}")

        if result is None:
            return ExecutionResult(False, 0.0, error="agent ended without a result")
        cost = float(result.total_cost_usd or 0.0)
        usage = result.usage or {}
        self.store.add_llm_call(
            purpose="execute",
            model=self.s.models.executor,
            cost_usd=cost,
            job_id=job_id,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0)),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        )
        files = deliverable_files(ws)
        if result.is_error:
            return ExecutionResult(False, cost, result.session_id, result.result or "",
                                   error=f"agent stopped: {result.subtype}", files=files)
        if not files:
            return ExecutionResult(False, cost, result.session_id, result.result or "",
                                   error="no files in deliverables/")
        return ExecutionResult(True, cost, result.session_id, result.result or "", files=files)


def deliverable_files(ws: Path) -> list[Path]:
    root = ws / DELIVERABLES
    skip = {"node_modules", ".venv", "venv", "__pycache__", ".git", ".pytest_cache"}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part in skip for part in p.relative_to(root).parts)
    )


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", "Makefile", "README"}


def files_for_review(ws: Path, max_chars: int = 150_000) -> tuple[str, str]:
    """Listing of all files plus contents of text files. Files that do not fit are named, so
    the reviewer knows exactly what it did not see."""
    root = ws / DELIVERABLES
    files = deliverable_files(ws)
    listing = "\n".join(f"{p.relative_to(root)}  ({p.stat().st_size} bytes)" for p in files)
    parts: list[str] = []
    used = 0
    omitted: list[str] = []
    # Notes and READMEs first: they describe everything else.
    ordered = sorted(files, key=lambda p: (p.name not in {"DELIVERY_NOTES.md", "README.md"}, str(p)))
    for p in ordered:
        rel = str(p.relative_to(root))
        if not is_text(p):
            omitted.append(f"{rel} (binary)")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if used + len(text) > max_chars:
            omitted.append(f"{rel} (not shown: review size limit)")
            continue
        used += len(text)
        parts.append(f'<file path="{rel}">\n{text}\n</file>')
    if omitted:
        parts.append("Not shown: " + "; ".join(omitted))
    return listing, "\n".join(parts)


def pack(ws: Path, name: str) -> Path:
    root = ws / DELIVERABLES
    archive = ws / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in deliverable_files(ws):
            zf.write(p, p.relative_to(root))
    return archive
