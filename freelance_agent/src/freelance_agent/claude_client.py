"""The Claude API client, with the workspace header that organization-level keys require."""

from __future__ import annotations

import anthropic

WORKSPACE_HEADER = "anthropic-workspace-id"
NOT_SCOPED_HINT = "not scoped to a workspace"


def make_client(api_key: str | None, workspace_id: str | None = None,
                max_retries: int = 2) -> anthropic.AsyncAnthropic:
    headers = {WORKSPACE_HEADER: workspace_id} if workspace_id else None
    return anthropic.AsyncAnthropic(api_key=api_key, default_headers=headers, max_retries=max_retries)


def needs_workspace(error_message: str) -> bool:
    return NOT_SCOPED_HINT in error_message


WORKSPACE_HELP = (
    "Ключ не привязан к workspace. Либо впишите в .env ANTHROPIC_WORKSPACE_ID=wrkspc_... "
    "(ID есть в консоли Anthropic: Settings -> Workspaces), либо создайте ключ внутри workspace."
)
