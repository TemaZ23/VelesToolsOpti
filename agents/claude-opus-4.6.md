# Agent: claude-opus-4.6

- Name: `claude-opus-4.6`
- Provider: Anthropic (Claude family)
- Model: `claude-opus-4.6`
- Type: LLM / conversational
- Description: High-capacity Claude Opus 4.6 model configuration for use as an external agent.

## Configuration example (JSON)

```json
{
  "id": "claude-opus-4.6",
  "provider": "anthropic",
  "model": "claude-opus-4.6",
  "type": "llm",
  "description": "Claude Opus 4.6 agent",
  "env": {
    "ANTHROPIC_API_KEY": "required"
  },
  "settings": {
    "temperature": 0.2,
    "max_tokens": 4096
  }
}
```

## Integration notes

- Store `ANTHROPIC_API_KEY` in CI/secret storage or local env before running.
- Respect rate limits and usage quotas from Anthropic; add retries/backoff in network utilities.
- Follow repo conventions: expose integration through `src/services/*` and keep raw transport in `src/api`.

## Security & compliance

- Do not hard-code API keys or secrets in the repo.
- Ensure PII/PHI handling follows project policy; add redaction or safe-logging as needed.

## Next steps

If desired, scaffold a connector module at `src/services/claude_opus_4_6.ts` and a small usage example.
