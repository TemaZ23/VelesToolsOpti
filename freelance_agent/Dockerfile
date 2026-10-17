FROM python:3.12-slim

# bubblewrap + socat: OS sandbox for the code the executor runs (Claude Agent SDK).
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap socat git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Runs as root so the bind-mounted ./agent-data is writable on any host; the container and the
# nested sandbox are the isolation layers for executed code.
WORKDIR /work
ENV FREELANCE_AGENT_DOCKER=1 PYTHONUNBUFFERED=1

ENTRYPOINT ["freelance-agent", "-c", "/work/config.yaml"]
CMD ["run"]
