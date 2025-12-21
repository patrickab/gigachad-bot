import os

PROJECT_ROOT = os.path.abspath(".")
DOCKERFILES_PATH = os.path.join(PROJECT_ROOT, "src", "dockersandbox", "dockerfiles")
DOCKERFILES_PYTHON_VERSION = "3.11-slim"

DOCKERTAG_BASE = "sandbox-base"
DOCKERTAG_AIDER = "sandbox-aider"

DOCKERFILE_BASE = f"""
FROM python:{DOCKERFILES_PYTHON_VERSION}

# System Layer - tcpdump for network traffic logging
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Tooling Layer (uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Workspace Setup
WORKDIR /workspace

# Environment
ENV UV_SYSTEM_PYTHON=1
"""

DOCKERFILE_DEFINITIONS = {
    f"{DOCKERTAG_AIDER}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN uv pip install aider-chat
""",
}