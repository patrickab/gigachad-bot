import os

PROJECT_ROOT = os.path.abspath(".")
DOCKERFILES_PATH = os.path.join(PROJECT_ROOT, "src", "dockersandbox", "dockerfiles")
DOCKERFILES_PYTHON_VERSION = "3.11-slim"

DOCKERTAG_BASE = "sandbox-base"
DOCKERTAG_AIDER = "sandbox-aider"

DOCKERFILE_BASE = """
FROM python:3.11-slim

# System Layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Tooling Layer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Workspace
WORKDIR /workspace
ENV UV_SYSTEM_PYTHON=1

# Install Agent
RUN uv pip install aider-chat

# Entrypoint: Standard shell, no background port forwarding scripts
CMD ["/bin/bash"]
"""

DOCKERFILE_DEFINITIONS = {
    f"{DOCKERTAG_AIDER}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN uv pip install aider-chat
""",
}