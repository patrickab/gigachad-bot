import os

PROJECT_ROOT = os.path.abspath(".")
DOCKERFILES_PATH = os.path.join(PROJECT_ROOT, "src", "dockersandbox", "dockerfiles")
DOCKERFILES_PYTHON_VERSION = "3.11-slim"

DOCKERTAG_BASE = "sandbox-base"
DOCKERTAG_AIDER = "sandbox-aider"
DOCKERTAG_GEMINI = "sandbox-gemini"
DOCKERTAG_QWEN = "sandbox-qwen"
DOCKERTAG_OPENCODE = "sandbox-opencode"
DOCKERTAG_CODEX = "sandbox-codex"
DOCKERTAG_CLAUDE = "sandbox-claude"
DOCKERTAG_CURSOR = "sandbox-cursor"

DOCKERFILE_BASE = """
FROM python:3.11-slim

# System Layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential tcpdump \
    gcc g++ make libc6-dev procps file \
    && rm -rf /var/lib/apt/lists/*

# Tooling Layer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV NONINTERACTIVE=1
RUN git clone https://github.com/Homebrew/brew /home/linuxbrew/.linuxbrew/Homebrew
RUN mkdir -p /home/linuxbrew/.linuxbrew/bin \
    && ln -s /home/linuxbrew/.linuxbrew/Homebrew/bin/brew /home/linuxbrew/.linuxbrew/bin/brew

ENV HOMEBREW_PREFIX=/home/linuxbrew/.linuxbrew
ENV HOMEBREW_CELLAR=/home/linuxbrew/.linuxbrew/Cellar
ENV HOMEBREW_REPOSITORY=/home/linuxbrew/.linuxbrew/Homebrew
ENV PATH=/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:$PATH

RUN brew update

# Workspace
WORKDIR /workspace
ENV UV_SYSTEM_PYTHON=1

# Entrypoint
CMD ["/bin/bash"]
"""

DOCKERFILE_DEFINITIONS = {
    # Aider
    f"{DOCKERTAG_AIDER}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN uv pip install aider-chat
""",
    # Gemini
    f"{DOCKERTAG_GEMINI}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN brew install gemini-cli
""",
    # Qwen
    f"{DOCKERTAG_QWEN}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN brew install qwen-code
""",
    # OpenCode
    f"{DOCKERTAG_OPENCODE}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN brew install opencode
""",
    # Codex
    f"{DOCKERTAG_CODEX}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN brew install --cask codex
""",
    # Claude Code
    f"{DOCKERTAG_CLAUDE}": f"""
FROM {DOCKERTAG_BASE}:latest
RUN brew install --cask claude-code
""",
    # Cursor CLI
f"{DOCKERTAG_CURSOR}": f"""
FROM {DOCKERTAG_BASE}:latest
# Use && to separate commands and \\ to continue the line in Docker
RUN curl https://cursor.com/install -fsS | bash && \\
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Update PATH for the image persistently
ENV PATH="/root/.local/bin:${{PATH}}"
""",
}
