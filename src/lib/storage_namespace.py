"""Canonical database namespaces for application-owned artifacts.

These keys identify database records. They are never derived from the local
Documents directory layout. ``PDFS`` and ``MINERU`` retain their physical names
because those are the only artifact classes permitted to mirror to Documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from lib.data_store import InvalidStorageKey, validate_key

CHAT = "chat"
PROJECT = "project"
NOTE = "note"
ATTACHMENT = "attachment"
DRAWING = "drawing"
GRAPH = "graph"
PROMPT = "prompt"
MEMORY = "memory"
MODEL = "model"
PDFS = "PDFs"
MINERU = "Mineru"
SANDBOX = "sandbox"
PDF_ANNOTATION = "pdf-annotation"
PROJECT_JSON = "project.json"
# The project catalog (names, slugs, archived flags), at the chat namespace root.
META_JSON = "projects-meta.json"
DEFAULT_SANDBOX_SCOPE = "workspace"


def _part(value: str, label: str) -> str:
    """Validate *value* as a single key segment (no separators)."""
    try:
        parsed = validate_key(value)
    except InvalidStorageKey as exc:
        raise ValueError(f"Invalid {label}") from exc
    if "/" in parsed:
        raise ValueError(f"Invalid {label}")
    return parsed


def project_document(slug: str, name: str) -> str:
    return f"{PROJECT}/{_part(slug, 'project slug')}/document/{_part(name, 'document name')}"


def project_document_prefix(slug: str) -> str:
    return f"{PROJECT}/{_part(slug, 'project slug')}/document"


def project_json(slug: str) -> str:
    """Per-project state (name, kanban, tabs, files, linked root) in the chat namespace."""
    return f"{CHAT}/{_part(slug, 'project slug')}/{PROJECT_JSON}"


def chat_root() -> str:
    """Root of every chat file, project directories included."""
    return CHAT


def project_dir(slug: str) -> str:
    """A project's directory of chats in the chat namespace. Rejects empty slugs and anything with a path separator."""
    if PurePosixPath(slug).name != slug or not slug:
        raise ValueError("Invalid project slug")
    return f"{CHAT}/{slug}"


def _memory_project_dir(slug: str) -> str:
    # Reject separators and dot segments so a slug cannot escape the project directory.
    if not slug:
        raise ValueError("Project slug is required")
    if "/" in slug or "\\" in slug or slug in {".", ".."}:
        raise ValueError("Invalid project slug")
    return f"{MEMORY}/project/{slug}/memory"


def memory_topics_key(project_slug: str | None) -> str:
    """Topic list of the global memory (None) or of one project's memory."""
    if project_slug is None:
        return f"{MEMORY}/global-topics.json"
    return f"{_memory_project_dir(project_slug)}/topics.json"


def memory_memories_key(project_slug: str | None) -> str:
    """Stored facts of the global memory (None) or of one project's memory."""
    if project_slug is None:
        return f"{MEMORY}/global-profile.json"
    return f"{_memory_project_dir(project_slug)}/memory.json"


def library_pdf(name: str) -> str:
    """A PDF in the user's shared library."""
    return f"{PDFS}/{name}"


def pdf_stem(name: str) -> str:
    """*name* without a case-insensitive `.pdf` suffix, the stem its MinerU output is keyed by."""
    return name[:-4] if name.lower().endswith(".pdf") else name


def library_markdown(stem: str) -> str:
    """MinerU markdown extracted from the library PDF with *stem*."""
    return f"{MINERU}/{stem}.md"


def library_images_prefix(stem: str) -> str:
    """MinerU images extracted from the library PDF with *stem*."""
    return f"{MINERU}/images/{stem}"



def architecture_prefix(slug: str) -> str:
    """One LikeC4 workspace per project: its `.c4` sources plus `.likec4/` saved layouts."""
    return f"{GRAPH}/{_part(slug, 'project slug')}"


def chat_upload_prefix(chat_id: str, slug: str | None = None) -> str:
    """Attachment prefix for a chat, project-scoped when *slug* is given."""
    chat = _part(chat_id, "chat ID")
    if slug:
        return f"{ATTACHMENT}/project/{_part(slug, 'project slug')}/chat/{chat}"
    return f"{ATTACHMENT}/chat/{chat}"


def project_attachment_prefix(slug: str) -> str:
    return f"{ATTACHMENT}/project/{_part(slug, 'project slug')}/chat"


def chat_upload(chat_id: str, name: str, slug: str | None = None) -> str:
    return f"{chat_upload_prefix(chat_id, slug)}/{_part(name, 'attachment name')}"


def sandbox_scope(scope: str) -> str:
    return _part(scope, "sandbox scope")


def sandbox_prefix(chat_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{SANDBOX}/{_part(chat_id, 'chat ID')}/{sandbox_scope(scope)}"


def sandbox_slot(chat_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/slot.json"


def sandbox_active(chat_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/active.json"


def sandbox_run(chat_id: str, tool_call_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/runs/{_part(tool_call_id, 'tool call ID')}.json"


def sandbox_run_prefix(chat_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/runs"


def sandbox_manifest(chat_id: str, manifest_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/manifests/{_part(manifest_id, 'manifest ID')}.json"


def sandbox_manifest_prefix(chat_id: str, scope: str = DEFAULT_SANDBOX_SCOPE) -> str:
    return f"{sandbox_prefix(chat_id, scope)}/manifests"


@dataclass(frozen=True)
class SandboxKey:
    """A parsed `sandbox/<chat_id>/<scope>/<manifests|runs>/<file>` record key."""

    chat_id: str
    scope: str
    kind: str
    name: str


def parse_sandbox_key(key: str) -> SandboxKey | None:
    """The parts of a sandbox manifest or run key, None for any other key."""
    parts = key.split("/")
    if len(parts) != 5 or parts[0] != SANDBOX or parts[3] not in ("manifests", "runs"):
        return None
    return SandboxKey(chat_id=parts[1], scope=parts[2], kind=parts[3], name=parts[4])




def sandbox_asset(sha256: str) -> str:
    """Content-addressed asset key shared across all chats."""
    return f"{SANDBOX}/assets/{_part(sha256, 'asset sha256')}"


def pdf_annotations(sha256: str) -> str:
    """Annotation canvas of a PDF, keyed by the SHA-256 of its bytes."""
    return f"{PDF_ANNOTATION}/{_part(sha256, 'PDF sha256')}.canvas"
