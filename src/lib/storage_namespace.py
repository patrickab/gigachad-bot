"""Canonical database namespaces for application-owned artifacts.

These keys identify database records. They are never derived from the local
Documents directory layout. ``PDFS`` and ``MINERU`` retain their physical names
because those are the only artifact classes permitted to mirror to Documents.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from lib.data_store import InvalidStorageKey, validate_key

CHAT = "chat"
PROJECT = "project"
NOTE = "note"
ATTACHMENT = "attachment"
DRAWING = "drawing"
SAVED = "saved"
GRAPH = "graph"
PROMPT = "prompt"
MEMORY = "memory"
MODEL = "model"
PDFS = "PDFs"
MINERU = "Mineru"


def _part(value: str, label: str) -> str:
    try:
        parsed = validate_key(value)
    except InvalidStorageKey as exc:
        raise ValueError(f"Invalid {label}") from exc
    if "/" in parsed:
        raise ValueError(f"Invalid {label}")
    return parsed


def project_document(slug: str, name: str) -> str:
    return f"{PROJECT}/{_part(slug, 'project slug')}/document/{_part(name, 'document name')}"


def project_memory(slug: str, name: str) -> str:
    return f"{MEMORY}/project/{_part(slug, 'project slug')}/{_part(name, 'memory name')}"


def project_document_prefix(slug: str) -> str:
    return f"{PROJECT}/{_part(slug, 'project slug')}/document"


def chat_upload_prefix(chat_id: str, slug: str | None = None) -> str:
    chat = _part(chat_id, "chat ID")
    if slug:
        return f"{ATTACHMENT}/project/{_part(slug, 'project slug')}/chat/{chat}"
    return f"{ATTACHMENT}/chat/{chat}"


def project_attachment_prefix(slug: str) -> str:
    return f"{ATTACHMENT}/project/{_part(slug, 'project slug')}/chat"

def legacy_key_to_native(key: str) -> str:
    """Translate one known Documents-relative key into its database namespace."""
    normalized = validate_key(key)
    parts = PurePosixPath(normalized).parts

    def under(prefix: str, rest: tuple[str, ...]) -> str:
        return f"{prefix}/{'/'.join(rest)}" if rest else prefix

    if parts[:2] == ("chat_history", "_notes"):
        return under(NOTE, parts[2:])
    if parts[:2] == ("chat_history", "_uploads"):
        return under(f"{ATTACHMENT}/chat", parts[2:])
    if len(parts) >= 3 and parts[0] == "chat_history" and parts[2] == "_uploads":
        return under(f"{ATTACHMENT}/project/{parts[1]}/chat", parts[3:])
    if len(parts) >= 3 and parts[0] == "chat_history" and parts[2] == "documents":
        return under(f"{PROJECT}/{parts[1]}/document", parts[3:])
    if parts[:2] == ("chat_history", "memory"):
        return under(MEMORY, parts[2:])
    if len(parts) >= 3 and parts[0] == "chat_history" and parts[2] == "memory":
        return under(f"{MEMORY}/project/{parts[1]}", parts[3:])
    if parts and parts[0] == "chat_history":
        return under(CHAT, parts[1:])
    if parts[:2] == ("Architecture_Graphs", ".drafts"):
        return under(f"{GRAPH}/draft", parts[2:])
    roots = {
        "Architecture_Graphs": GRAPH,
        "Drawings": DRAWING,
        "Prompts": PROMPT,
        "Markdown": f"{SAVED}/markdown",
        "LaTeX": f"{SAVED}/latex",
    }
    if parts and parts[0] in roots:
        return under(roots[parts[0]], parts[1:])
    if normalized in {"model-providers.yaml", "model-defaults.yaml", "model-tab-order.yaml"}:
        return f"{MODEL}/{normalized}"
    return normalized


def chat_upload(chat_id: str, name: str, slug: str | None = None) -> str:
    return f"{chat_upload_prefix(chat_id, slug)}/{_part(name, 'attachment name')}"


def asset_filename(key: str) -> str:
    return PurePosixPath(validate_key(key)).name
