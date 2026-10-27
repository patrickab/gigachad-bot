"""Request and response models shared across routes."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lib.document_library import document_meta


class ChatSnapshot(BaseModel):
    """A chat as the client saves it, whether into the history or into a project tab."""

    messages: list[dict[str, Any]] = []
    chat_id: str | None = None
    title: str | None = None
    usage: dict[str, int] | None = None
    parent_id: str | None = None
    branch_message_idx: int | None = None
    children: list[dict[str, Any]] | None = None


class FileMeta(BaseModel):
    path: str
    name: str
    mime: str


def file_meta(path: str | Path, **override: str) -> FileMeta:
    """The API description of *path*, with any `path`/`name`/`mime` field replaced by *override*."""
    return FileMeta(**{**document_meta(path), **override})


class FileListResponse(BaseModel):
    documents: list[FileMeta]


class FileContent(BaseModel):
    path: str
    content: str


class AttachResult(BaseModel):
    name: str
    mime: str
    # Set by file-root attaches only, stored on the Attachment and used to re-read content
    path: str | None = None
    content: str | None = None
    parsedMd: str | None = None
