"""Request and response models shared by the file-shaped routes."""

from pydantic import BaseModel


class FileMeta(BaseModel):
    path: str
    name: str
    mime: str


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
