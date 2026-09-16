"""Response and request models shared by the file-shaped routes.

The documents, files, file-vault and file-viewer routes all describe the same
handful of shapes: a file's metadata, a file's text content, and the result of
attaching a file to a chat. Declared once here so the JSON contract stays
identical whichever route the frontend asks.
"""

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
    # Set only by live-reference attaches (file vaults): the frontend stores it
    # on the Attachment and reads content back from that path. Copy-style
    # attaches leave it unset.
    path: str | None = None
    content: str | None = None
    parsedMd: str | None = None
