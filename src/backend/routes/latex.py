"""LaTeX builds of one `.tex` file, for logins with host access."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.deps import get_file_roots
from backend.routes.fileviewer import resolve_allowed
from lib import latex
from lib.file_roots import FileRoots

router = APIRouter(prefix="/api/latex", tags=["latex"], dependencies=[host_access_only("LaTeX")])


class TexPath(BaseModel):
    """Absolute path of the `.tex` file."""

    path: str


def _resolve_tex_file(path: str, roots: FileRoots) -> Path:
    resolved = resolve_allowed(path, roots)
    if resolved.suffix != ".tex":
        raise HTTPException(status_code=400, detail="Not a .tex file")
    return resolved


@router.post("/build")
async def post_build(body: TexPath, roots: FileRoots = Depends(get_file_roots)) -> latex.LatexBuild:
    source = await asyncio.to_thread(_resolve_tex_file, body.path, roots)
    with http_errors((latex.LatexUnavailable, 503, None)):
        return await latex.build(source)
