"""`/api/factory/*`: the factory core's actions for logins with host access (`backend.identity.HOST_CAPABILITY`).

- Consumes `lib.pi_host.factory_call`/`factory_actions`: pi-host runs the action in the factory plugin's core, which owns
  repos, settings, `factory.yaml`, tasks and their runners. This backend keeps no factory state
- Provides `POST /api/factory/call {action, params}` (the core's result as it is) and `GET /api/factory/actions` (`{actions}`)
- `tasks.start` without a `tag` gets the requesting login as its tag: the core would fall back to the host's system
  user, which every login shares. Characters the core refuses in a tag segment become `-` (`login_tag`)
- `tasks.remove` of a merged (`done`) task: a dry run (`git merge-tree`) first refuses a conflicting revert with 409
  `refused`, then, after the core removed the task, `git revert` of its merge commit in the repo's checkout
  (`lib.factory_revert`). Reverting last matters: it rewrites the checkout, which reloads a dev server mid-request
- Another behaviour of its own: after a successful `tasks.remove` it stops the removed task's preview
  (`lib.factory_preview`, keyed by the result's `removed`), so a refused removal keeps the preview
- `POST /api/factory/images` stores an image for a task's request or message (`lib.factory_images`) and answers its
  absolute `{path}`, which the frontend appends to the text: the core takes text only. No png/jpg/gif/webp 400, over 20 MB 413
- A core refusal answers `{"detail": {"code", "message", "detail"?}}` (`detail` the core's hint, e.g. `{"force": true}`)
  with 400 `invalid`, 404 `not_found`, 409 `busy`/`refused`, 500 `failed`; 503 `factory_plugin_missing`; a preview that
  does not stop after the removal 502 `failed`. pi-host unreachable 503, an unknown answer 502 (`call_upstream`)

Host code execution: `host_access_only`.
"""

import asyncio
from collections.abc import Awaitable
from datetime import datetime
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from backend.http_plumbing import http_errors
from backend.identity import RequestIdentity, get_request_identity, host_access_only
from backend.routes.upstream import call_upstream
from lib import factory_images, factory_preview, factory_revert, pi_host

router = APIRouter(prefix="/api/factory", tags=["factory"], dependencies=[host_access_only("the factory")])

FACTORY_ERROR_STATUS: dict[pi_host.FactoryErrorCode, int] = {
    "invalid": 400,
    "not_found": 404,
    "busy": 409,
    "refused": 409,
    "failed": 500,
}


class FactoryCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    params: dict[str, Any] = {}


def _error(status: int, code: str, message: str, detail: dict[str, Any] | None = None) -> HTTPException:
    body: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return HTTPException(status_code=status, detail=body)


async def factory_result[Result](request: Awaitable[Result]) -> Result:
    """Await a factory call, its refusals as `{"detail": {"code", "message", "detail"?}}`, pi-host's failures as
    `call_upstream` maps them."""
    try:
        return await call_upstream(request)
    except pi_host.FactoryCallError as exc:
        raise _error(FACTORY_ERROR_STATUS[exc.code], exc.code, exc.message, exc.detail) from exc
    except pi_host.FactoryPluginMissing as exc:
        raise _error(503, "factory_plugin_missing", str(exc)) from exc


async def _stop_removed_preview(task_id: str) -> None:
    """Stop the preview of the task the core just removed."""
    try:
        await asyncio.to_thread(factory_preview.stop_preview, task_id)
    except factory_preview.PreviewError as exc:
        raise _error(502, "failed", f"Task {task_id} was removed, its preview did not stop: {exc}") from exc


async def _commit_to_revert(params: dict[str, Any]) -> tuple[str, str] | None:
    """`(repo, commit)` of a merged task's code that deleting it must drop, None when there is none.

    A revert that would conflict refuses the delete here, before anything is removed."""
    task = await factory_result(pi_host.factory_call("tasks.get", {"id": params.get("id")}))
    if task.get("state") != "done" or task.get("reason") == "no_changes":
        return None
    timeline = await factory_result(pi_host.factory_call("tasks.timeline", {"id": task["id"]}))
    commit = factory_revert.merged_commit(timeline["entries"])
    if commit is None:
        return None
    try:
        pending = await asyncio.to_thread(factory_revert.check_revert, task["repo"], commit)
    except factory_revert.RevertError as exc:
        raise _error(409, "refused", f"{exc}. The task was not deleted") from exc
    return (task["repo"], commit) if pending else None


async def _revert_after_removal(repo: str, commit: str) -> None:
    """Revert the commit once the task is gone: the revert rewrites the checkout, which reloads a dev server and
    would cut a request still waiting to delete."""
    try:
        await asyncio.to_thread(factory_revert.revert_merge, repo, commit)
    except factory_revert.RevertError as exc:
        raise _error(409, "failed", f"The task was deleted, its code stays: {exc}") from exc


# What the core accepts in one tag segment: a login with anything else would make every start of it fail.
_TAG_SEGMENT_REFUSED = re.compile(r"[^A-Za-z0-9._@+-]")


def login_tag(login: str) -> str:
    """*login* as a one-segment tag: each character the core refuses in a tag replaced by `-`."""
    return _TAG_SEGMENT_REFUSED.sub("-", login)


@router.post("/call")
async def post_call(body: FactoryCall, identity: Annotated[RequestIdentity, Depends(get_request_identity)]) -> Any:  # noqa: ANN401 - the core's JSON, as each action shapes it
    """Run the core action *action* with *params*; the answer is its result."""
    params = {"tag": login_tag(identity.login)} | body.params if body.action == "tasks.start" else body.params
    revert = await _commit_to_revert(params) if body.action == "tasks.remove" else None
    result = await factory_result(pi_host.factory_call(body.action, params))
    if body.action == "tasks.remove":
        await _stop_removed_preview(result["removed"])
        if revert is not None:
            await _revert_after_removal(*revert)
    return result


@router.get("/actions")
async def get_actions() -> dict[str, Any]:
    """The core's actions: name -> `{summary, params (JSON schema), mutates}`."""
    return {"actions": await factory_result(pi_host.factory_actions())}


@router.post("/images")
async def post_image(file: Annotated[UploadFile, File()]) -> dict[str, str]:
    """Store an image attached to a task's request or message, answer its absolute path."""
    filename = file.filename or ""
    with http_errors((ValueError, 400, None)):
        suffix = factory_images.image_suffix(filename, file.content_type)
    content = await file.read(factory_images.MAX_IMAGE_BYTES + 1)
    if len(content) > factory_images.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Images are at most 20 MB")
    stored = await asyncio.to_thread(factory_images.store_image, filename, suffix, content, datetime.now())
    return {"path": str(stored)}
