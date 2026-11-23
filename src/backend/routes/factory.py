"""`/api/factory/*`: the factory core's actions for logins with host access (`backend.identity.HOST_CAPABILITY`).

- Consumes `lib.pi_host.factory_call`/`factory_actions`: pi-host runs the action in the factory plugin's core, which owns
  repos, settings, `factory.yaml`, tasks and their runners. This backend keeps no factory state
- Provides `POST /api/factory/call {action, params}` (the core's result as it is) and `GET /api/factory/actions` (`{actions}`)
- The one behaviour of its own: after a successful `tasks.remove` it stops the removed task's preview
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
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

from backend.http_plumbing import http_errors
from backend.identity import host_access_only
from backend.routes.upstream import call_upstream
from lib import factory_images, factory_preview, pi_host

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


@router.post("/call")
async def post_call(body: FactoryCall) -> Any:  # noqa: ANN401 - the core's JSON, as each action shapes it
    """Run the core action *action* with *params*; the answer is its result."""
    result = await factory_result(pi_host.factory_call(body.action, body.params))
    if body.action == "tasks.remove":
        await _stop_removed_preview(result["removed"])
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
