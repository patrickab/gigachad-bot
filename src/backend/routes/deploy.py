import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.routes.upstream import host_access_only

router = APIRouter(prefix="/api/deploy", tags=["deploy"], dependencies=[host_access_only("redeploying")])

REDEPLOY_SCRIPT = Path(__file__).resolve().parents[3] / "redeploy.sh"
REDEPLOY_UNIT = "gigachad-bot-redeploy"


@router.post("/redeploy", status_code=202)
async def post_redeploy() -> None:
    """Start `redeploy.sh` as a transient user unit, it restarts this backend so it cannot be our child.

    The fixed unit name refuses a second run while one is active (409).
    """
    process = await asyncio.create_subprocess_exec(
        "systemd-run", "--user", "--collect", f"--unit={REDEPLOY_UNIT}", "bash", str(REDEPLOY_SCRIPT),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise HTTPException(status_code=409, detail=stderr.decode(errors="replace").strip() or "Could not start the redeploy")
