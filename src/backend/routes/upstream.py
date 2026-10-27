from collections.abc import Awaitable

from fastapi import HTTPException

from lib.upstream import UpstreamContractError, UpstreamRequestFailed, UpstreamUnavailable


async def call_upstream[Result](request: Awaitable[Result]) -> Result:
    """Await an adapter call, mapping its failures to HTTP: unreachable 503, unknown shape 502, refusal passed through."""
    try:
        return await request
    except UpstreamUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UpstreamContractError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except UpstreamRequestFailed as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
