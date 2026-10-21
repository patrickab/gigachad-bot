from collections.abc import Awaitable
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.params import Depends as DependsMarker

from backend.identity import HOST_CAPABILITY, RequestIdentity, get_request_identity
from lib.upstream import UpstreamContractError, UpstreamRequestFailed, UpstreamUnavailable


def host_access_only(feature: str) -> DependsMarker:
    """Router dependency refusing logins without the host capability, for features that act on the machine owner's host."""

    def require_host_access(identity: Annotated[RequestIdentity, Depends(get_request_identity)]) -> None:
        if not identity.has_host_access:
            raise HTTPException(status_code=403, detail=f"{feature} is limited to logins granted {HOST_CAPABILITY}")

    return Depends(require_host_access)


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
