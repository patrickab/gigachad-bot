from collections.abc import Awaitable

from fastapi import Request
from fastapi.responses import JSONResponse

from lib.upstream import UpstreamContractError, UpstreamRequestFailed, UpstreamUnavailable


async def _upstream_error(_request: Request, exc: Exception) -> JSONResponse:
    """An adapter failure as HTTP: unreachable 503, unknown shape 502, refusal passed through."""
    status = exc.status if isinstance(exc, UpstreamRequestFailed) else 502 if isinstance(exc, UpstreamContractError) else 503
    return JSONResponse(status_code=status, content={"detail": str(exc)})


# The apps' `exception_handlers`: routes let the `lib.upstream` failures escape, mapped here once.
UPSTREAM_ERRORS = dict.fromkeys((UpstreamUnavailable, UpstreamContractError, UpstreamRequestFailed), _upstream_error)


async def call_upstream[Result](request: Awaitable[Result]) -> Result:
    """Kept only for `routes/factory.py` (another change owns it): `UPSTREAM_ERRORS` maps the failures."""
    return await request
