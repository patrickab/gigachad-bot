"""Trusted Tailnet identity and browser-device registration.

- Tailscale Serve injects `Tailscale-User-Login` and, for granted users, `Tailscale-App-Capabilities` (`--accept-app-caps`)
- Serve strips client-sent copies of both, and the backend listens on loopback only, so neither can be forged from the tailnet
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, HTTPException
from fastapi.params import Depends as DependsMarker

from config import get_postgres_pool, is_development

# Granted in the tailnet policy file: the holder acts as the host user (omp, Neovim, every file outside the mounts).
HOST_CAPABILITY = "gigachad.local/cap/host"
CAPABILITIES_HEADER = "Tailscale-App-Capabilities"


@dataclass(frozen=True)
class RequestIdentity:
    user_id: UUID
    login: str
    device_id: UUID | None
    has_host_access: bool = False


def _development_login() -> str | None:
    return os.environ.get("GIGACHAD_DEV_USER") if is_development() else None


def is_host_capability_granted(capabilities_header: str | None) -> bool:
    """Whether Serve's `Tailscale-App-Capabilities` JSON grants `HOST_CAPABILITY`, malformed counts as not granted."""
    if not capabilities_header:
        return False
    try:
        capabilities = json.loads(capabilities_header)
    except ValueError:
        return False
    return isinstance(capabilities, dict) and HOST_CAPABILITY in capabilities


def has_host_access(login: str, capabilities_header: str | None) -> bool:
    """The local dev login always acts as the host user, tailnet logins only with the granted capability."""
    return login == _development_login() or is_host_capability_granted(capabilities_header)


def get_request_identity(
    tailscale_login: str | None = Header(default=None, alias="Tailscale-User-Login"),
    capabilities_header: str | None = Header(default=None, alias=CAPABILITIES_HEADER),
    device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> RequestIdentity:
    """Resolve a proxy-injected Tailscale login to one database user."""
    login = tailscale_login or _development_login()
    if not login:
        raise HTTPException(status_code=401, detail="Tailscale identity is required")
    # EventSource cannot send headers, so a device id is optional: it only labels
    # writes for echo suppression, never grants access.
    try:
        parsed_device_id = UUID(device_id) if device_id else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Device-Id must be a UUID") from exc

    with get_postgres_pool().connection() as connection, connection.transaction():
        user_id = connection.execute(
            """
            INSERT INTO users (tailscale_login) VALUES (%s)
            ON CONFLICT (tailscale_login) DO UPDATE SET tailscale_login = EXCLUDED.tailscale_login
            RETURNING id
            """,
            (login,),
        ).fetchone()[0]
        if parsed_device_id is not None:
            connection.execute(
                """
                INSERT INTO devices (id, user_id) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET user_id = EXCLUDED.user_id, last_seen_at = now()
                """,
                (parsed_device_id, user_id),
            )
    return RequestIdentity(
        user_id=user_id, login=login, device_id=parsed_device_id, has_host_access=has_host_access(login, capabilities_header)
    )


def host_denied(feature: str) -> str:
    """The refusal message for *feature* when the login lacks `HOST_CAPABILITY`."""
    return f"{feature} is limited to logins granted {HOST_CAPABILITY}"


def host_access_only(feature: str) -> DependsMarker:
    """Router dependency refusing logins without the host capability, for features that act on the machine owner's host."""

    def require_host_access(identity: RequestIdentity = Depends(get_request_identity)) -> None:
        if not identity.has_host_access:
            raise HTTPException(status_code=403, detail=host_denied(feature))

    return Depends(require_host_access)
