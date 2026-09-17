"""Trusted Tailnet identity and browser-device registration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import UUID

from fastapi import Header, HTTPException

from config import get_postgres_pool, storage_mode


@dataclass(frozen=True)
class RequestIdentity:
    user_id: UUID
    login: str
    device_id: UUID | None


def _development_login() -> str | None:
    if os.environ.get("GIGACHAD_ENV") == "development":
        return os.environ.get("GIGACHAD_DEV_USER")
    return None


def get_request_identity(
    tailscale_login: str | None = Header(default=None, alias="Tailscale-User-Login"),
    device_id: str | None = Header(default=None, alias="X-Device-Id"),
) -> RequestIdentity | None:
    """Resolve a proxy-injected Tailscale login to one database user."""
    if storage_mode() == "local":
        return None
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
    return RequestIdentity(user_id=user_id, login=login, device_id=parsed_device_id)
