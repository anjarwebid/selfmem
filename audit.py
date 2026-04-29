"""Audit log helper. Wraps db.write_audit with safe defaults and never blocks
the request — failures are logged but swallowed."""
from __future__ import annotations

import asyncio
import logging

import db

log = logging.getLogger(__name__)


def log_event(
    org_id: str,
    action: str,
    actor_user_id: str | None = None,
    actor_api_key_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
    ip_address: str | None = None,
) -> asyncio.Task:
    """Fire-and-forget audit write. Returns the task so callers can await if they want."""
    return asyncio.create_task(
        _write(
            org_id, action, actor_user_id, actor_api_key_id,
            resource_type, resource_id, metadata, ip_address,
        )
    )


async def _write(
    org_id: str, action: str, actor_user_id, actor_api_key_id,
    resource_type, resource_id, metadata, ip_address,
) -> None:
    try:
        await db.write_audit(
            org_id=org_id,
            action=action,
            actor_user_id=actor_user_id,
            actor_api_key_id=actor_api_key_id,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata,
            ip_address=ip_address,
        )
    except Exception:
        log.exception("audit log write failed for action=%s org=%s", action, org_id)
