"""Quota enforcement.

In selfhosted mode (HOSTING_MODE=selfhosted) every helper short-circuits to
True without touching the DB — single env-flag turns SaaS limits on/off.
"""
from __future__ import annotations

import logging

import config
import db

log = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    def __init__(self, resource: str, limit: int, org_slug: str | None = None):
        self.resource = resource
        self.limit = limit
        self.org_slug = org_slug
        super().__init__(f"quota_exceeded:{resource}:{limit}")


async def check_memory_quota(org_id: str, plan_tier: str, org_slug: str | None = None) -> None:
    if not config.IS_SAAS or plan_tier != "free":
        return
    usage = await db.get_org_usage(org_id)
    count = (usage or {}).get("memory_count", 0)
    if count >= config.FREE_MEMORY_LIMIT:
        raise QuotaExceeded("memory", config.FREE_MEMORY_LIMIT, org_slug)


async def check_project_quota(org_id: str, plan_tier: str, org_slug: str | None = None) -> None:
    if not config.IS_SAAS or plan_tier != "free":
        return
    usage = await db.get_org_usage(org_id)
    count = (usage or {}).get("project_count", 0)
    if count >= config.FREE_PROJECT_LIMIT:
        raise QuotaExceeded("project", config.FREE_PROJECT_LIMIT, org_slug)


async def check_member_quota(org_id: str, plan_tier: str, org_slug: str | None = None) -> None:
    if not config.IS_SAAS or plan_tier != "free":
        return
    usage = await db.get_org_usage(org_id)
    count = (usage or {}).get("member_count", 0)
    if count >= config.FREE_MEMBER_LIMIT:
        raise QuotaExceeded("member", config.FREE_MEMBER_LIMIT, org_slug)


async def check_api_call_quota(org_id: str, plan_tier: str, org_slug: str | None = None) -> int:
    """Atomically increment today's API call counter for the org and check the limit.
    Returns the post-increment count. Raises QuotaExceeded if over the limit.
    No-op in selfhosted mode (returns 0)."""
    if not config.IS_SAAS:
        return 0
    count = await db.increment_api_calls(org_id)
    limit = config.FREE_API_LIMIT_DAILY if plan_tier == "free" else config.PAID_API_LIMIT_DAILY
    if count > limit:
        raise QuotaExceeded("api_call", limit, org_slug)
    return count
