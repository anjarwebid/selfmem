"""GDPR helpers: account-wide export + delete.

Both modes (selfhosted + saas) expose this. Self-hosted users still benefit
from a clean data-export feature even without legal pressure.
"""
from __future__ import annotations

import logging

import db

log = logging.getLogger(__name__)


async def export_user_data(user_id: str) -> dict:
    """Collect everything tied to this user_id into a single JSON-friendly dict."""
    user = await db.get_user_by_id(user_id)
    orgs = await db.list_user_orgs(user_id)
    accessible = await db.list_user_accessible_projects(user_id)
    keys = await db.list_user_api_keys(user_id)

    project_ids = [p["id"] for p in accessible]
    memories = await db.export_memories(project_ids) if project_ids else []

    audit_per_org = []
    for o in orgs:
        events = await db.list_audit(o["id"], limit=10000)
        audit_per_org.append({"org_slug": o["slug"], "events": events})

    return {
        "exported_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "user": user,
        "orgs": orgs,
        "projects": accessible,
        "api_keys": [
            {k: v for k, v in key.items() if k != "key_hash"}
            for key in keys
        ],
        "memories": memories,
        "audit_log": audit_per_org,
    }


async def delete_account(user_id: str) -> dict:
    """Soft-delete the user, revoke their keys, hard-delete any orgs where
    they were the sole owner (cascading), and remove org_member/project_member
    rows from any remaining shared orgs (so member counts stay consistent).
    """
    only_owner_orgs = await db.list_user_owned_only_orgs(user_id)
    keys_revoked = await db.revoke_all_user_keys(user_id)

    deleted_orgs = []
    for org in only_owner_orgs:
        await db.delete_org(org["id"])
        deleted_orgs.append({"slug": org["slug"], "name": org["name"], "is_personal": org["is_personal"]})

    # For any remaining shared orgs, drop the user's org_member row.
    # cascading project_members rows go with them via FK on user_id.
    remaining = await db.list_user_orgs(user_id)
    removed_from = []
    for org in remaining:
        await db.remove_org_member(org["id"], user_id)
        removed_from.append({"slug": org["slug"], "name": org["name"]})

    await db.soft_delete_user(user_id)

    log.info("[gdpr] account deleted user=%s keys=%d orgs_destroyed=%d orgs_left=%d",
             user_id, keys_revoked, len(deleted_orgs), len(removed_from))
    return {
        "user_id": user_id,
        "keys_revoked": keys_revoked,
        "orgs_deleted": deleted_orgs,
        "orgs_left": removed_from,
    }
