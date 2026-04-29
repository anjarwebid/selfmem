from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
from pgvector.asyncpg import register_vector

import config

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- Users
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    email_verified  BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ,
    deleted_at      TIMESTAMPTZ
);

-- Orgs (workspaces)
CREATE TABLE IF NOT EXISTS orgs (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                        VARCHAR(100) UNIQUE NOT NULL,
    name                        VARCHAR(255) NOT NULL,
    is_personal                 BOOLEAN DEFAULT FALSE,
    plan_tier                   VARCHAR(20) DEFAULT 'free' CHECK (plan_tier IN ('free', 'unlimited')),
    stripe_customer_id          VARCHAR(255),
    stripe_subscription_id      VARCHAR(255),
    subscription_active_until   TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS org_members (
    org_id     UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role       VARCHAR(20) NOT NULL CHECK (role IN ('owner', 'member')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (org_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members (user_id);

-- Projects (globally unique slug; FK target for memories)
CREATE TABLE IF NOT EXISTS projects (
    id          VARCHAR(100) PRIMARY KEY,
    org_id      UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name        VARCHAR(255),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_projects_org ON projects (org_id);

CREATE TABLE IF NOT EXISTS project_members (
    project_id  VARCHAR(100) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members (user_id);

-- API keys with project ACL
CREATE TABLE IF NOT EXISTS api_keys (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    key_prefix    VARCHAR(12)  NOT NULL,
    key_hash      VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash_active
    ON api_keys (key_hash) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys (user_id);

CREATE TABLE IF NOT EXISTS api_key_projects (
    api_key_id  UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    project_id  VARCHAR(100) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    PRIMARY KEY (api_key_id, project_id)
);

-- Invitations + password resets
CREATE TABLE IF NOT EXISTS invitations (
    token        VARCHAR(64) PRIMARY KEY,
    org_id       UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    email        VARCHAR(255),
    role         VARCHAR(20) NOT NULL CHECK (role IN ('owner', 'member')),
    created_by   UUID REFERENCES users(id),
    expires_at   TIMESTAMPTZ NOT NULL,
    used_at      TIMESTAMPTZ,
    used_by      UUID REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS password_resets (
    token       VARCHAR(64) PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ
);

-- SaaS plumbing (counters + audit + Stripe events) — written always
CREATE TABLE IF NOT EXISTS org_usage (
    org_id              UUID PRIMARY KEY REFERENCES orgs(id) ON DELETE CASCADE,
    memory_count        INT DEFAULT 0,
    project_count       INT DEFAULT 0,
    member_count        INT DEFAULT 1,
    api_calls_today     INT DEFAULT 0,
    api_calls_reset_on  DATE DEFAULT CURRENT_DATE,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id           UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    actor_user_id    UUID REFERENCES users(id),
    actor_api_key_id UUID REFERENCES api_keys(id),
    action           VARCHAR(100) NOT NULL,
    resource_type    VARCHAR(50),
    resource_id      VARCHAR(255),
    metadata         JSONB,
    ip_address       INET,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_org_time ON audit_log (org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log (actor_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS stripe_events (
    id           VARCHAR(255) PRIMARY KEY,
    type         VARCHAR(100) NOT NULL,
    payload      JSONB NOT NULL,
    received_at  TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

-- Memories (existing) + FK on project_id
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    tags TEXT[] DEFAULT '{{}}'::TEXT[],
    embedding vector({config.EMBEDDING_DIM}),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ DEFAULT NULL,
    pinned BOOLEAN DEFAULT FALSE
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_memories_project' AND table_name = 'memories'
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT fk_memories_project
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE;
    END IF;
END $$;
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (to_tsvector('english', content))",
    "CREATE INDEX IF NOT EXISTS idx_memories_project_category ON memories (project_id, category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_project_active ON memories (project_id) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_memories_project_pinned ON memories (project_id, pinned) WHERE pinned = TRUE AND deleted_at IS NULL",
]

async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_db() -> None:
    global _pool
    bootstrap = await asyncpg.connect(config.DATABASE_URL)
    try:
        await bootstrap.execute(SCHEMA_SQL)
        for idx_sql in INDEX_SQL:
            await bootstrap.execute(idx_sql)
    finally:
        await bootstrap.close()

    _pool = await asyncpg.create_pool(
        config.DATABASE_URL,
        min_size=2,
        max_size=10,
        init=_init_conn,
    )
    log.info("Database initialized (hosting_mode=%s)", config.HOSTING_MODE)


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _row_to_dict(row: asyncpg.Record) -> dict:
    d = dict(row)
    for key, val in list(d.items()):
        if isinstance(val, uuid.UUID):
            d[key] = str(val)
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
    if "embedding" in d:
        del d["embedding"]
    if "tags" in d and d["tags"] is None:
        d["tags"] = []
    return d


# =============================================================================
# Users
# =============================================================================

async def create_user(email: str, password_hash: str) -> dict:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (email, password_hash)
            VALUES ($1, $2)
            RETURNING id, email, email_verified, created_at, last_login_at, deleted_at
            """,
            email.lower(),
            password_hash,
        )
    return _row_to_dict(row)


async def get_user_by_email(email: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, password_hash, email_verified, created_at, last_login_at, deleted_at "
            "FROM users WHERE email = $1 AND deleted_at IS NULL",
            email.lower(),
        )
    return _row_to_dict(row) if row else None


async def get_user_by_id(user_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, email_verified, created_at, last_login_at, deleted_at "
            "FROM users WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(user_id),
        )
    return _row_to_dict(row) if row else None


async def update_last_login(user_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login_at = NOW() WHERE id = $1",
            uuid.UUID(user_id),
        )


async def update_user_password(user_id: str, password_hash: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET password_hash = $2 WHERE id = $1",
            uuid.UUID(user_id),
            password_hash,
        )


async def soft_delete_user(user_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET deleted_at = NOW() WHERE id = $1",
            uuid.UUID(user_id),
        )


# =============================================================================
# Orgs + members
# =============================================================================

async def create_org(slug: str, name: str, is_personal: bool, owner_id: str) -> dict:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO orgs (slug, name, is_personal)
                VALUES ($1, $2, $3)
                RETURNING id, slug, name, is_personal, plan_tier, created_at
                """,
                slug,
                name,
                is_personal,
            )
            org_id = row["id"]
            await conn.execute(
                "INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, 'owner')",
                org_id,
                uuid.UUID(owner_id),
            )
            await conn.execute(
                "INSERT INTO org_usage (org_id, member_count) VALUES ($1, 1)",
                org_id,
            )
    return _row_to_dict(row)


async def get_org_by_slug(slug: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, slug, name, is_personal, plan_tier, stripe_customer_id, "
            "stripe_subscription_id, subscription_active_until, created_at "
            "FROM orgs WHERE slug = $1",
            slug,
        )
    return _row_to_dict(row) if row else None


async def get_org_by_id(org_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, slug, name, is_personal, plan_tier, stripe_customer_id, "
            "stripe_subscription_id, subscription_active_until, created_at "
            "FROM orgs WHERE id = $1",
            uuid.UUID(org_id),
        )
    return _row_to_dict(row) if row else None


async def list_user_orgs(user_id: str) -> list[dict]:
    """Orgs the user belongs to, with their role."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT o.id, o.slug, o.name, o.is_personal, o.plan_tier, o.created_at, m.role
            FROM orgs o
            JOIN org_members m ON m.org_id = o.id
            WHERE m.user_id = $1
            ORDER BY o.is_personal DESC, o.created_at
            """,
            uuid.UUID(user_id),
        )
    return [_row_to_dict(r) for r in rows]


async def slug_exists(slug: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1 FROM orgs WHERE slug = $1", slug)
    return result is not None


async def get_org_member(org_id: str, user_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id, user_id, role, created_at FROM org_members "
            "WHERE org_id = $1 AND user_id = $2",
            uuid.UUID(org_id),
            uuid.UUID(user_id),
        )
    return _row_to_dict(row) if row else None


async def list_org_members(org_id: str) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT u.id AS user_id, u.email, m.role, m.created_at
            FROM org_members m
            JOIN users u ON u.id = m.user_id
            WHERE m.org_id = $1 AND u.deleted_at IS NULL
            ORDER BY m.role, m.created_at
            """,
            uuid.UUID(org_id),
        )
    return [_row_to_dict(r) for r in rows]


async def add_org_member(org_id: str, user_id: str, role: str) -> None:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, $3) "
                "ON CONFLICT DO NOTHING",
                uuid.UUID(org_id),
                uuid.UUID(user_id),
                role,
            )
            await conn.execute(
                "UPDATE org_usage SET member_count = member_count + 1, updated_at = NOW() "
                "WHERE org_id = $1",
                uuid.UUID(org_id),
            )


async def remove_org_member(org_id: str, user_id: str) -> None:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM org_members WHERE org_id = $1 AND user_id = $2",
                uuid.UUID(org_id),
                uuid.UUID(user_id),
            )
            await conn.execute(
                "UPDATE org_usage SET member_count = GREATEST(0, member_count - 1), updated_at = NOW() "
                "WHERE org_id = $1",
                uuid.UUID(org_id),
            )


# =============================================================================
# Projects + members
# =============================================================================

async def create_project(slug: str, org_id: str, name: str | None, owner_user_id: str) -> dict:
    """Create project under org. Owner is auto-added as project_member."""
    async with _pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO projects (id, org_id, name)
                VALUES ($1, $2, $3)
                RETURNING id, org_id, name, created_at
                """,
                slug,
                uuid.UUID(org_id),
                name,
            )
            await conn.execute(
                "INSERT INTO project_members (project_id, user_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                slug,
                uuid.UUID(owner_user_id),
            )
            await conn.execute(
                "UPDATE org_usage SET project_count = project_count + 1, updated_at = NOW() "
                "WHERE org_id = $1",
                uuid.UUID(org_id),
            )
    return _row_to_dict(row)


async def get_project(project_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, org_id, name, created_at FROM projects WHERE id = $1",
            project_id,
        )
    return _row_to_dict(row) if row else None


async def list_org_projects(org_id: str) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, org_id, name, created_at FROM projects WHERE org_id = $1 ORDER BY id",
            uuid.UUID(org_id),
        )
    return [_row_to_dict(r) for r in rows]


async def list_user_accessible_projects(user_id: str) -> list[dict]:
    """All projects the user can access:
    - any project in an org where they are owner
    - any project where they have an explicit project_members row
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT p.id, p.org_id, p.name, p.created_at, o.slug AS org_slug
            FROM projects p
            JOIN orgs o ON o.id = p.org_id
            WHERE p.id IN (
                SELECT id FROM projects WHERE org_id IN (
                    SELECT org_id FROM org_members WHERE user_id = $1 AND role = 'owner'
                )
                UNION
                SELECT project_id FROM project_members WHERE user_id = $1
            )
            ORDER BY p.id
            """,
            uuid.UUID(user_id),
        )
    return [_row_to_dict(r) for r in rows]


async def list_project_members(project_id: str) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT u.id AS user_id, u.email, pm.created_at
            FROM project_members pm
            JOIN users u ON u.id = pm.user_id
            WHERE pm.project_id = $1 AND u.deleted_at IS NULL
            ORDER BY pm.created_at
            """,
            project_id,
        )
    return [_row_to_dict(r) for r in rows]


async def add_project_member(project_id: str, user_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO project_members (project_id, user_id) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING",
            project_id,
            uuid.UUID(user_id),
        )


async def remove_project_member(project_id: str, user_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM project_members WHERE project_id = $1 AND user_id = $2",
            project_id,
            uuid.UUID(user_id),
        )


async def delete_project(project_id: str) -> None:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT org_id FROM projects WHERE id = $1",
                project_id,
            )
            if not row:
                return
            await conn.execute("DELETE FROM projects WHERE id = $1", project_id)
            await conn.execute(
                "UPDATE org_usage SET project_count = GREATEST(0, project_count - 1), updated_at = NOW() "
                "WHERE org_id = $1",
                row["org_id"],
            )


async def project_id_exists(project_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1 FROM projects WHERE id = $1", project_id)
    return result is not None


# =============================================================================
# API keys + ACL
# =============================================================================

async def create_api_key(
    user_id: str, name: str, key_prefix: str, key_hash: str, project_ids: list[str]
) -> dict:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO api_keys (user_id, name, key_prefix, key_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING id, user_id, name, key_prefix, created_at, last_used_at, revoked_at
                """,
                uuid.UUID(user_id),
                name,
                key_prefix,
                key_hash,
            )
            api_key_id = row["id"]
            for pid in project_ids:
                await conn.execute(
                    "INSERT INTO api_key_projects (api_key_id, project_id) VALUES ($1, $2)",
                    api_key_id,
                    pid,
                )
    return _row_to_dict(row)


async def get_api_key_by_hash(key_hash: str) -> dict | None:
    """Look up an active API key by its SHA-256 hash. Returns key + ACL project_ids."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT k.id, k.user_id, k.name, k.key_prefix, k.last_used_at,
                   COALESCE(array_agg(akp.project_id) FILTER (WHERE akp.project_id IS NOT NULL), '{}') AS project_ids
            FROM api_keys k
            LEFT JOIN api_key_projects akp ON akp.api_key_id = k.id
            WHERE k.key_hash = $1 AND k.revoked_at IS NULL
            GROUP BY k.id
            """,
            key_hash,
        )
    if not row:
        return None
    d = _row_to_dict(row)
    d["project_ids"] = list(d.get("project_ids") or [])
    return d


async def list_user_api_keys(user_id: str) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT k.id, k.name, k.key_prefix, k.created_at, k.last_used_at, k.revoked_at,
                   COALESCE(array_agg(akp.project_id) FILTER (WHERE akp.project_id IS NOT NULL), '{}') AS project_ids
            FROM api_keys k
            LEFT JOIN api_key_projects akp ON akp.api_key_id = k.id
            WHERE k.user_id = $1
            GROUP BY k.id
            ORDER BY k.revoked_at NULLS FIRST, k.created_at DESC
            """,
            uuid.UUID(user_id),
        )
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["project_ids"] = list(d.get("project_ids") or [])
        out.append(d)
    return out


async def revoke_api_key(key_id: str, user_id: str) -> bool:
    """Revoke a key, but only if it belongs to user_id (defense in depth)."""
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE api_keys SET revoked_at = NOW() "
            "WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL",
            uuid.UUID(key_id),
            uuid.UUID(user_id),
        )
    return result == "UPDATE 1"


async def update_api_key_acl(key_id: str, user_id: str, project_ids: list[str]) -> bool:
    async with _pool.acquire() as conn:
        async with conn.transaction():
            owner = await conn.fetchval(
                "SELECT user_id FROM api_keys WHERE id = $1",
                uuid.UUID(key_id),
            )
            if not owner or str(owner) != user_id:
                return False
            await conn.execute(
                "DELETE FROM api_key_projects WHERE api_key_id = $1",
                uuid.UUID(key_id),
            )
            for pid in project_ids:
                await conn.execute(
                    "INSERT INTO api_key_projects (api_key_id, project_id) VALUES ($1, $2)",
                    uuid.UUID(key_id),
                    pid,
                )
    return True


async def touch_api_key(key_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
            uuid.UUID(key_id),
        )


# =============================================================================
# Invitations
# =============================================================================

async def create_invitation(
    token: str, org_id: str, email: str | None, role: str, created_by: str, ttl_hours: int = 168
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO invitations (token, org_id, email, role, created_by, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            token,
            uuid.UUID(org_id),
            email,
            role,
            uuid.UUID(created_by),
            expires_at,
        )


async def get_invitation(token: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT i.token, i.org_id, i.email, i.role, i.created_by,
                   i.expires_at, i.used_at, o.slug AS org_slug, o.name AS org_name
            FROM invitations i
            JOIN orgs o ON o.id = i.org_id
            WHERE i.token = $1
            """,
            token,
        )
    return _row_to_dict(row) if row else None


async def consume_invitation(token: str, user_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE invitations SET used_at = NOW(), used_by = $2 WHERE token = $1",
            token,
            uuid.UUID(user_id),
        )


# =============================================================================
# Password resets
# =============================================================================

async def create_password_reset(token: str, user_id: str, ttl_hours: int = 1) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO password_resets (token, user_id, expires_at) VALUES ($1, $2, $3)",
            token,
            uuid.UUID(user_id),
            expires_at,
        )


async def get_password_reset(token: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT token, user_id, expires_at, used_at FROM password_resets WHERE token = $1",
            token,
        )
    return _row_to_dict(row) if row else None


async def consume_password_reset(token: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE password_resets SET used_at = NOW() WHERE token = $1",
            token,
        )


# =============================================================================
# Org usage (quota counters)
# =============================================================================

async def get_org_usage(org_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM org_usage WHERE org_id = $1",
            uuid.UUID(org_id),
        )
    return _row_to_dict(row) if row else None


async def increment_memory_count(org_id: str, delta: int = 1) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE org_usage SET memory_count = GREATEST(0, memory_count + $2), updated_at = NOW() "
            "WHERE org_id = $1",
            uuid.UUID(org_id),
            delta,
        )


async def increment_api_calls(org_id: str) -> int:
    """Atomically bump today's counter (resets across day boundaries). Returns post-increment count."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE org_usage
            SET api_calls_today = CASE
                    WHEN api_calls_reset_on < CURRENT_DATE THEN 1
                    ELSE api_calls_today + 1
                END,
                api_calls_reset_on = CURRENT_DATE,
                updated_at = NOW()
            WHERE org_id = $1
            RETURNING api_calls_today
            """,
            uuid.UUID(org_id),
        )
    return row["api_calls_today"] if row else 0


# =============================================================================
# Audit log
# =============================================================================

async def write_audit(
    org_id: str,
    action: str,
    actor_user_id: str | None = None,
    actor_api_key_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
    ip_address: str | None = None,
) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO audit_log
                (org_id, actor_user_id, actor_api_key_id, action,
                 resource_type, resource_id, metadata, ip_address)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            uuid.UUID(org_id),
            uuid.UUID(actor_user_id) if actor_user_id else None,
            uuid.UUID(actor_api_key_id) if actor_api_key_id else None,
            action,
            resource_type,
            resource_id,
            json.dumps(metadata) if metadata else None,
            ip_address,
        )


# =============================================================================
# Stripe event log (idempotency)
# =============================================================================

async def stripe_event_seen(event_id: str) -> bool:
    """Check if we've already processed this Stripe event."""
    async with _pool.acquire() as conn:
        row = await conn.fetchval(
            "SELECT processed_at FROM stripe_events WHERE id = $1",
            event_id,
        )
    return row is not None


async def record_stripe_event(event_id: str, event_type: str, payload: dict) -> bool:
    """Insert a Stripe event row. Returns True if newly inserted, False if duplicate."""
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO stripe_events (id, type, payload)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            event_id,
            event_type,
            json.dumps(payload),
        )
    return row is not None


async def mark_stripe_event_processed(event_id: str) -> None:
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE stripe_events SET processed_at = NOW() WHERE id = $1",
            event_id,
        )


# =============================================================================
# Org plan / subscription updates
# =============================================================================

async def update_org_plan(
    org_id: str,
    plan_tier: str,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    subscription_active_until: datetime | None = None,
) -> None:
    """Update an org's plan tier + Stripe linkage. Pass None to leave a field unchanged."""
    sets = ["plan_tier = $2"]
    params: list = [uuid.UUID(org_id), plan_tier]
    idx = 3
    if stripe_customer_id is not None:
        sets.append(f"stripe_customer_id = ${idx}")
        params.append(stripe_customer_id)
        idx += 1
    if stripe_subscription_id is not None:
        sets.append(f"stripe_subscription_id = ${idx}")
        params.append(stripe_subscription_id)
        idx += 1
    if subscription_active_until is not None:
        sets.append(f"subscription_active_until = ${idx}")
        params.append(subscription_active_until)
        idx += 1
    async with _pool.acquire() as conn:
        await conn.execute(f"UPDATE orgs SET {', '.join(sets)} WHERE id = $1", *params)


async def get_org_by_stripe_customer(customer_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, slug, name, plan_tier, stripe_customer_id, stripe_subscription_id, subscription_active_until "
            "FROM orgs WHERE stripe_customer_id = $1",
            customer_id,
        )
    return _row_to_dict(row) if row else None


async def get_org_by_stripe_subscription(subscription_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, slug, name, plan_tier, stripe_customer_id, stripe_subscription_id, subscription_active_until "
            "FROM orgs WHERE stripe_subscription_id = $1",
            subscription_id,
        )
    return _row_to_dict(row) if row else None


# =============================================================================
# GDPR helpers
# =============================================================================

async def list_user_owned_only_orgs(user_id: str) -> list[dict]:
    """Orgs where this user is the SOLE owner (will be deleted on account-delete)."""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT o.id, o.slug, o.name, o.is_personal
            FROM orgs o
            JOIN org_members m ON m.org_id = o.id AND m.role = 'owner' AND m.user_id = $1
            WHERE NOT EXISTS (
                SELECT 1 FROM org_members om2
                WHERE om2.org_id = o.id AND om2.role = 'owner' AND om2.user_id != $1
            )
            """,
            uuid.UUID(user_id),
        )
    return [_row_to_dict(r) for r in rows]


async def delete_org(org_id: str) -> None:
    """Hard-delete an org (cascades to projects, members, memories, keys)."""
    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM orgs WHERE id = $1", uuid.UUID(org_id))


async def revoke_all_user_keys(user_id: str) -> int:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE user_id = $1 AND revoked_at IS NULL",
            uuid.UUID(user_id),
        )
    # result is "UPDATE n"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def list_audit(org_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT a.id, a.action, a.resource_type, a.resource_id, a.metadata,
                   a.ip_address, a.created_at,
                   u.email AS actor_email, k.name AS actor_key_name
            FROM audit_log a
            LEFT JOIN users u ON u.id = a.actor_user_id
            LEFT JOIN api_keys k ON k.id = a.actor_api_key_id
            WHERE a.org_id = $1
            ORDER BY a.created_at DESC
            LIMIT $2 OFFSET $3
            """,
            uuid.UUID(org_id),
            limit,
            offset,
        )
    out = []
    for r in rows:
        d = _row_to_dict(r)
        if d.get("metadata") and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (ValueError, TypeError):
                pass
        out.append(d)
    return out


# =============================================================================
# Memories (existing API, unchanged signatures)
# =============================================================================

async def save_memory(
    project_id: str,
    content: str,
    category: str,
    tags: list[str],
    embedding: list[float],
    pinned: bool = False,
    created_at: datetime | None = None,
) -> dict:
    """Insert a memory. created_at and pinned are optional and used when
    importing from a backup to preserve original timestamps + pin state.
    """
    async with _pool.acquire() as conn:
        if created_at is not None:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (project_id, content, category, tags, embedding, pinned, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                RETURNING id, project_id, content, category, tags, pinned, created_at, updated_at
                """,
                project_id, content, category, tags, embedding, pinned, created_at,
            )
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (project_id, content, category, tags, embedding, pinned)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, project_id, content, category, tags, pinned, created_at, updated_at
                """,
                project_id, content, category, tags, embedding, pinned,
            )
    return _row_to_dict(row)


async def get_memory(memory_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, project_id, content, category, tags, pinned, created_at, updated_at, deleted_at
            FROM memories WHERE id = $1
            """,
            uuid.UUID(memory_id),
        )
    return _row_to_dict(row) if row else None


async def update_memory(
    memory_id: str,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memories WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(memory_id),
        )
        if not row:
            return None

        new_content = content if content is not None else row["content"]
        new_category = category if category is not None else row["category"]
        new_tags = tags if tags is not None else row["tags"]
        new_embedding = embedding if embedding is not None else row["embedding"]
        now = datetime.now(timezone.utc)

        updated = await conn.fetchrow(
            """
            UPDATE memories
            SET content = $1, category = $2, tags = $3, embedding = $4, updated_at = $5
            WHERE id = $6 AND deleted_at IS NULL
            RETURNING id, project_id, content, category, tags, pinned, created_at, updated_at
            """,
            new_content,
            new_category,
            new_tags,
            new_embedding,
            now,
            uuid.UUID(memory_id),
        )
    return _row_to_dict(updated) if updated else None


async def delete_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE memories SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(memory_id),
        )
    return result == "UPDATE 1"


async def list_memories(
    project_id: str = "",
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
    offset: int = 0,
    project_ids: list[str] | None = None,
) -> list[dict]:
    """List memories. project_id (single) or project_ids (list) MUST be supplied
    by the caller — passing neither returns [] (fail-closed for ACL safety)."""
    if not project_id and project_ids is None:
        return []
    if project_ids is not None and not project_ids:
        return []
    query = "SELECT id, project_id, content, category, tags, pinned, created_at, updated_at FROM memories WHERE deleted_at IS NULL"
    params: list = []
    idx = 1

    if project_id:
        query += f" AND project_id = ${idx}"
        params.append(project_id)
        idx += 1
    elif project_ids is not None:
        query += f" AND project_id = ANY(${idx}::varchar[])"
        params.append(project_ids)
        idx += 1

    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1

    if tags:
        query += f" AND tags && ${idx}"
        params.append(tags)
        idx += 1

    query += f" ORDER BY pinned DESC, updated_at DESC LIMIT ${idx} OFFSET ${idx + 1}"
    params.extend([limit, offset])

    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [_row_to_dict(r) for r in rows]


async def search_memories(
    project_id: str = "",
    query_text: str = "",
    query_embedding: list[float] | None = None,
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 20,
    project_ids: list[str] | None = None,
) -> list[dict]:
    """Hybrid FTS + cosine search. Scoped by project_id OR project_ids list."""
    if project_id:
        scope_filter = "AND project_id = $2"
        scope_arg = [project_id]
        next_idx = 3
    elif project_ids is not None:
        if not project_ids:
            return []
        scope_filter = "AND project_id = ANY($2::varchar[])"
        scope_arg = [project_ids]
        next_idx = 3
    else:
        scope_filter = ""
        scope_arg = []
        next_idx = 2

    params: list = [query_text] + scope_arg
    embed_idx = next_idx
    params.append(query_embedding)
    limit_idx = embed_idx + 1
    params.append(limit)

    extra_filters = ""
    idx = limit_idx + 1
    if category:
        extra_filters += f" AND m.category = ${idx}"
        params.append(category)
        idx += 1
    if tags:
        extra_filters += f" AND m.tags && ${idx}"
        params.append(tags)
        idx += 1

    sql = f"""
    WITH fts AS (
        SELECT id,
               ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS text_rank
        FROM memories
        WHERE deleted_at IS NULL
          {scope_filter}
          AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
    ),
    sem AS (
        SELECT id,
               1 - (embedding <=> ${embed_idx}::vector) AS cosine_sim
        FROM memories
        WHERE deleted_at IS NULL
          {scope_filter}
    )
    SELECT m.id, m.project_id, m.content, m.category, m.tags, m.created_at, m.updated_at,
           COALESCE(f.text_rank, 0) * 0.4 + COALESCE(s.cosine_sim, 0) * 0.6 AS score
    FROM memories m
    LEFT JOIN fts f ON m.id = f.id
    LEFT JOIN sem s ON m.id = s.id
    WHERE m.deleted_at IS NULL
      {scope_filter}
      AND (f.id IS NOT NULL OR COALESCE(s.cosine_sim, 0) > 0.3)
      {extra_filters}
    ORDER BY score DESC
    LIMIT ${limit_idx}
    """

    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["score"] = float(r["score"])
        out.append(d)
    return out


# Stats / categories / tags — now scoped by accessible projects

async def get_categories(project_ids: list[str]) -> list[str]:
    if not project_ids:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT category FROM memories "
            "WHERE deleted_at IS NULL AND project_id = ANY($1::varchar[]) ORDER BY category",
            project_ids,
        )
    return [r["category"] for r in rows]


async def get_categories_with_counts(project_ids: list[str]) -> list[dict]:
    if not project_ids:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT project_id, category, COUNT(*) AS count
            FROM memories
            WHERE deleted_at IS NULL AND project_id = ANY($1::varchar[])
            GROUP BY project_id, category
            ORDER BY project_id, count DESC
            """,
            project_ids,
        )
    return [dict(r) for r in rows]


async def get_tags_with_counts(project_ids: list[str]) -> list[dict]:
    if not project_ids:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT project_id, tag, COUNT(*) AS count
            FROM memories, UNNEST(tags) AS tag
            WHERE deleted_at IS NULL AND project_id = ANY($1::varchar[])
            GROUP BY project_id, tag
            ORDER BY project_id, count DESC
            """,
            project_ids,
        )
    return [dict(r) for r in rows]


async def count_memories(project_id: str = "", category: str = "") -> int:
    query = "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
    params: list = []
    idx = 1
    if project_id:
        query += f" AND project_id = ${idx}"
        params.append(project_id)
        idx += 1
    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    async with _pool.acquire() as conn:
        return await conn.fetchval(query, *params)


async def count_archived(project_id: str = "") -> int:
    if project_id:
        query = "SELECT COUNT(*) FROM memories WHERE project_id = $1 AND deleted_at IS NOT NULL"
        params = [project_id]
    else:
        query = "SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL"
        params = []
    async with _pool.acquire() as conn:
        return await conn.fetchval(query, *params)


# Pin / related / export / archive — unchanged shape
async def pin_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE memories SET pinned = TRUE, updated_at = NOW() "
            "WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(memory_id),
        )
    return result == "UPDATE 1"


async def unpin_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE memories SET pinned = FALSE, updated_at = NOW() "
            "WHERE id = $1 AND deleted_at IS NULL",
            uuid.UUID(memory_id),
        )
    return result == "UPDATE 1"


async def get_related_memories(
    memory_id: str, project_id: str = "", limit: int = 5
) -> list[dict]:
    project_filter = "AND project_id = $2" if project_id else ""
    params: list = [uuid.UUID(memory_id)]
    if project_id:
        params.append(project_id)
    limit_idx = len(params) + 1
    params.append(limit)

    sql = f"""
    SELECT id, project_id, content, category, tags, pinned, updated_at,
           1 - (embedding <=> (SELECT embedding FROM memories WHERE id = $1)) AS similarity
    FROM memories
    WHERE id != $1 AND deleted_at IS NULL {project_filter}
    ORDER BY embedding <=> (SELECT embedding FROM memories WHERE id = $1)
    LIMIT ${limit_idx}
    """
    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["similarity"] = round(float(r["similarity"]), 3)
        out.append(d)
    return out


async def export_memories(project_ids: list[str]) -> list[dict]:
    if not project_ids:
        return []
    query = (
        "SELECT id, project_id, content, category, tags, pinned, created_at, updated_at "
        "FROM memories WHERE deleted_at IS NULL AND project_id = ANY($1::varchar[]) "
        "ORDER BY pinned DESC, updated_at DESC"
    )
    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, project_ids)
    return [_row_to_dict(r) for r in rows]


async def list_archived(
    project_ids: list[str], limit: int = 50, offset: int = 0
) -> list[dict]:
    if not project_ids:
        return []
    query = (
        "SELECT id, project_id, content, category, tags, created_at, updated_at, deleted_at "
        "FROM memories WHERE deleted_at IS NOT NULL AND project_id = ANY($1::varchar[]) "
        "ORDER BY deleted_at DESC LIMIT $2 OFFSET $3"
    )
    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, project_ids, limit, offset)
    return [_row_to_dict(r) for r in rows]


async def restore_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE memories SET deleted_at = NULL, updated_at = NOW() "
            "WHERE id = $1 AND deleted_at IS NOT NULL",
            uuid.UUID(memory_id),
        )
    return result == "UPDATE 1"


async def purge_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM memories WHERE id = $1 AND deleted_at IS NOT NULL",
            uuid.UUID(memory_id),
        )
    return result == "DELETE 1"
