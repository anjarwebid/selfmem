from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import asyncpg
from pgvector.asyncpg import register_vector

import config

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    tags TEXT[] DEFAULT '{{}}'::TEXT[],
    embedding vector({config.EMBEDDING_DIM}),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ DEFAULT NULL
);
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (to_tsvector('english', content))",
    "CREATE INDEX IF NOT EXISTS idx_memories_user_category ON memories (user_id, category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories (user_id) WHERE deleted_at IS NULL",
]


async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_db() -> None:
    global _pool
    # Create extension and schema before pool init, because pool's init
    # callback registers the vector type which requires the extension to exist.
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
    log.info("Database initialized")


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _row_to_dict(row: asyncpg.Record) -> dict:
    d = dict(row)
    for key in ("id",):
        if key in d and isinstance(d[key], uuid.UUID):
            d[key] = str(d[key])
    for key in ("created_at", "updated_at", "deleted_at"):
        if key in d and isinstance(d[key], datetime):
            d[key] = d[key].isoformat()
    if "embedding" in d:
        del d["embedding"]
    if "tags" in d and d["tags"] is None:
        d["tags"] = []
    return d


async def save_memory(
    user_id: str,
    content: str,
    category: str,
    tags: list[str],
    embedding: list[float],
) -> dict:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories (user_id, content, category, tags, embedding)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, user_id, content, category, tags, created_at, updated_at
            """,
            user_id,
            content,
            category,
            tags,
            embedding,
        )
    return _row_to_dict(row)


async def get_memory(memory_id: str) -> dict | None:
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, content, category, tags, created_at, updated_at, deleted_at
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
            RETURNING id, user_id, content, category, tags, created_at, updated_at
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
    user_id: str = "",
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    query = "SELECT id, user_id, content, category, tags, created_at, updated_at FROM memories WHERE deleted_at IS NULL"
    params: list = []
    idx = 1

    if user_id:
        query += f" AND user_id = ${idx}"
        params.append(user_id)
        idx += 1

    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1

    if tags:
        query += f" AND tags && ${idx}"
        params.append(tags)
        idx += 1

    query += f" ORDER BY updated_at DESC LIMIT ${idx} OFFSET ${idx + 1}"
    params.extend([limit, offset])

    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [_row_to_dict(r) for r in rows]


async def search_memories(
    user_id: str = "",
    query_text: str = "",
    query_embedding: list[float] | None = None,
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    user_filter = f"AND user_id = $2" if user_id else ""
    params: list = [query_text]
    if user_id:
        params.append(user_id)
    embed_idx = len(params) + 1
    params.append(query_embedding)
    limit_idx = len(params) + 1
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
          {user_filter}
          AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
    ),
    sem AS (
        SELECT id,
               1 - (embedding <=> ${embed_idx}::vector) AS cosine_sim
        FROM memories
        WHERE deleted_at IS NULL
          {user_filter}
    )
    SELECT m.id, m.user_id, m.content, m.category, m.tags, m.created_at, m.updated_at,
           COALESCE(f.text_rank, 0) * 0.4 + COALESCE(s.cosine_sim, 0) * 0.6 AS score
    FROM memories m
    LEFT JOIN fts f ON m.id = f.id
    LEFT JOIN sem s ON m.id = s.id
    WHERE m.deleted_at IS NULL
      {user_filter}
      AND (f.id IS NOT NULL OR COALESCE(s.cosine_sim, 0) > 0.3)
      {extra_filters}
    ORDER BY score DESC
    LIMIT ${limit_idx}
    """

    async with _pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    results = []
    for r in rows:
        d = _row_to_dict(r)
        d["score"] = float(r["score"])
        results.append(d)
    return results


# --- Stats functions ---


async def get_user_stats() -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT user_id,
                   COUNT(*) FILTER (WHERE deleted_at IS NULL) AS count,
                   COUNT(DISTINCT category) FILTER (WHERE deleted_at IS NULL) AS category_count,
                   COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) AS archived_count,
                   MAX(updated_at) AS latest_at
            FROM memories
            GROUP BY user_id
            ORDER BY user_id
            """
        )
    results = []
    for r in rows:
        d = dict(r)
        if d.get("latest_at") and isinstance(d["latest_at"], datetime):
            d["latest_at"] = d["latest_at"].isoformat()
        results.append(d)
    return results


async def get_categories(user_id: str = "") -> list[str]:
    if user_id:
        query = "SELECT DISTINCT category FROM memories WHERE user_id = $1 AND deleted_at IS NULL ORDER BY category"
        params = [user_id]
    else:
        query = "SELECT DISTINCT category FROM memories WHERE deleted_at IS NULL ORDER BY category"
        params = []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [r["category"] for r in rows]


async def get_categories_with_counts(user_id: str = "") -> list[dict]:
    query = """
        SELECT user_id, category, COUNT(*) AS count
        FROM memories
        WHERE deleted_at IS NULL
    """
    params = []
    if user_id:
        query += " AND user_id = $1"
        params.append(user_id)
    query += " GROUP BY user_id, category ORDER BY user_id, count DESC"
    async with _pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


async def count_memories(user_id: str = "", category: str = "") -> int:
    query = "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
    params: list = []
    idx = 1
    if user_id:
        query += f" AND user_id = ${idx}"
        params.append(user_id)
        idx += 1
    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    async with _pool.acquire() as conn:
        return await conn.fetchval(query, *params)


async def count_archived(user_id: str = "") -> int:
    if user_id:
        query = "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND deleted_at IS NOT NULL"
        params = [user_id]
    else:
        query = "SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL"
        params = []
    async with _pool.acquire() as conn:
        return await conn.fetchval(query, *params)


# --- Archive functions ---


async def list_archived(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, content, category, tags, created_at, updated_at, deleted_at
            FROM memories
            WHERE user_id = $1 AND deleted_at IS NOT NULL
            ORDER BY deleted_at DESC
            LIMIT $2 OFFSET $3
            """,
            user_id,
            limit,
            offset,
        )
    return [_row_to_dict(r) for r in rows]


async def restore_memory(memory_id: str) -> bool:
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE memories SET deleted_at = NULL, updated_at = NOW() WHERE id = $1 AND deleted_at IS NOT NULL",
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
