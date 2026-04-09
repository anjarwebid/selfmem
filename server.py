import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP

DATA_DIR = os.environ.get("LLM_MEMORY_DATA", "/data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
CONTEXTS_DIR = os.path.join(DATA_DIR, "contexts")

mcp = FastMCP("llm-memory", host="0.0.0.0", port=8818)


def get_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contexts (
            key TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS contexts_fts
        USING fts5(key, content, tags, category, content='contexts', content_rowid='rowid')
    """)
    # Triggers to keep FTS in sync
    for trigger_sql in [
        """CREATE TRIGGER IF NOT EXISTS contexts_ai AFTER INSERT ON contexts BEGIN
            INSERT INTO contexts_fts(rowid, key, content, tags, category)
            VALUES (new.rowid, new.key, new.content, new.tags, new.category);
        END""",
        """CREATE TRIGGER IF NOT EXISTS contexts_ad AFTER DELETE ON contexts BEGIN
            INSERT INTO contexts_fts(contexts_fts, rowid, key, content, tags, category)
            VALUES ('delete', old.rowid, old.key, old.content, old.tags, old.category);
        END""",
        """CREATE TRIGGER IF NOT EXISTS contexts_au AFTER UPDATE ON contexts BEGIN
            INSERT INTO contexts_fts(contexts_fts, rowid, key, content, tags, category)
            VALUES ('delete', old.rowid, old.key, old.content, old.tags, old.category);
            INSERT INTO contexts_fts(rowid, key, content, tags, category)
            VALUES (new.rowid, new.key, new.content, new.tags, new.category);
        END""",
    ]:
        conn.execute(trigger_sql)
    conn.commit()
    return conn


def save_markdown(key: str, content: str, tags: str, category: str):
    cat_dir = os.path.join(CONTEXTS_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    safe_key = key.replace("/", "_").replace(" ", "-")
    filepath = os.path.join(cat_dir, f"{safe_key}.md")
    with open(filepath, "w") as f:
        f.write(f"---\nkey: {key}\ntags: {tags}\ncategory: {category}\nupdated: {datetime.now(timezone.utc).isoformat()}\n---\n\n{content}\n")


def delete_markdown(key: str, category: str):
    safe_key = key.replace("/", "_").replace(" ", "-")
    filepath = os.path.join(CONTEXTS_DIR, category, f"{safe_key}.md")
    if os.path.exists(filepath):
        os.remove(filepath)


def row_to_dict(row):
    return dict(row) if row else None


@mcp.tool()
def save_context(key: str, content: str, tags: str = "", category: str = "general") -> str:
    """Save a knowledge entry. Key must be unique. Tags are comma-separated."""
    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    existing = db.execute("SELECT key FROM contexts WHERE key = ?", (key,)).fetchone()
    if existing:
        db.close()
        return f"Key '{key}' already exists. Use update_context to modify it."
    db.execute(
        "INSERT INTO contexts (key, content, tags, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (key, content, tags, category, now, now),
    )
    db.commit()
    db.close()
    save_markdown(key, content, tags, category)
    return f"Saved: {key} [{category}]"


@mcp.tool()
def get_context(key: str) -> str:
    """Retrieve a specific context entry by key."""
    db = get_db()
    row = db.execute("SELECT * FROM contexts WHERE key = ?", (key,)).fetchone()
    db.close()
    if not row:
        return f"Not found: {key}"
    return json.dumps(row_to_dict(row), indent=2)


@mcp.tool()
def search_context(query: str, tags: str = "", category: str = "") -> str:
    """Full-text search across all stored knowledge. Optionally filter by tags or category."""
    db = get_db()
    if query:
        sql = "SELECT c.* FROM contexts c JOIN contexts_fts f ON c.rowid = f.rowid WHERE contexts_fts MATCH ?"
        params = [query]
    else:
        sql = "SELECT * FROM contexts WHERE 1=1"
        params = []

    if tags:
        for tag in tags.split(","):
            sql += " AND c.tags LIKE ?" if query else " AND tags LIKE ?"
            params.append(f"%{tag.strip()}%")
    if category:
        sql += " AND c.category = ?" if query else " AND category = ?"
        params.append(category)

    sql += " LIMIT 50"
    rows = db.execute(sql, params).fetchall()
    db.close()
    if not rows:
        return "No results found."
    return json.dumps([row_to_dict(r) for r in rows], indent=2)


@mcp.tool()
def list_contexts(category: str = "", tags: str = "") -> str:
    """List all context entries. Optionally filter by category or tags."""
    db = get_db()
    sql = "SELECT key, tags, category, updated_at FROM contexts WHERE 1=1"
    params = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if tags:
        for tag in tags.split(","):
            sql += " AND tags LIKE ?"
            params.append(f"%{tag.strip()}%")
    sql += " ORDER BY updated_at DESC LIMIT 100"
    rows = db.execute(sql, params).fetchall()
    db.close()
    if not rows:
        return "No entries found."
    return json.dumps([row_to_dict(r) for r in rows], indent=2)


@mcp.tool()
def update_context(key: str, content: str, tags: str = "", category: str = "") -> str:
    """Update an existing context entry. Only provided fields are updated."""
    db = get_db()
    row = db.execute("SELECT * FROM contexts WHERE key = ?", (key,)).fetchone()
    if not row:
        db.close()
        return f"Not found: {key}"
    now = datetime.now(timezone.utc).isoformat()
    new_tags = tags if tags else row["tags"]
    new_category = category if category else row["category"]
    db.execute(
        "UPDATE contexts SET content = ?, tags = ?, category = ?, updated_at = ? WHERE key = ?",
        (content, new_tags, new_category, now, key),
    )
    db.commit()
    db.close()
    save_markdown(key, content, new_tags, new_category)
    return f"Updated: {key}"


@mcp.tool()
def delete_context(key: str) -> str:
    """Delete a context entry by key."""
    db = get_db()
    row = db.execute("SELECT category FROM contexts WHERE key = ?", (key,)).fetchone()
    if not row:
        db.close()
        return f"Not found: {key}"
    db.execute("DELETE FROM contexts WHERE key = ?", (key,))
    db.commit()
    db.close()
    delete_markdown(key, row["category"])
    return f"Deleted: {key}"


if __name__ == "__main__":
    mcp.run(transport="sse")
