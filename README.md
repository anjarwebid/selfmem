# SelfMem

Self-hosted personal memory server for LLM agents and AI assistants.

SelfMem provides a unified memory layer accessible via MCP (Model Context Protocol), REST API, and a web UI. It stores memories with hybrid full-text + semantic search, backed by PostgreSQL with pgvector. All embeddings are computed locally — no external API keys required.

## Why SelfMem?

Existing solutions like OpenMemory scope memories per MCP client session (app-level ACL), making memories created by one client invisible to another. SelfMem fixes this by routing memories by `project_id` — any client with a valid API key sees the same memories for the same project namespace.

## Features

- **MCP server** — HTTP streamable transport, works with Claude Code, Claude Desktop, and any MCP client
- **REST API** — Full CRUD at `/api/v1/memories` and `/api/v1/archive`
- **Web UI** — Dark-themed management dashboard at `/ui/` with search, filter, create, edit, delete, archive/restore
- **Hybrid search** — Combined full-text (PostgreSQL tsvector) + semantic (pgvector cosine similarity) search
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs inside the container, no OpenAI dependency
- **API key auth** — Single `X-API-Key` header protects all endpoints (MCP, REST, UI)
- **Soft-delete with archive** — Deleted memories are archived, restorable, and purgeable separately
- **Multi-project** — `project_id` is a parameter, not derived from client session. (Future: real authenticated users that own multiple projects.)
- **Single container** — One Docker Compose stack: PostgreSQL + pgvector + SelfMem app

## Quick Start

```bash
# Clone and configure
git clone <repo-url> && cd selfmem
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and SELFMEM_API_KEYS

# Start
docker compose up -d --build

# Verify
curl http://localhost:8818/health
```

Open the web UI at `http://localhost:8818/ui/login` and enter your API key.

## Configuration

All configuration is via environment variables (set in `.env`):

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_PASSWORD` | `changeme` | PostgreSQL password |
| `SELFMEM_API_KEYS` | _(empty)_ | Comma-separated API keys. If empty, all requests are allowed |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `EMBEDDING_DIM` | `384` | Embedding vector dimensions (must match model) |
| `SELFMEM_PORT` | `8818` | Server port |
| `SELFMEM_SESSION_SECRET` | `change-me-in-production` | Secret for signing UI session cookies |

## Connecting Claude Code

```bash
claude mcp add selfmem \
  --transport http \
  --header "X-API-Key: YOUR_API_KEY" \
  http://localhost:8818/mcp
```

## MCP Tools

| Tool | Description |
|---|---|
| `save_memory` | Save a memory with auto-embedding |
| `search_memory` | Hybrid full-text + semantic search |
| `list_memories` | List memories with optional category/tag filters |
| `get_memory` | Get a specific memory by ID |
| `update_memory` | Update content, tags, or category (re-embeds on content change) |
| `delete_memory` | Soft-delete (archive) a memory |

All tools accept `project_id` as a parameter (the namespace that scopes memories — e.g. a Linux user, a project name).

## REST API

All endpoints require `X-API-Key` header (except `/health`).

### Memories

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/memories` | Create a memory |
| `GET` | `/api/v1/memories?project_id=...` | List or search (add `&query=...` for search) |
| `GET` | `/api/v1/memories/{id}` | Get by ID |
| `PUT` | `/api/v1/memories/{id}` | Update |
| `DELETE` | `/api/v1/memories/{id}` | Soft-delete (archive) |

### Archive

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/archive?project_id=...` | List archived memories |
| `POST` | `/api/v1/archive/{id}/restore` | Restore from archive |
| `DELETE` | `/api/v1/archive/{id}` | Permanently delete (purge) |

### Examples

```bash
API_KEY="your-key"

# Save a memory
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"project_id":"slvr","content":"PostgreSQL uses MVCC for concurrency","category":"technical","tags":["postgres","database"]}' \
  http://localhost:8818/api/v1/memories

# Search
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8818/api/v1/memories?project_id=slvr&query=database+concurrency"

# Delete (soft)
curl -X DELETE -H "X-API-Key: $API_KEY" \
  http://localhost:8818/api/v1/memories/{id}

# Restore from archive
curl -X POST -H "X-API-Key: $API_KEY" \
  http://localhost:8818/api/v1/archive/{id}/restore
```

## Web UI

Dark-themed management interface at `/ui/`:

- **Dashboard** — Stats overview, quick search, recent memories
- **Memories** — Search, filter by category, add/edit/delete with HTMX (no page reloads)
- **Archive** — View soft-deleted memories, restore or purge
- **Settings** — Server info, DB status, API key (masked), project list
- **Project switcher** — Switch between projects from the sidebar

## Architecture

```
Client (Claude Code / Browser / any HTTP client)
    |
    v
+--------------------------------------+
|  FastAPI + FastMCP (port 8818)       |
|  |- /mcp          MCP (HTTP)         |
|  |- /api/v1/*     REST API           |
|  |- /ui/*         Web UI (Jinja2)    |
|  |- /health       Healthcheck        |
|  Auth: X-API-Key header              |
+--------------------------------------+
    |
    v
+--------------------------------------+
|  PostgreSQL 17 + pgvector            |
|  |- Full-text search (tsvector/GIN)  |
|  |- Vector search (HNSW index)       |
|  |- Hybrid scoring (0.4 FTS + 0.6   |
|     cosine similarity)               |
+--------------------------------------+

Embeddings: sentence-transformers
            all-MiniLM-L6-v2 (384d)
            Runs locally inside container
```

## Project Structure

```
.
├── server.py           # FastAPI + FastMCP: MCP tools, REST API, UI routes
├── db.py               # asyncpg + pgvector: schema, queries, hybrid search
├── embeddings.py       # sentence-transformers lazy-load wrapper
├── auth.py             # API key middleware (header + session cookie)
├── config.py           # Environment-based configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── templates/          # Jinja2 templates (dark theme)
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── memories.html
│   ├── archive.html
│   ├── settings.html
│   └── partials/       # HTMX partial templates
└── static/
    └── style.css
```

## Importing from OpenMemory

A migration script is included to import all memories from an existing OpenMemory instance:

```bash
python import_openmemory.py YOUR_SELFMEM_API_KEY
```

Edit `OPENMEMORY_BASE` in the script to point to your OpenMemory instance.

## License

MIT
