from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import math

import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import config
import db
import embeddings
from auth import APIKeyMiddleware, create_session_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("selfmem")


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    embeddings.get_embedding("warmup")
    log.info("SelfMem ready on %s:%s", config.HOST, config.PORT)
    yield
    await db.close_db()


# --- FastAPI app ---

app = FastAPI(title="SelfMem", version="1.0.0", lifespan=lifespan)
app.add_middleware(APIKeyMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

PAGE_SIZE = 20


# --- MCP server ---

mcp = FastMCP("selfmem")


@mcp.tool()
async def save_memory(
    content: str,
    user_id: str,
    tags: list[str] | None = None,
    category: str = "general",
) -> str:
    """Save a memory. Content is auto-embedded for semantic search."""
    embedding = embeddings.get_embedding(content)
    result = await db.save_memory(user_id, content, category, tags or [], embedding)
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_memory(
    query: str,
    user_id: str,
    tags: list[str] | None = None,
    category: str = "",
    limit: int = 20,
) -> str:
    """Search memories using hybrid full-text + semantic search."""
    query_embedding = embeddings.get_embedding(query)
    results = await db.search_memories(
        user_id, query, query_embedding, category, tags, limit
    )
    if not results:
        return "No memories found."
    return json.dumps(results, indent=2)


@mcp.tool()
async def list_memories(
    user_id: str,
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
) -> str:
    """List memories for a user, optionally filtered by category or tags."""
    results = await db.list_memories(user_id, category, tags, limit)
    if not results:
        return "No memories found."
    return json.dumps(results, indent=2)


@mcp.tool()
async def get_memory(id: str) -> str:
    """Get a specific memory by ID."""
    result = await db.get_memory(id)
    if not result:
        return f"Memory not found: {id}"
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_memory(
    id: str,
    content: str = "",
    tags: list[str] | None = None,
    category: str = "",
) -> str:
    """Update an existing memory. Re-embeds if content changes."""
    embedding = embeddings.get_embedding(content) if content else None
    result = await db.update_memory(
        id,
        content=content or None,
        category=category or None,
        tags=tags,
        embedding=embedding,
    )
    if not result:
        return f"Memory not found or archived: {id}"
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_memory(id: str) -> str:
    """Soft-delete a memory (moves to archive). Use the REST API to restore or purge."""
    ok = await db.delete_memory(id)
    if not ok:
        return f"Memory not found or already archived: {id}"
    return f"Archived: {id}"


# Mount MCP at /mcp
app.mount("/", mcp.streamable_http_app())


# --- REST API ---


class MemoryCreate(BaseModel):
    user_id: str
    content: str
    category: str = "general"
    tags: list[str] = []


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "name": "selfmem"}


@app.post("/api/v1/memories")
async def api_save_memory(body: MemoryCreate):
    embedding = embeddings.get_embedding(body.content)
    result = await db.save_memory(
        body.user_id, body.content, body.category, body.tags, embedding
    )
    return JSONResponse(result, status_code=201)


@app.get("/api/v1/memories")
async def api_list_or_search_memories(
    user_id: str = Query(...),
    query: str = Query(""),
    category: str = Query(""),
    tags: list[str] = Query([]),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    if query:
        query_embedding = embeddings.get_embedding(query)
        results = await db.search_memories(
            user_id, query, query_embedding, category, tags or None, limit
        )
    else:
        results = await db.list_memories(
            user_id, category, tags or None, limit, offset
        )
    return {"items": results, "total": len(results)}


@app.get("/api/v1/memories/{memory_id}")
async def api_get_memory(memory_id: str):
    result = await db.get_memory(memory_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return result


@app.put("/api/v1/memories/{memory_id}")
async def api_update_memory(memory_id: str, body: MemoryUpdate):
    embedding = embeddings.get_embedding(body.content) if body.content else None
    result = await db.update_memory(
        memory_id,
        content=body.content,
        category=body.category,
        tags=body.tags,
        embedding=embedding,
    )
    if not result:
        return JSONResponse({"error": "Not found or archived"}, status_code=404)
    return result


@app.delete("/api/v1/memories/{memory_id}")
async def api_delete_memory(memory_id: str):
    ok = await db.delete_memory(memory_id)
    if not ok:
        return JSONResponse(
            {"error": "Not found or already archived"}, status_code=404
        )
    return {"status": "archived", "id": memory_id}


# --- Archive API ---


@app.get("/api/v1/archive")
async def api_list_archived(
    user_id: str = Query(...),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    results = await db.list_archived(user_id, limit, offset)
    return {"items": results, "total": len(results)}


@app.post("/api/v1/archive/{memory_id}/restore")
async def api_restore_memory(memory_id: str):
    ok = await db.restore_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found or not archived"}, status_code=404)
    return {"status": "restored", "id": memory_id}


@app.delete("/api/v1/archive/{memory_id}")
async def api_purge_memory(memory_id: str):
    ok = await db.purge_memory(memory_id)
    if not ok:
        return JSONResponse(
            {"error": "Not found or not archived (must soft-delete first)"},
            status_code=404,
        )
    return {"status": "purged", "id": memory_id}


# --- UI Routes ---


async def _get_current_user(request: Request) -> str:
    user = request.cookies.get("selfmem_user", "")
    if not user:
        stats = await db.get_user_stats()
        user = stats[0]["user_id"] if stats else "default"
    return user


async def _base_context(request: Request, active_page: str) -> dict:
    users = await db.get_user_stats()
    current_user = await _get_current_user(request)
    return {
        "request": request,
        "users": users,
        "current_user": current_user,
        "active_page": active_page,
    }


def _toast_headers(message: str, type: str = "success") -> dict:
    return {"HX-Trigger": json.dumps({"showToast": {"message": message, "type": type}})}


def _render(request: Request, name: str, ctx: dict, **kwargs):
    ctx.pop("request", None)
    return templates.TemplateResponse(request, name, ctx, **kwargs)


@app.get("/ui/login", response_class=HTMLResponse)
async def ui_login(request: Request):
    return _render(request, "login.html", {"error": ""})


@app.post("/ui/login")
async def ui_login_post(request: Request, api_key: str = Form(...)):
    if config.API_KEYS and api_key not in config.API_KEYS:
        return _render(request, "login.html", {"error": "Invalid API key"})
    response = RedirectResponse("/ui/", status_code=302)
    token = create_session_token(api_key)
    response.set_cookie("selfmem_session", token, httponly=True, max_age=config.SESSION_MAX_AGE)
    return response


@app.get("/ui/logout")
async def ui_logout():
    response = RedirectResponse("/ui/login", status_code=302)
    response.delete_cookie("selfmem_session")
    response.delete_cookie("selfmem_user")
    return response


@app.post("/ui/set-user")
async def ui_set_user(request: Request):
    form = await request.form()
    user_id = form.get("user_id", "")
    response = HTMLResponse("")
    response.set_cookie("selfmem_user", user_id, max_age=config.SESSION_MAX_AGE)
    return response


@app.get("/ui/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    ctx = await _base_context(request, "dashboard")
    user = ctx["current_user"]
    mem_count = await db.count_memories(user)
    categories = await db.get_categories(user)
    archived = await db.count_archived(user)
    # Find latest from user stats
    latest = ""
    for u in ctx["users"]:
        if u["user_id"] == user and u.get("latest_at"):
            latest = u["latest_at"][:16]
            break
    ctx["stats"] = {
        "memories": mem_count,
        "categories": len(categories),
        "archived": archived,
        "latest": latest,
    }
    ctx["memories"] = await db.list_memories(user, limit=10)
    return _render(request, "dashboard.html", ctx)


@app.get("/ui/memories", response_class=HTMLResponse)
async def ui_memories(request: Request):
    ctx = await _base_context(request, "memories")
    user = ctx["current_user"]
    ctx["categories"] = await db.get_categories(user)
    ctx["total"] = await db.count_memories(user)
    memories = await db.list_memories(user, limit=PAGE_SIZE)
    total_pages = max(1, math.ceil(ctx["total"] / PAGE_SIZE))
    ctx["memories"] = memories
    ctx["page"] = 1
    ctx["total_pages"] = total_pages
    return _render(request, "memories.html", ctx)


@app.get("/ui/partials/memories", response_class=HTMLResponse)
async def ui_partials_memories(
    request: Request,
    user_id: str = Query(""),
    query: str = Query(""),
    category: str = Query(""),
    page: int = Query(1),
):
    user = user_id or await _get_current_user(request)
    offset = (page - 1) * PAGE_SIZE

    if query:
        query_embedding = embeddings.get_embedding(query)
        memories = await db.search_memories(user, query, query_embedding, category, limit=PAGE_SIZE)
        total = len(memories)
        total_pages = 1
    else:
        total = await db.count_memories(user, category)
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        memories = await db.list_memories(user, category=category, limit=PAGE_SIZE, offset=offset)

    return _render(request, "partials/memory_list.html", {
        "memories": memories,
        "page": page,
        "total_pages": total_pages,
        "current_user": user,
    })


@app.get("/ui/partials/memory/{memory_id}", response_class=HTMLResponse)
async def ui_partials_memory(request: Request, memory_id: str):
    mem = await db.get_memory(memory_id)
    return _render(request, "partials/memory_row.html", {"mem": mem})


@app.get("/ui/partials/memory/{memory_id}/edit", response_class=HTMLResponse)
async def ui_partials_memory_edit(request: Request, memory_id: str):
    mem = await db.get_memory(memory_id)
    return _render(request, "partials/memory_form.html", {"mem": mem})


@app.post("/ui/memories", response_class=HTMLResponse)
async def ui_create_memory(request: Request):
    form = await request.form()
    user_id = form.get("user_id", "")
    content = form.get("content", "")
    category = form.get("category", "") or "general"
    tags_str = form.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    embedding = embeddings.get_embedding(content)
    await db.save_memory(user_id, content, category, tags, embedding)

    # Re-render the full list
    memories = await db.list_memories(user_id, limit=PAGE_SIZE)
    total = await db.count_memories(user_id)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))

    response = _render(request, "partials/memory_list.html", {
        "memories": memories,
        "page": 1,
        "total_pages": total_pages,
        "current_user": user_id,
    })
    response.headers.update(_toast_headers("Memory saved"))
    return response


@app.put("/ui/memories/{memory_id}", response_class=HTMLResponse)
async def ui_update_memory(request: Request, memory_id: str):
    form = await request.form()
    content = form.get("content", "")
    category = form.get("category", "")
    tags_str = form.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    emb = embeddings.get_embedding(content) if content else None
    await db.update_memory(memory_id, content=content or None, category=category or None, tags=tags, embedding=emb)

    mem = await db.get_memory(memory_id)
    response = _render(request, "partials/memory_row.html", {"mem": mem})
    response.headers.update(_toast_headers("Memory updated"))
    return response


@app.delete("/ui/memories/{memory_id}", response_class=HTMLResponse)
async def ui_delete_memory(request: Request, memory_id: str):
    await db.delete_memory(memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory archived"))
    return response


# --- UI Archive ---


@app.get("/ui/archive", response_class=HTMLResponse)
async def ui_archive(request: Request):
    ctx = await _base_context(request, "archive")
    user = ctx["current_user"]
    ctx["memories"] = await db.list_archived(user)
    return _render(request, "archive.html", ctx)


@app.get("/ui/partials/archive", response_class=HTMLResponse)
async def ui_partials_archive(request: Request):
    user = await _get_current_user(request)
    memories = await db.list_archived(user)
    return _render(request, "partials/archive_list.html", {"memories": memories})


@app.post("/ui/archive/{memory_id}/restore", response_class=HTMLResponse)
async def ui_restore_memory(request: Request, memory_id: str):
    await db.restore_memory(memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory restored"))
    return response


@app.delete("/ui/archive/{memory_id}", response_class=HTMLResponse)
async def ui_purge_memory_ui(request: Request, memory_id: str):
    await db.purge_memory(memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory permanently deleted"))
    return response


# --- UI Settings ---


@app.get("/ui/settings", response_class=HTMLResponse)
async def ui_settings(request: Request):
    ctx = await _base_context(request, "settings")
    # Check DB connectivity
    try:
        await db.count_memories(ctx["current_user"])
        db_ok = True
    except Exception:
        db_ok = False

    # Mask API key
    api_key_hint = ""
    if config.API_KEYS:
        first_key = next(iter(config.API_KEYS))
        api_key_hint = first_key[-4:] if len(first_key) >= 4 else "****"

    ctx.update({
        "db_ok": db_ok,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "api_key_hint": api_key_hint,
    })
    return _render(request, "settings.html", ctx)


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
