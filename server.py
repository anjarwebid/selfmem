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


# --- MCP server ---

mcp = FastMCP(
    "selfmem",
    host="0.0.0.0",
    port=config.PORT,
    transport_security={
        "enable_dns_rebinding_protection": False,
    },
)


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    embeddings.get_embedding("warmup")
    # Initialize MCP session manager for streamable HTTP
    mcp_app = mcp.streamable_http_app()
    app.mount("/", mcp_app)
    async with mcp._session_manager.run():
        log.info("SelfMem ready on %s:%s", config.HOST, config.PORT)
        yield
    await db.close_db()


# --- FastAPI app ---

app = FastAPI(title="SelfMem", version="1.0.0", lifespan=lifespan)
app.add_middleware(APIKeyMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

PAGE_SIZE = 20


@mcp.tool()
async def save_memory(
    content: str,
    project_id: str,
    tags: list[str] | None = None,
    category: str = "general",
) -> str:
    """Save a memory to a project namespace. Content is auto-embedded for semantic search.

    project_id is the namespace that scopes memories (e.g. a Linux user, a project name).
    """
    embedding = embeddings.get_embedding(content)
    result = await db.save_memory(project_id, content, category, tags or [], embedding)
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_memory(
    query: str,
    project_id: str,
    tags: list[str] | None = None,
    category: str = "",
    limit: int = 20,
) -> str:
    """Search memories in a project namespace using hybrid full-text + semantic search."""
    query_embedding = embeddings.get_embedding(query)
    results = await db.search_memories(
        project_id, query, query_embedding, category, tags, limit
    )
    if not results:
        return "No memories found."
    return json.dumps(results, indent=2)


@mcp.tool()
async def list_memories(
    project_id: str,
    category: str = "",
    tags: list[str] | None = None,
    limit: int = 50,
) -> str:
    """List memories for a project, optionally filtered by category or tags."""
    results = await db.list_memories(project_id, category, tags, limit)
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


# --- REST API ---


class MemoryCreate(BaseModel):
    project_id: str
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
        body.project_id, body.content, body.category, body.tags, embedding
    )
    return JSONResponse(result, status_code=201)


@app.get("/api/v1/memories")
async def api_list_or_search_memories(
    project_id: str = Query(...),
    query: str = Query(""),
    category: str = Query(""),
    tags: list[str] = Query([]),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    if query:
        query_embedding = embeddings.get_embedding(query)
        results = await db.search_memories(
            project_id, query, query_embedding, category, tags or None, limit
        )
    else:
        results = await db.list_memories(
            project_id, category, tags or None, limit, offset
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
    project_id: str = Query(...),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    results = await db.list_archived(project_id, limit, offset)
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


# --- Pin API ---


@app.post("/api/v1/memories/{memory_id}/pin")
async def api_pin_memory(memory_id: str):
    ok = await db.pin_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"status": "pinned", "id": memory_id}


@app.delete("/api/v1/memories/{memory_id}/pin")
async def api_unpin_memory(memory_id: str):
    ok = await db.unpin_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"status": "unpinned", "id": memory_id}


# --- Export/Import API ---


@app.get("/api/v1/export")
async def api_export(project_id: str = Query("")):
    memories = await db.export_memories(project_id)
    # Strip internal fields for clean export
    export_data = []
    for m in memories:
        export_data.append({
            "content": m["content"],
            "project_id": m["project_id"],
            "category": m["category"],
            "tags": m["tags"],
            "pinned": m.get("pinned", False),
            "created_at": m["created_at"],
        })
    from fastapi.responses import StreamingResponse
    import io
    content = json.dumps(export_data, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=selfmem-export-{project_id or 'all'}.json"},
    )


@app.post("/api/v1/import")
async def api_import(request: Request):
    body = await request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    if not isinstance(data, list):
        return JSONResponse({"error": "Expected JSON array"}, status_code=400)

    imported = 0
    for item in data:
        content = item.get("content", "").strip()
        project_id = item.get("project_id", "").strip()
        if not content or not project_id:
            continue
        category = item.get("category", "general")
        tags = item.get("tags", [])
        embedding = embeddings.get_embedding(content)
        await db.save_memory(project_id, content, category, tags, embedding)
        imported += 1

    return {"status": "ok", "imported": imported, "total": len(data)}


# --- UI Routes ---


async def _get_current_project(request: Request) -> str:
    project = request.cookies.get("selfmem_project", "")
    if project == "":
        # Check if explicitly set to "" (All Projects) vs never set
        if "selfmem_project" in request.cookies:
            return "__all__"
        stats = await db.get_project_stats()
        project = stats[0]["project_id"] if stats else "default"
    return project


def _resolve_project(current_project: str) -> str:
    """Convert UI project to DB project_id. __all__ -> empty string for db queries."""
    return "" if current_project == "__all__" else current_project


async def _base_context(request: Request, active_page: str) -> dict:
    projects = await db.get_project_stats()
    current_project = await _get_current_project(request)
    return {
        "request": request,
        "projects": projects,
        "current_project": current_project,
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
    response.delete_cookie("selfmem_project")
    return response


@app.post("/ui/set-project")
async def ui_set_project(request: Request):
    form = await request.form()
    project_id = form.get("project_id", "")
    response = HTMLResponse("")
    response.set_cookie("selfmem_project", project_id, max_age=config.SESSION_MAX_AGE)
    return response


@app.get("/ui/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    ctx = await _base_context(request, "dashboard")
    db_project = _resolve_project(ctx["current_project"])
    mem_count = await db.count_memories(db_project)
    categories = await db.get_categories(db_project)
    archived = await db.count_archived(db_project) if db_project else 0
    # Find latest from project stats
    latest = ""
    for p in ctx["projects"]:
        if (not db_project or p["project_id"] == db_project) and p.get("latest_at"):
            if not latest or p["latest_at"] > latest:
                latest = p["latest_at"]
    ctx["stats"] = {
        "memories": mem_count,
        "categories": len(categories),
        "archived": archived,
        "latest": latest if latest else "",
    }
    ctx["memories"] = await db.list_memories(db_project, limit=10)
    return _render(request, "dashboard.html", ctx)


@app.get("/ui/memories", response_class=HTMLResponse)
async def ui_memories(
    request: Request,
    filter_category: str = Query(""),
    filter_tag: str = Query(""),
):
    ctx = await _base_context(request, "memories")
    db_project = _resolve_project(ctx["current_project"])
    ctx["categories"] = await db.get_categories(db_project)
    ctx["filter_category"] = filter_category
    ctx["filter_tag"] = filter_tag

    tags_filter = [filter_tag] if filter_tag else None
    ctx["total"] = await db.count_memories(db_project, filter_category)
    memories = await db.list_memories(db_project, category=filter_category, tags=tags_filter, limit=PAGE_SIZE)
    total_pages = max(1, math.ceil(ctx["total"] / PAGE_SIZE))
    ctx["memories"] = memories
    ctx["page"] = 1
    ctx["total_pages"] = total_pages
    return _render(request, "memories.html", ctx)


@app.get("/ui/partials/memories", response_class=HTMLResponse)
async def ui_partials_memories(
    request: Request,
    project_id: str = Query(""),
    query: str = Query(""),
    category: str = Query(""),
    page: int = Query(1),
):
    raw_project = project_id or await _get_current_project(request)
    db_project = _resolve_project(raw_project)
    offset = (page - 1) * PAGE_SIZE

    if query:
        query_embedding = embeddings.get_embedding(query)
        memories = await db.search_memories(db_project, query, query_embedding, category, limit=PAGE_SIZE)
        total = len(memories)
        total_pages = 1
    else:
        total = await db.count_memories(db_project, category)
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        memories = await db.list_memories(db_project, category=category, limit=PAGE_SIZE, offset=offset)

    return _render(request, "partials/memory_list.html", {
        "memories": memories,
        "page": page,
        "total_pages": total_pages,
        "current_project": raw_project,
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
    project_id = form.get("project_id", "")
    content = form.get("content", "")
    category = form.get("category", "") or "general"
    tags_str = form.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    embedding = embeddings.get_embedding(content)
    await db.save_memory(project_id, content, category, tags, embedding)

    # Re-render the full list
    memories = await db.list_memories(project_id, limit=PAGE_SIZE)
    total = await db.count_memories(project_id)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))

    response = _render(request, "partials/memory_list.html", {
        "memories": memories,
        "page": 1,
        "total_pages": total_pages,
        "current_project": project_id,
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


# --- UI Pin/Related ---


@app.post("/ui/memories/{memory_id}/pin", response_class=HTMLResponse)
async def ui_pin_memory(request: Request, memory_id: str):
    await db.pin_memory(memory_id)
    mem = await db.get_memory(memory_id)
    raw_project = await _get_current_project(request)
    response = _render(request, "partials/memory_row.html", {"mem": mem, "current_project": raw_project})
    response.headers.update(_toast_headers("Memory pinned"))
    return response


@app.post("/ui/memories/{memory_id}/unpin", response_class=HTMLResponse)
async def ui_unpin_memory(request: Request, memory_id: str):
    await db.unpin_memory(memory_id)
    mem = await db.get_memory(memory_id)
    raw_project = await _get_current_project(request)
    response = _render(request, "partials/memory_row.html", {"mem": mem, "current_project": raw_project})
    response.headers.update(_toast_headers("Memory unpinned"))
    return response


@app.get("/ui/partials/memory/{memory_id}/related", response_class=HTMLResponse)
async def ui_related_memories(request: Request, memory_id: str):
    mem = await db.get_memory(memory_id)
    project_id = mem["project_id"] if mem else ""
    related = await db.get_related_memories(memory_id, project_id, limit=5)
    return _render(request, "partials/related_list.html", {"memories": related, "source_id": memory_id})


# --- UI Export/Import ---


@app.get("/ui/export", response_class=HTMLResponse)
async def ui_export(request: Request):
    raw_project = await _get_current_project(request)
    db_project = _resolve_project(raw_project)
    memories = await db.export_memories(db_project)
    export_data = []
    for m in memories:
        export_data.append({
            "content": m["content"],
            "project_id": m["project_id"],
            "category": m["category"],
            "tags": m["tags"],
            "pinned": m.get("pinned", False),
            "created_at": m["created_at"],
        })
    import io
    from starlette.responses import StreamingResponse
    content = json.dumps(export_data, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=selfmem-export-{db_project or 'all'}.json"},
    )


@app.post("/ui/import", response_class=HTMLResponse)
async def ui_import(request: Request):
    form = await request.form()
    upload = form.get("file")
    if not upload:
        response = HTMLResponse("<div class='text-red-400 text-sm'>No file selected</div>")
        response.headers.update(_toast_headers("No file selected", "error"))
        return response

    content = await upload.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        response = HTMLResponse("<div class='text-red-400 text-sm'>Invalid JSON file</div>")
        response.headers.update(_toast_headers("Invalid JSON file", "error"))
        return response

    if not isinstance(data, list):
        response = HTMLResponse("<div class='text-red-400 text-sm'>Expected JSON array</div>")
        response.headers.update(_toast_headers("Expected JSON array", "error"))
        return response

    imported = 0
    for item in data:
        c = item.get("content", "").strip()
        pid = item.get("project_id", "").strip()
        if not c or not pid:
            continue
        embedding = embeddings.get_embedding(c)
        await db.save_memory(pid, c, item.get("category", "general"), item.get("tags", []), embedding)
        imported += 1

    response = HTMLResponse(f"<div class='text-emerald-400 text-sm'>{imported} memories imported</div>")
    response.headers.update(_toast_headers(f"{imported} memories imported"))
    return response


# --- UI Tags ---


@app.get("/ui/tags", response_class=HTMLResponse)
async def ui_tags(request: Request):
    ctx = await _base_context(request, "tags")
    db_project = _resolve_project(ctx["current_project"])
    rows = await db.get_tags_with_counts(db_project)

    tags_by_project: dict[str, list] = {}
    for row in rows:
        pid = row["project_id"]
        if pid not in tags_by_project:
            tags_by_project[pid] = []
        tags_by_project[pid].append(row)

    ctx["tags_by_project"] = tags_by_project
    return _render(request, "tags.html", ctx)


# --- UI Categories ---


@app.get("/ui/categories", response_class=HTMLResponse)
async def ui_categories(request: Request):
    ctx = await _base_context(request, "categories")
    db_project = _resolve_project(ctx["current_project"])
    rows = await db.get_categories_with_counts(db_project)

    categories_by_project: dict[str, list] = {}
    for row in rows:
        pid = row["project_id"]
        if pid not in categories_by_project:
            categories_by_project[pid] = []
        categories_by_project[pid].append(row)

    ctx["categories_by_project"] = categories_by_project
    return _render(request, "categories.html", ctx)


# --- UI Archive ---


@app.get("/ui/archive", response_class=HTMLResponse)
async def ui_archive(request: Request):
    ctx = await _base_context(request, "archive")
    db_project = _resolve_project(ctx["current_project"])
    ctx["memories"] = await db.list_archived(db_project)
    return _render(request, "archive.html", ctx)


@app.get("/ui/partials/archive", response_class=HTMLResponse)
async def ui_partials_archive(request: Request):
    raw_project = await _get_current_project(request)
    db_project = _resolve_project(raw_project)
    memories = await db.list_archived(db_project)
    return _render(request, "partials/archive_list.html", {"memories": memories, "current_project": raw_project})


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
        await db.count_memories(ctx["current_project"])
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
