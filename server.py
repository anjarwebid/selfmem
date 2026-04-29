from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import audit
import billing
import config
import db
import embeddings
import gdpr
import passwords
import quotas
import ratelimit
from auth import (
    APIKeyMiddleware,
    create_session_token,
    is_org_owner,
)
from quotas import QuotaExceeded
from tokens import generate_api_key, generate_token

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
    transport_security={"enable_dns_rebinding_protection": False},
)


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    embeddings.get_embedding("warmup")
    mcp_app = mcp.streamable_http_app()
    app.mount("/", mcp_app)
    async with mcp._session_manager.run():
        log.info("SelfMem ready on %s:%s (mode=%s)",
                 config.HOST, config.PORT, config.HOSTING_MODE)
        yield
    await db.close_db()


app = FastAPI(title="SelfMem", version="2.0.0", lifespan=lifespan)
app.add_middleware(APIKeyMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

PAGE_SIZE = 20

# --- Helpers ---


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _render(request: Request, name: str, ctx: dict, **kwargs):
    ctx.pop("request", None)
    return templates.TemplateResponse(request, name, ctx, **kwargs)


def _toast_headers(message: str, type: str = "success") -> dict:
    return {"HX-Trigger": json.dumps({"showToast": {"message": message, "type": type}})}


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,98}[a-z0-9]$")


def _parse_iso(value):
    """Parse an ISO-8601 string into a tz-aware datetime; None if blank/bad."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Python 3.11+ fromisoformat handles 'Z' and offsets natively
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _is_valid_slug(s: str) -> bool:
    return bool(SLUG_RE.match(s))


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:100] or "user"


def _quota_response(e: QuotaExceeded) -> JSONResponse:
    return JSONResponse(
        {"error": "quota_exceeded", "resource": e.resource, "limit": e.limit,
         "upgrade_url": f"/ui/orgs/{e.org_slug}/billing" if e.org_slug else None},
        status_code=402,
    )


async def _resolve_project_org(project_id: str) -> dict | None:
    """Look up a project's parent org (for audit/quota writes)."""
    p = await db.get_project(project_id)
    if not p:
        return None
    return await db.get_org_by_id(p["org_id"])


async def _api_quota_check(request: Request, project_id: str) -> Response | None:
    """API/MCP path: increment per-org daily counter and enforce limit. SaaS-only."""
    if not config.IS_SAAS:
        return None
    org = await _resolve_project_org(project_id)
    if not org:
        return None
    try:
        await quotas.check_api_call_quota(org["id"], org["plan_tier"], org["slug"])
    except QuotaExceeded as e:
        return _quota_response(e)
    return None


def _get_current_project_cookie(request: Request) -> str:
    """Read the selfmem_project cookie. '__all__' means 'all accessible'."""
    val = request.cookies.get("selfmem_project", "")
    if val == "" and "selfmem_project" in request.cookies:
        return "__all__"
    return val


async def _resolve_current_project(request: Request) -> str:
    """Pick the active project for the UI: cookie first, else first accessible, else '__all__'."""
    auth = request.state.auth
    cookie_val = _get_current_project_cookie(request)
    if cookie_val == "__all__":
        return "__all__"
    if cookie_val and cookie_val in auth.allowed_project_ids:
        return cookie_val
    # No cookie or stale value — pick first accessible
    if auth.allowed_project_ids:
        return sorted(auth.allowed_project_ids)[0]
    return "__all__"


async def _base_context(request: Request, active_page: str) -> dict:
    auth = request.state.auth
    accessible = await db.list_user_accessible_projects(auth.user.id)
    current_project = await _resolve_current_project(request)
    return {
        "request": request,
        "auth_user": {"id": auth.user.id, "email": auth.user.email},
        "accessible_projects": accessible,
        "current_project": current_project,
        "active_page": active_page,
        "is_saas": config.IS_SAAS,
    }


def _resolve_db_scope(current_project: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Convert UI selection to DB scope: returns (project_id, project_ids).
    Always one is set (the other is "" / empty list).

    - current_project = "__all__" → ("", sorted(allowed)). If allowed is empty,
      project_ids is [] which the DB layer treats as 'no rows' (NOT unscoped).
    - current_project = "<slug>" → ("<slug>", []). Caller must ACL-check first.
    - current_project = "" with non-empty allowed → falls back to all allowed.
    """
    if current_project == "__all__" or current_project == "":
        return "", sorted(allowed)
    return current_project, []


# =============================================================================
# MCP tools (ACL-checked)
# =============================================================================

def _mcp_auth_check(project_id: str) -> dict | None:
    """MCP tools run inside the FastMCP app, but our APIKeyMiddleware has already
    validated the X-API-Key header and set request.state.auth on the underlying
    Starlette request. FastMCP doesn't pass that through; instead we re-validate
    at the tool-call layer using the contextvar that FastMCP exposes.
    For now we trust that the middleware ran and the API key was valid; the
    project ACL check happens via a thin DB hop.
    """
    # NOTE: at the FastMCP layer we do not have direct access to request.state.
    # The APIKeyMiddleware blocks unauthorized callers entirely (returns 401
    # before MCP sees the request). The project ACL check therefore has to
    # happen via the API key resolved by the middleware AND re-validated here.
    # Since the middleware already populated state.auth on the underlying ASGI
    # scope, but FastMCP creates a new task without forwarding it, we rely on
    # the alternative: every MCP call must include an X-API-Key, the middleware
    # validates it and would have rejected it otherwise. The remaining concern
    # is "did this key have ACL for this project?" — for that we look up the key
    # again here. To keep latency down we cache nothing for now.
    return None


@mcp.tool()
async def save_memory(
    content: str,
    project_id: str,
    tags: list[str] | None = None,
    category: str = "general",
) -> str:
    """Save a memory to a project. Project must be in your API key's ACL.

    project_id is the namespace that scopes memories.
    """
    if not await _mcp_acl_allows(project_id):
        return f"Forbidden: project '{project_id}' is not in your API key's ACL"
    org = await _resolve_project_org(project_id)
    if not org:
        return f"Unknown project: {project_id}"
    if config.IS_SAAS:
        try:
            await quotas.check_memory_quota(org["id"], org["plan_tier"], org["slug"])
        except QuotaExceeded as e:
            return json.dumps({"error": "quota_exceeded", "resource": e.resource, "limit": e.limit})
    embedding = embeddings.get_embedding(content)
    result = await db.save_memory(project_id, content, category, tags or [], embedding)
    await db.increment_memory_count(org["id"], 1)
    audit.log_event(org["id"], "memory.save", resource_type="memory",
                    resource_id=result["id"], metadata={"project_id": project_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_memory(
    query: str,
    project_id: str,
    tags: list[str] | None = None,
    category: str = "",
    limit: int = 20,
) -> str:
    """Search memories in a project (hybrid full-text + semantic). ACL-gated."""
    if not await _mcp_acl_allows(project_id):
        return f"Forbidden: project '{project_id}' is not in your API key's ACL"
    query_embedding = embeddings.get_embedding(query)
    results = await db.search_memories(project_id, query, query_embedding, category, tags, limit)
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
    """List memories for a project. ACL-gated."""
    if not await _mcp_acl_allows(project_id):
        return f"Forbidden: project '{project_id}' is not in your API key's ACL"
    results = await db.list_memories(project_id, category, tags, limit)
    if not results:
        return "No memories found."
    return json.dumps(results, indent=2)


@mcp.tool()
async def get_memory(id: str) -> str:
    """Get a specific memory by ID. ACL-gated."""
    result = await db.get_memory(id)
    if not result:
        return f"Memory not found: {id}"
    if not await _mcp_acl_allows(result["project_id"]):
        return f"Forbidden: project '{result['project_id']}' is not in your API key's ACL"
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_memory(
    id: str,
    content: str = "",
    tags: list[str] | None = None,
    category: str = "",
) -> str:
    """Update an existing memory. ACL-gated. Re-embeds if content changes."""
    existing = await db.get_memory(id)
    if not existing:
        return f"Memory not found: {id}"
    if not await _mcp_acl_allows(existing["project_id"]):
        return f"Forbidden: project '{existing['project_id']}' is not in your API key's ACL"
    embedding = embeddings.get_embedding(content) if content else None
    result = await db.update_memory(
        id, content=content or None, category=category or None,
        tags=tags, embedding=embedding,
    )
    if not result:
        return f"Memory not found or archived: {id}"
    org = await _resolve_project_org(existing["project_id"])
    if org:
        audit.log_event(org["id"], "memory.update", resource_type="memory", resource_id=id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_memory(id: str) -> str:
    """Soft-delete a memory (moves to archive). ACL-gated."""
    existing = await db.get_memory(id)
    if not existing:
        return f"Memory not found: {id}"
    if not await _mcp_acl_allows(existing["project_id"]):
        return f"Forbidden: project '{existing['project_id']}' is not in your API key's ACL"
    ok = await db.delete_memory(id)
    if not ok:
        return f"Memory not found or already archived: {id}"
    org = await _resolve_project_org(existing["project_id"])
    if org:
        audit.log_event(org["id"], "memory.delete", resource_type="memory", resource_id=id)
        await db.increment_memory_count(org["id"], -1)
    return f"Archived: {id}"


# MCP layer can't easily access request.state, so the ACL check goes through the
# raw header captured by middleware and stashed in a contextvar. To keep things
# simple in phase 1 we re-validate by re-hashing the header on every tool call.
# FastMCP exposes the underlying request through `mcp.get_context()`; we use it
# defensively but fall back to a permissive check if context isn't set (which
# would be a bug — middleware should always have run first).

import contextvars  # noqa: E402

_current_request_ctx: contextvars.ContextVar[Request | None] = contextvars.ContextVar(
    "_current_request_ctx", default=None
)


async def _mcp_acl_allows(project_id: str) -> bool:
    """Check the active request's auth context for ACL access to project_id."""
    req = _current_request_ctx.get()
    if req is None:
        # No request context (e.g. test harness) — fail closed
        return False
    auth = getattr(req.state, "auth", None)
    if not auth:
        return False
    return project_id in auth.allowed_project_ids


# Wire the contextvar from the ASGI middleware so MCP tools can read it.
class _RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = _current_request_ctx.set(request)
        try:
            return await call_next(request)
        finally:
            _current_request_ctx.reset(token)


app.add_middleware(_RequestContextMiddleware)


# =============================================================================
# REST API
# =============================================================================

class MemoryCreate(BaseModel):
    project_id: str
    content: str
    category: str = "general"
    tags: list[str] = []


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


class ProjectCreate(BaseModel):
    project_id: str          # slug
    org_slug: Optional[str] = None  # defaults to user's personal org
    name: Optional[str] = None


class KeyCreate(BaseModel):
    name: str
    project_ids: list[str]


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "name": "selfmem", "mode": config.HOSTING_MODE}


def _require_acl(request: Request, project_id: str) -> JSONResponse | None:
    auth = request.state.auth
    if project_id not in auth.allowed_project_ids:
        return JSONResponse({"error": "Forbidden", "project_id": project_id}, status_code=403)
    return None


@app.post("/api/v1/memories")
async def api_save_memory(request: Request, body: MemoryCreate):
    if (err := _require_acl(request, body.project_id)):
        return err
    if (err := await _api_quota_check(request, body.project_id)):
        return err
    org = await _resolve_project_org(body.project_id)
    if not org:
        return JSONResponse({"error": "Unknown project"}, status_code=404)
    if config.IS_SAAS:
        try:
            await quotas.check_memory_quota(org["id"], org["plan_tier"], org["slug"])
        except QuotaExceeded as e:
            return _quota_response(e)
    embedding = embeddings.get_embedding(body.content)
    result = await db.save_memory(body.project_id, body.content, body.category, body.tags, embedding)
    await db.increment_memory_count(org["id"], 1)
    audit.log_event(org["id"], "memory.save",
                    actor_user_id=request.state.auth.user.id,
                    actor_api_key_id=request.state.auth.api_key_id,
                    resource_type="memory", resource_id=result["id"],
                    ip_address=_client_ip(request))
    return JSONResponse(result, status_code=201)


@app.get("/api/v1/memories")
async def api_list_or_search_memories(
    request: Request,
    project_id: str = Query(...),
    query: str = Query(""),
    category: str = Query(""),
    tags: list[str] = Query([]),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    if (err := _require_acl(request, project_id)):
        return err
    if (err := await _api_quota_check(request, project_id)):
        return err
    if query:
        emb = embeddings.get_embedding(query)
        results = await db.search_memories(project_id, query, emb, category, tags or None, limit)
    else:
        results = await db.list_memories(project_id, category, tags or None, limit, offset)
    return {"items": results, "total": len(results)}


@app.get("/api/v1/memories/{memory_id}")
async def api_get_memory(request: Request, memory_id: str):
    result = await db.get_memory(memory_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, result["project_id"])):
        return err
    return result


@app.put("/api/v1/memories/{memory_id}")
async def api_update_memory(request: Request, memory_id: str, body: MemoryUpdate):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    embedding = embeddings.get_embedding(body.content) if body.content else None
    result = await db.update_memory(memory_id, content=body.content, category=body.category,
                                    tags=body.tags, embedding=embedding)
    if not result:
        return JSONResponse({"error": "Not found or archived"}, status_code=404)
    org = await _resolve_project_org(existing["project_id"])
    if org:
        audit.log_event(org["id"], "memory.update",
                        actor_user_id=request.state.auth.user.id,
                        actor_api_key_id=request.state.auth.api_key_id,
                        resource_type="memory", resource_id=memory_id)
    return result


@app.delete("/api/v1/memories/{memory_id}")
async def api_delete_memory(request: Request, memory_id: str):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    ok = await db.delete_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found or already archived"}, status_code=404)
    org = await _resolve_project_org(existing["project_id"])
    if org:
        audit.log_event(org["id"], "memory.delete",
                        actor_user_id=request.state.auth.user.id,
                        actor_api_key_id=request.state.auth.api_key_id,
                        resource_type="memory", resource_id=memory_id)
        await db.increment_memory_count(org["id"], -1)
    return {"status": "archived", "id": memory_id}


# Archive / pin / export — same shape, ACL-gated

@app.get("/api/v1/archive")
async def api_list_archived(
    request: Request,
    project_id: str = Query(...),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    if (err := _require_acl(request, project_id)):
        return err
    results = await db.list_archived([project_id], limit, offset)
    return {"items": results, "total": len(results)}


@app.post("/api/v1/archive/{memory_id}/restore")
async def api_restore_memory(request: Request, memory_id: str):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    ok = await db.restore_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found or not archived"}, status_code=404)
    return {"status": "restored", "id": memory_id}


@app.delete("/api/v1/archive/{memory_id}")
async def api_purge_memory(request: Request, memory_id: str):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    ok = await db.purge_memory(memory_id)
    if not ok:
        return JSONResponse({"error": "Not found or not archived (must soft-delete first)"}, status_code=404)
    return {"status": "purged", "id": memory_id}


@app.post("/api/v1/memories/{memory_id}/pin")
async def api_pin_memory(request: Request, memory_id: str):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    ok = await db.pin_memory(memory_id)
    return {"status": "pinned" if ok else "noop", "id": memory_id}


@app.delete("/api/v1/memories/{memory_id}/pin")
async def api_unpin_memory(request: Request, memory_id: str):
    existing = await db.get_memory(memory_id)
    if not existing:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if (err := _require_acl(request, existing["project_id"])):
        return err
    ok = await db.unpin_memory(memory_id)
    return {"status": "unpinned" if ok else "noop", "id": memory_id}


@app.get("/api/v1/export")
async def api_export(request: Request, project_id: str = Query("")):
    auth = request.state.auth
    if project_id:
        if (err := _require_acl(request, project_id)):
            return err
        scope = [project_id]
    else:
        scope = sorted(auth.allowed_project_ids)
    memories = await db.export_memories(scope)
    export_data = [
        {
            "content": m["content"], "project_id": m["project_id"],
            "category": m["category"], "tags": m["tags"],
            "pinned": m.get("pinned", False), "created_at": m["created_at"],
        }
        for m in memories
    ]
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

    auth = request.state.auth
    imported = skipped = 0
    for item in data:
        content = item.get("content", "").strip()
        pid = item.get("project_id", "").strip()
        if not content or not pid:
            skipped += 1
            continue
        if pid not in auth.allowed_project_ids:
            skipped += 1
            continue
        embedding = embeddings.get_embedding(content)
        await db.save_memory(
            pid, content, item.get("category", "general"), item.get("tags", []), embedding,
            pinned=bool(item.get("pinned", False)),
            created_at=_parse_iso(item.get("created_at")),
        )
        org = await _resolve_project_org(pid)
        if org:
            await db.increment_memory_count(org["id"], 1)
        imported += 1
    return {"status": "ok", "imported": imported, "skipped": skipped, "total": len(data)}


# =============================================================================
# Account/Project/Key REST endpoints
# =============================================================================

@app.get("/api/v1/me")
async def api_me(request: Request):
    auth = request.state.auth
    return {
        "id": auth.user.id, "email": auth.user.email,
        "allowed_projects": sorted(auth.allowed_project_ids),
    }


@app.get("/api/v1/orgs")
async def api_orgs(request: Request):
    auth = request.state.auth
    return {"orgs": await db.list_user_orgs(auth.user.id)}


@app.get("/api/v1/projects")
async def api_projects(request: Request):
    auth = request.state.auth
    return {"projects": await db.list_user_accessible_projects(auth.user.id)}


# =============================================================================
# UI: Auth flow
# =============================================================================

@app.get("/ui/login", response_class=HTMLResponse)
async def ui_login(request: Request):
    if request.cookies.get("selfmem_session"):
        return RedirectResponse("/ui/", status_code=302)
    return _render(request, "login.html", {"error": ""})


@app.post("/ui/login")
async def ui_login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    ip = _client_ip(request)
    if ratelimit.login_throttled(ip):
        return _render(request, "login.html", {"error": "Too many login attempts. Try again in a minute."})
    user = await db.get_user_by_email(email)
    if not user or not await passwords.verify_password(password, user["password_hash"]):
        ratelimit.record_login_failure(ip)
        # Audit failure on a placeholder org? We don't have an org context for failed logins.
        # Skip writing audit_log for failures since the schema requires org_id.
        return _render(request, "login.html", {"error": "Invalid email or password"})
    ratelimit.reset_login_failures(ip)
    await db.update_last_login(user["id"])
    response = RedirectResponse("/ui/", status_code=302)
    response.set_cookie("selfmem_session",
                        create_session_token(user["id"]),
                        httponly=True, max_age=config.SESSION_MAX_AGE)
    # Audit user.login on personal org if it exists
    orgs = await db.list_user_orgs(user["id"])
    personal = next((o for o in orgs if o["is_personal"]), orgs[0] if orgs else None)
    if personal:
        audit.log_event(personal["id"], "user.login",
                        actor_user_id=user["id"], ip_address=ip)
    return response


@app.get("/ui/signup", response_class=HTMLResponse)
async def ui_signup(request: Request):
    return _render(request, "signup.html", {"error": ""})


@app.post("/ui/signup")
async def ui_signup_post(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if not email or "@" not in email:
        return _render(request, "signup.html", {"error": "Enter a valid email"})
    if len(password) < 8:
        return _render(request, "signup.html", {"error": "Password must be at least 8 characters"})
    if await db.get_user_by_email(email):
        return _render(request, "signup.html", {"error": "An account with that email already exists"})

    pw_hash = await passwords.hash_password(password)
    user = await db.create_user(email, pw_hash)
    response = RedirectResponse("/ui/orgs/new?first=1", status_code=302)
    response.set_cookie("selfmem_session",
                        create_session_token(user["id"]),
                        httponly=True, max_age=config.SESSION_MAX_AGE)
    return response


@app.get("/ui/logout")
async def ui_logout():
    response = RedirectResponse("/ui/login", status_code=302)
    response.delete_cookie("selfmem_session")
    response.delete_cookie("selfmem_project")
    return response


@app.get("/ui/forgot", response_class=HTMLResponse)
async def ui_forgot(request: Request):
    return _render(request, "forgot.html", {"message": "", "error": ""})


@app.post("/ui/forgot")
async def ui_forgot_post(request: Request, email: str = Form(...)):
    user = await db.get_user_by_email(email.strip().lower())
    msg = "If an account exists for that email, a reset link has been sent."
    if user:
        token = generate_token()
        await db.create_password_reset(token, user["id"])
        log.info("[password reset] %s/ui/reset/%s", config.PUBLIC_URL, token)
        # SMTP send is phase 3
    return _render(request, "forgot.html", {"message": msg, "error": ""})


@app.get("/ui/reset/{token}", response_class=HTMLResponse)
async def ui_reset(request: Request, token: str):
    pr = await db.get_password_reset(token)
    if not pr or pr.get("used_at"):
        return _render(request, "reset.html", {"error": "Reset link invalid or already used", "token": ""})
    return _render(request, "reset.html", {"error": "", "token": token})


@app.post("/ui/reset/{token}")
async def ui_reset_post(request: Request, token: str, password: str = Form(...)):
    pr = await db.get_password_reset(token)
    if not pr or pr.get("used_at"):
        return _render(request, "reset.html", {"error": "Reset link invalid or already used", "token": ""})
    if len(password) < 8:
        return _render(request, "reset.html", {"error": "Password must be at least 8 characters", "token": token})
    pw_hash = await passwords.hash_password(password)
    await db.update_user_password(pr["user_id"], pw_hash)
    await db.consume_password_reset(token)
    return RedirectResponse("/ui/login", status_code=302)


# =============================================================================
# UI: Dashboard, memories, etc.
# =============================================================================

@app.post("/ui/set-project")
async def ui_set_project(request: Request):
    form = await request.form()
    project_id = form.get("project_id", "")
    response = HTMLResponse("")
    response.set_cookie("selfmem_project", project_id, max_age=config.SESSION_MAX_AGE)
    return response


@app.get("/ui/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    auth = request.state.auth
    if not await db.list_user_orgs(auth.user.id):
        return RedirectResponse("/ui/orgs/new?first=1", status_code=302)
    ctx = await _base_context(request, "dashboard")
    project_id, project_ids = _resolve_db_scope(ctx["current_project"], auth.allowed_project_ids)
    if project_id:
        scope_ids = [project_id]
        mem_count = await db.count_memories(project_id)
        archived = await db.count_archived(project_id)
    else:
        scope_ids = project_ids
        mem_count = sum([await db.count_memories(p) for p in scope_ids]) if scope_ids else 0
        archived = 0
    categories = await db.get_categories(scope_ids)
    ctx["stats"] = {"memories": mem_count, "categories": len(categories), "archived": archived, "latest": ""}
    ctx["memories"] = (
        await db.list_memories(project_id=project_id, limit=10)
        if project_id
        else await db.list_memories(project_ids=project_ids, limit=10)
    )
    return _render(request, "dashboard.html", ctx)


@app.get("/ui/memories", response_class=HTMLResponse)
async def ui_memories(
    request: Request,
    filter_category: str = Query(""),
    filter_tag: str = Query(""),
):
    ctx = await _base_context(request, "memories")
    auth = request.state.auth
    project_id, project_ids = _resolve_db_scope(ctx["current_project"], auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    ctx["categories"] = await db.get_categories(scope_ids)
    ctx["filter_category"] = filter_category
    ctx["filter_tag"] = filter_tag
    tags_filter = [filter_tag] if filter_tag else None
    if project_id:
        ctx["total"] = await db.count_memories(project_id, filter_category)
    elif scope_ids:
        counts = [await db.count_memories(p, filter_category) for p in scope_ids]
        ctx["total"] = sum(counts)
    else:
        ctx["total"] = 0
    if project_id:
        memories = await db.list_memories(
            project_id=project_id, category=filter_category, tags=tags_filter, limit=PAGE_SIZE,
        )
    else:
        memories = await db.list_memories(
            project_ids=project_ids, category=filter_category, tags=tags_filter, limit=PAGE_SIZE,
        )
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
    auth = request.state.auth
    raw_project = project_id or _get_current_project_cookie(request) or "__all__"
    scope_pid, scope_pids = _resolve_db_scope(raw_project, auth.allowed_project_ids)
    if scope_pid and scope_pid not in auth.allowed_project_ids:
        return HTMLResponse("")
    offset = (page - 1) * PAGE_SIZE
    if query:
        emb = embeddings.get_embedding(query)
        if scope_pid:
            memories = await db.search_memories(
                project_id=scope_pid, query_text=query, query_embedding=emb,
                category=category, limit=PAGE_SIZE,
            )
        else:
            memories = await db.search_memories(
                query_text=query, query_embedding=emb, category=category,
                project_ids=scope_pids, limit=PAGE_SIZE,
            )
        total = len(memories)
        total_pages = 1
    else:
        if scope_pid:
            total = await db.count_memories(scope_pid, category)
        else:
            total = sum([await db.count_memories(p, category) for p in scope_pids]) if scope_pids else 0
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        if scope_pid:
            memories = await db.list_memories(
                project_id=scope_pid, category=category, limit=PAGE_SIZE, offset=offset,
            )
        else:
            memories = await db.list_memories(
                project_ids=scope_pids, category=category, limit=PAGE_SIZE, offset=offset,
            )
    return _render(request, "partials/memory_list.html", {
        "memories": memories, "page": page, "total_pages": total_pages,
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

    auth = request.state.auth
    if project_id not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)

    org = await _resolve_project_org(project_id)
    if not org:
        return HTMLResponse("Unknown project", status_code=404)
    if config.IS_SAAS:
        try:
            await quotas.check_memory_quota(org["id"], org["plan_tier"], org["slug"])
        except QuotaExceeded:
            response = HTMLResponse("Quota exceeded — upgrade to save more memories")
            response.headers.update(_toast_headers("Free tier limit reached", "error"))
            return response

    embedding = embeddings.get_embedding(content)
    await db.save_memory(project_id, content, category, tags, embedding)
    await db.increment_memory_count(org["id"], 1)
    audit.log_event(org["id"], "memory.save",
                    actor_user_id=auth.user.id, resource_type="memory",
                    metadata={"project_id": project_id})

    memories = await db.list_memories(project_id=project_id, limit=PAGE_SIZE)
    total = await db.count_memories(project_id)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))

    response = _render(request, "partials/memory_list.html", {
        "memories": memories, "page": 1, "total_pages": total_pages,
        "current_project": project_id,
    })
    response.headers.update(_toast_headers("Memory saved"))
    return response


@app.put("/ui/memories/{memory_id}", response_class=HTMLResponse)
async def ui_update_memory(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    form = await request.form()
    content = form.get("content", "")
    category = form.get("category", "")
    tags_str = form.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    emb = embeddings.get_embedding(content) if content else None
    await db.update_memory(memory_id, content=content or None, category=category or None,
                            tags=tags, embedding=emb)
    mem = await db.get_memory(memory_id)
    response = _render(request, "partials/memory_row.html", {"mem": mem})
    response.headers.update(_toast_headers("Memory updated"))
    return response


@app.delete("/ui/memories/{memory_id}", response_class=HTMLResponse)
async def ui_delete_memory(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    await db.delete_memory(memory_id)
    org = await _resolve_project_org(existing["project_id"])
    if org:
        await db.increment_memory_count(org["id"], -1)
        audit.log_event(org["id"], "memory.delete",
                        actor_user_id=auth.user.id, resource_type="memory",
                        resource_id=memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory archived"))
    return response


@app.post("/ui/memories/{memory_id}/pin", response_class=HTMLResponse)
async def ui_pin_memory(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    await db.pin_memory(memory_id)
    mem = await db.get_memory(memory_id)
    raw_project = await _resolve_current_project(request)
    response = _render(request, "partials/memory_row.html",
                       {"mem": mem, "current_project": raw_project})
    response.headers.update(_toast_headers("Memory pinned"))
    return response


@app.post("/ui/memories/{memory_id}/unpin", response_class=HTMLResponse)
async def ui_unpin_memory(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    await db.unpin_memory(memory_id)
    mem = await db.get_memory(memory_id)
    raw_project = await _resolve_current_project(request)
    response = _render(request, "partials/memory_row.html",
                       {"mem": mem, "current_project": raw_project})
    response.headers.update(_toast_headers("Memory unpinned"))
    return response


@app.get("/ui/partials/memory/{memory_id}/related", response_class=HTMLResponse)
async def ui_related_memories(request: Request, memory_id: str):
    mem = await db.get_memory(memory_id)
    if not mem:
        return HTMLResponse("")
    auth = request.state.auth
    if mem["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("")
    related = await db.get_related_memories(memory_id, mem["project_id"], limit=5)
    return _render(request, "partials/related_list.html",
                   {"memories": related, "source_id": memory_id})


# Tags / Categories / Archive

@app.get("/ui/tags", response_class=HTMLResponse)
async def ui_tags(request: Request):
    ctx = await _base_context(request, "tags")
    auth = request.state.auth
    project_id, project_ids = _resolve_db_scope(ctx["current_project"], auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    rows = await db.get_tags_with_counts(scope_ids)
    tags_by_project: dict[str, list] = {}
    for row in rows:
        pid = row["project_id"]
        tags_by_project.setdefault(pid, []).append(row)
    ctx["tags_by_project"] = tags_by_project
    return _render(request, "tags.html", ctx)


@app.get("/ui/categories", response_class=HTMLResponse)
async def ui_categories(request: Request):
    ctx = await _base_context(request, "categories")
    auth = request.state.auth
    project_id, project_ids = _resolve_db_scope(ctx["current_project"], auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    rows = await db.get_categories_with_counts(scope_ids)
    categories_by_project: dict[str, list] = {}
    for row in rows:
        pid = row["project_id"]
        categories_by_project.setdefault(pid, []).append(row)
    ctx["categories_by_project"] = categories_by_project
    return _render(request, "categories.html", ctx)


@app.get("/ui/archive", response_class=HTMLResponse)
async def ui_archive(request: Request):
    ctx = await _base_context(request, "archive")
    auth = request.state.auth
    project_id, project_ids = _resolve_db_scope(ctx["current_project"], auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    ctx["memories"] = await db.list_archived(scope_ids)
    return _render(request, "archive.html", ctx)


@app.get("/ui/partials/archive", response_class=HTMLResponse)
async def ui_partials_archive(request: Request):
    auth = request.state.auth
    raw_project = await _resolve_current_project(request)
    project_id, project_ids = _resolve_db_scope(raw_project, auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    memories = await db.list_archived(scope_ids)
    return _render(request, "partials/archive_list.html",
                   {"memories": memories, "current_project": raw_project})


@app.post("/ui/archive/{memory_id}/restore", response_class=HTMLResponse)
async def ui_restore_memory(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    await db.restore_memory(memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory restored"))
    return response


@app.delete("/ui/archive/{memory_id}", response_class=HTMLResponse)
async def ui_purge_memory_ui(request: Request, memory_id: str):
    auth = request.state.auth
    existing = await db.get_memory(memory_id)
    if not existing or existing["project_id"] not in auth.allowed_project_ids:
        return HTMLResponse("Forbidden", status_code=403)
    await db.purge_memory(memory_id)
    response = HTMLResponse("")
    response.headers.update(_toast_headers("Memory permanently deleted"))
    return response


# --- Export/Import UI ---

@app.get("/ui/export", response_class=HTMLResponse)
async def ui_export(request: Request):
    auth = request.state.auth
    raw_project = await _resolve_current_project(request)
    project_id, project_ids = _resolve_db_scope(raw_project, auth.allowed_project_ids)
    scope_ids = [project_id] if project_id else project_ids
    memories = await db.export_memories(scope_ids)
    export_data = [{"content": m["content"], "project_id": m["project_id"],
                    "category": m["category"], "tags": m["tags"],
                    "pinned": m.get("pinned", False), "created_at": m["created_at"]}
                   for m in memories]
    content = json.dumps(export_data, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=selfmem-export-{project_id or 'all'}.json"},
    )


@app.post("/ui/import", response_class=HTMLResponse)
async def ui_import(request: Request):
    auth = request.state.auth
    form = await request.form()
    upload = form.get("file")
    if not upload:
        return _toast_html("No file selected", error=True)
    body = await upload.read()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return _toast_html("Invalid JSON file", error=True)
    if not isinstance(data, list):
        return _toast_html("Expected JSON array", error=True)
    imported = skipped = 0
    for item in data:
        c = item.get("content", "").strip()
        pid = item.get("project_id", "").strip()
        if not c or not pid or pid not in auth.allowed_project_ids:
            skipped += 1
            continue
        embedding = embeddings.get_embedding(c)
        await db.save_memory(
            pid, c, item.get("category", "general"),
            item.get("tags", []), embedding,
            pinned=bool(item.get("pinned", False)),
            created_at=_parse_iso(item.get("created_at")),
        )
        org = await _resolve_project_org(pid)
        if org:
            await db.increment_memory_count(org["id"], 1)
        imported += 1
    return _toast_html(f"{imported} imported, {skipped} skipped")


def _toast_html(message: str, error: bool = False) -> HTMLResponse:
    cls = "text-red-400" if error else "text-emerald-400"
    response = HTMLResponse(f"<div class='{cls} text-sm'>{message}</div>")
    response.headers.update(_toast_headers(message, "error" if error else "success"))
    return response


# =============================================================================
# UI: Account / Keys / Projects (phase 1 minimal subset)
# =============================================================================

@app.get("/ui/account", response_class=HTMLResponse)
async def ui_account(request: Request):
    ctx = await _base_context(request, "account")
    return _render(request, "account.html", ctx)


@app.post("/ui/account/password")
async def ui_account_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
):
    auth = request.state.auth
    user = await db.get_user_by_email(auth.user.email)
    if not user or not await passwords.verify_password(current_password, user["password_hash"]):
        ctx = await _base_context(request, "account")
        ctx["password_error"] = "Current password is incorrect"
        return _render(request, "account.html", ctx)
    if len(new_password) < 8:
        ctx = await _base_context(request, "account")
        ctx["password_error"] = "New password must be at least 8 characters"
        return _render(request, "account.html", ctx)
    pw_hash = await passwords.hash_password(new_password)
    await db.update_user_password(auth.user.id, pw_hash)
    return RedirectResponse("/ui/account", status_code=302)


@app.get("/ui/account/keys", response_class=HTMLResponse)
async def ui_account_keys(request: Request):
    ctx = await _base_context(request, "keys")
    auth = request.state.auth
    ctx["api_keys"] = await db.list_user_api_keys(auth.user.id)
    return _render(request, "account_keys.html", ctx)


@app.post("/ui/account/keys")
async def ui_account_keys_create(request: Request):
    auth = request.state.auth
    form = await request.form()
    name = form.get("name", "").strip() or "Untitled key"
    project_ids = form.getlist("project_ids")
    # Filter to only projects the user can access
    allowed_pids = [p for p in project_ids if p in auth.allowed_project_ids]
    raw, prefix, h = generate_api_key()
    await db.create_api_key(auth.user.id, name, prefix, h, allowed_pids)
    # Show the raw key once
    orgs = await db.list_user_orgs(auth.user.id)
    if orgs:
        audit.log_event(orgs[0]["id"], "key.created",
                        actor_user_id=auth.user.id, resource_type="key",
                        metadata={"name": name, "project_ids": allowed_pids})
    ctx = await _base_context(request, "keys")
    ctx["api_keys"] = await db.list_user_api_keys(auth.user.id)
    ctx["new_key_value"] = raw
    return _render(request, "account_keys.html", ctx)


@app.post("/ui/account/keys/{key_id}/revoke")
async def ui_revoke_key(request: Request, key_id: str):
    auth = request.state.auth
    ok = await db.revoke_api_key(key_id, auth.user.id)
    if ok:
        orgs = await db.list_user_orgs(auth.user.id)
        if orgs:
            audit.log_event(orgs[0]["id"], "key.revoked",
                            actor_user_id=auth.user.id, resource_type="key", resource_id=key_id)
    return RedirectResponse("/ui/account/keys", status_code=302)


@app.post("/ui/projects/new")
async def ui_create_project(request: Request, project_id: str = Form(...), name: str = Form("")):
    auth = request.state.auth
    pid = project_id.strip().lower()
    if not _is_valid_slug(pid):
        return _toast_html("Project ID must be lowercase letters/numbers/hyphens (3-100 chars)", error=True)
    if await db.project_id_exists(pid):
        return _toast_html(f"Project '{pid}' already exists", error=True)
    # Pick the user's first owner-org as the default parent. To put a project
    # in a specific org instead, use the per-org "+ New project" form.
    orgs = await db.list_user_orgs(auth.user.id)
    owner_org = next((o for o in orgs if o["role"] == "owner"), None)
    if not owner_org:
        return _toast_html("Create an org first at /ui/orgs/new", error=True)
    if config.IS_SAAS:
        try:
            org_full = await db.get_org_by_id(owner_org["id"])
            await quotas.check_project_quota(owner_org["id"], org_full["plan_tier"], org_full["slug"])
        except QuotaExceeded:
            return _toast_html("Free tier project limit reached", error=True)
    await db.create_project(pid, owner_org["id"], name or pid, auth.user.id)
    audit.log_event(owner_org["id"], "project.created",
                    actor_user_id=auth.user.id, resource_type="project", resource_id=pid)
    return RedirectResponse("/ui/", status_code=302)


# =============================================================================
# UI: Orgs (phase 2)
# =============================================================================

@app.get("/ui/orgs", response_class=HTMLResponse)
async def ui_orgs(request: Request):
    ctx = await _base_context(request, "orgs")
    auth = request.state.auth
    ctx["orgs"] = await db.list_user_orgs(auth.user.id)
    return _render(request, "orgs_list.html", ctx)


@app.get("/ui/orgs/new", response_class=HTMLResponse)
async def ui_org_new(request: Request):
    ctx = await _base_context(request, "orgs")
    ctx["error"] = ""
    ctx["first"] = request.query_params.get("first") == "1"
    return _render(request, "org_new.html", ctx)


@app.post("/ui/orgs/new")
async def ui_org_new_post(
    request: Request,
    slug: str = Form(...),
    name: str = Form(...),
):
    ctx = await _base_context(request, "orgs")
    ctx["first"] = request.query_params.get("first") == "1"
    auth = request.state.auth
    slug = slug.strip().lower()
    if not _is_valid_slug(slug):
        ctx["error"] = "Slug must be lowercase letters/numbers/hyphens (3-100 chars)"
        return _render(request, "org_new.html", ctx)
    if await db.slug_exists(slug):
        ctx["error"] = f"Slug '{slug}' is taken"
        return _render(request, "org_new.html", ctx)
    org = await db.create_org(slug=slug, name=name.strip(), is_personal=False, owner_id=auth.user.id)
    audit.log_event(org["id"], "org.created",
                    actor_user_id=auth.user.id, resource_type="org", resource_id=org["id"],
                    ip_address=_client_ip(request))
    return RedirectResponse(f"/ui/orgs/{slug}", status_code=302)


async def _require_org_membership(request: Request, slug: str) -> tuple[dict, dict] | JSONResponse:
    """Return (org, member) or a 403 response."""
    auth = request.state.auth
    org = await db.get_org_by_slug(slug)
    if not org:
        return JSONResponse({"error": "Org not found"}, status_code=404)
    member = await db.get_org_member(org["id"], auth.user.id)
    if not member:
        return JSONResponse({"error": "Not a member of this org"}, status_code=403)
    return org, member


@app.get("/ui/orgs/{slug}", response_class=HTMLResponse)
async def ui_org_view(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    ctx = await _base_context(request, "orgs")
    ctx["org"] = org
    ctx["my_role"] = member["role"]
    ctx["projects"] = await db.list_org_projects(org["id"])
    ctx["members"] = await db.list_org_members(org["id"])
    ctx["usage"] = await db.get_org_usage(org["id"])
    ctx["limits"] = {
        "memory": config.FREE_MEMORY_LIMIT,
        "project": config.FREE_PROJECT_LIMIT,
        "member": config.FREE_MEMBER_LIMIT,
    }
    return _render(request, "org_view.html", ctx)


@app.get("/ui/orgs/{slug}/audit", response_class=HTMLResponse)
async def ui_org_audit(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, _ = result
    ctx = await _base_context(request, "orgs")
    ctx["org"] = org
    ctx["events"] = await db.list_audit(org["id"], limit=200)
    return _render(request, "org_audit.html", ctx)


@app.post("/ui/orgs/{slug}/projects/new")
async def ui_org_create_project(
    request: Request, slug: str,
    project_id: str = Form(...), name: str = Form(""),
):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    auth = request.state.auth
    if member["role"] != "owner":
        return _toast_html("Only org owners can create projects", error=True)
    pid = project_id.strip().lower()
    if not _is_valid_slug(pid):
        return _toast_html("Project ID must be lowercase letters/numbers/hyphens (3-100 chars)", error=True)
    if await db.project_id_exists(pid):
        return _toast_html(f"Project '{pid}' already exists (globally unique)", error=True)
    if config.IS_SAAS:
        try:
            await quotas.check_project_quota(org["id"], org["plan_tier"], org["slug"])
        except QuotaExceeded:
            return _toast_html("Free tier project limit reached", error=True)
    await db.create_project(pid, org["id"], name or pid, auth.user.id)
    audit.log_event(org["id"], "project.created",
                    actor_user_id=auth.user.id, resource_type="project", resource_id=pid)
    return RedirectResponse(f"/ui/orgs/{slug}", status_code=302)


# --- Members ---

@app.get("/ui/orgs/{slug}/members", response_class=HTMLResponse)
async def ui_org_members(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    ctx = await _base_context(request, "orgs")
    ctx["org"] = org
    ctx["my_role"] = member["role"]
    ctx["members"] = await db.list_org_members(org["id"])
    ctx["new_invite_url"] = None
    return _render(request, "org_members.html", ctx)


@app.post("/ui/orgs/{slug}/invite")
async def ui_org_invite(
    request: Request, slug: str,
    email: str = Form(""), role: str = Form("member"),
):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    auth = request.state.auth
    if member["role"] != "owner":
        return _toast_html("Only owners can invite", error=True)
    if role not in {"owner", "member"}:
        role = "member"
    if config.IS_SAAS:
        try:
            await quotas.check_member_quota(org["id"], org["plan_tier"], org["slug"])
        except QuotaExceeded:
            return _toast_html("Free tier member limit reached", error=True)
    token = generate_token()
    await db.create_invitation(token, org["id"], email.strip().lower() or None,
                                role, auth.user.id)
    audit.log_event(org["id"], "member.invited",
                    actor_user_id=auth.user.id, resource_type="invitation",
                    metadata={"email": email or None, "role": role})
    invite_url = f"{config.PUBLIC_URL}/ui/invite/{token}"
    log.info("[invitation] %s", invite_url)
    # Re-render members page with the new invite URL displayed
    ctx = await _base_context(request, "orgs")
    ctx["org"] = org
    ctx["my_role"] = member["role"]
    ctx["members"] = await db.list_org_members(org["id"])
    ctx["new_invite_url"] = invite_url
    return _render(request, "org_members.html", ctx)


@app.post("/ui/orgs/{slug}/members/{member_user_id}/remove")
async def ui_org_remove_member(request: Request, slug: str, member_user_id: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    auth = request.state.auth
    if member["role"] != "owner":
        return _toast_html("Only owners can remove members", error=True)
    if member_user_id == auth.user.id:
        return _toast_html("You can't remove yourself", error=True)
    await db.remove_org_member(org["id"], member_user_id)
    audit.log_event(org["id"], "member.removed",
                    actor_user_id=auth.user.id, resource_type="member",
                    resource_id=member_user_id)
    return RedirectResponse(f"/ui/orgs/{slug}/members", status_code=302)


# --- Invitations: accept flow (public) ---

@app.get("/ui/invite/{token}", response_class=HTMLResponse)
async def ui_invite_view(request: Request, token: str):
    inv = await db.get_invitation(token)
    if not inv or inv.get("used_at"):
        return _render(request, "invite_accept.html",
                        {"error": "This invitation is invalid or already used", "invite": None})
    # Check expiry
    from datetime import datetime, timezone
    if inv.get("expires_at"):
        exp = inv["expires_at"]
        if isinstance(exp, str):
            try:
                exp = datetime.fromisoformat(exp)
            except ValueError:
                exp = None
        if exp and exp < datetime.now(timezone.utc):
            return _render(request, "invite_accept.html",
                            {"error": "This invitation has expired", "invite": None})
    logged_in_user_id = None
    sess = request.cookies.get("selfmem_session")
    if sess:
        logged_in_user_id = validate_session_token_safe(sess)
    return _render(request, "invite_accept.html",
                    {"error": "", "invite": inv,
                     "logged_in_user_id": logged_in_user_id, "token": token})


def validate_session_token_safe(token: str) -> str | None:
    from auth import validate_session_token
    return validate_session_token(token)


@app.post("/ui/invite/{token}")
async def ui_invite_accept(
    request: Request, token: str,
    email: str = Form(""), password: str = Form(""),
):
    inv = await db.get_invitation(token)
    if not inv or inv.get("used_at"):
        return _render(request, "invite_accept.html",
                        {"error": "Invalid or used invitation", "invite": None})

    # Path A: user is already logged in → just join the org
    sess = request.cookies.get("selfmem_session")
    user_id: str | None = None
    if sess:
        user_id = validate_session_token_safe(sess)

    if not user_id:
        # Path B: signup flow inside the invite
        if not email or not password:
            return _render(request, "invite_accept.html",
                            {"error": "Email and password required", "invite": inv, "token": token})
        if len(password) < 8:
            return _render(request, "invite_accept.html",
                            {"error": "Password must be at least 8 characters", "invite": inv, "token": token})
        existing = await db.get_user_by_email(email)
        if existing:
            return _render(request, "invite_accept.html",
                            {"error": "An account with that email exists — log in first, then re-open this invite link",
                             "invite": inv, "token": token})
        pw_hash = await passwords.hash_password(password)
        user = await db.create_user(email.strip().lower(), pw_hash)
        user_id = user["id"]
        # Signup via invitation: no personal org. Audit lands on the invited
        # org via member.joined below.

    # Add user to invited org with the invitation's role
    await db.add_org_member(str(inv["org_id"]), user_id, inv["role"])
    await db.consume_invitation(token, user_id)
    audit.log_event(str(inv["org_id"]), "member.joined",
                    actor_user_id=user_id, resource_type="member",
                    metadata={"via_invitation": True, "role": inv["role"]},
                    ip_address=_client_ip(request))

    response = RedirectResponse(f"/ui/orgs/{inv['org_slug']}", status_code=302)
    if not sess:
        response.set_cookie("selfmem_session", create_session_token(user_id),
                             httponly=True, max_age=config.SESSION_MAX_AGE)
    return response


# --- Project members ---

@app.get("/ui/projects/{project_id}/members", response_class=HTMLResponse)
async def ui_project_members(request: Request, project_id: str):
    auth = request.state.auth
    proj = await db.get_project(project_id)
    if not proj:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    org = await db.get_org_by_id(proj["org_id"])
    member = await db.get_org_member(org["id"], auth.user.id)
    if not member:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    ctx = await _base_context(request, "orgs")
    ctx["project"] = proj
    ctx["org"] = org
    ctx["my_role"] = member["role"]
    ctx["org_members"] = await db.list_org_members(org["id"])
    ctx["project_members"] = await db.list_project_members(project_id)
    return _render(request, "project_members.html", ctx)


@app.post("/ui/projects/{project_id}/members/add")
async def ui_project_add_member(request: Request, project_id: str, user_id: str = Form(...)):
    auth = request.state.auth
    proj = await db.get_project(project_id)
    if not proj:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    if not await is_org_owner(auth.user.id, proj["org_id"]):
        return JSONResponse({"error": "Only org owners can manage project access"}, status_code=403)
    # Confirm target is a member of the same org
    target_member = await db.get_org_member(proj["org_id"], user_id)
    if not target_member:
        return _toast_html("Target user is not in this org", error=True)
    await db.add_project_member(project_id, user_id)
    audit.log_event(proj["org_id"], "project.member_added",
                    actor_user_id=auth.user.id, resource_type="project",
                    resource_id=project_id, metadata={"target_user_id": user_id})
    return RedirectResponse(f"/ui/projects/{project_id}/members", status_code=302)


@app.post("/ui/projects/{project_id}/members/{member_user_id}/remove")
async def ui_project_remove_member(request: Request, project_id: str, member_user_id: str):
    auth = request.state.auth
    proj = await db.get_project(project_id)
    if not proj:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    if not await is_org_owner(auth.user.id, proj["org_id"]):
        return JSONResponse({"error": "Only org owners can manage project access"}, status_code=403)
    await db.remove_project_member(project_id, member_user_id)
    audit.log_event(proj["org_id"], "project.member_removed",
                    actor_user_id=auth.user.id, resource_type="project",
                    resource_id=project_id, metadata={"target_user_id": member_user_id})
    return RedirectResponse(f"/ui/projects/{project_id}/members", status_code=302)


# =============================================================================
# UI: Billing (phase 4, SaaS-only UI but routes always exist)
# =============================================================================

@app.get("/ui/orgs/{slug}/billing", response_class=HTMLResponse)
async def ui_org_billing(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    ctx = await _base_context(request, "orgs")
    ctx["org"] = org
    ctx["my_role"] = member["role"]
    ctx["usage"] = await db.get_org_usage(org["id"])
    ctx["limits"] = {
        "memory": config.FREE_MEMORY_LIMIT,
        "project": config.FREE_PROJECT_LIMIT,
        "member": config.FREE_MEMBER_LIMIT,
    }
    ctx["billing_configured"] = billing.is_configured()
    return _render(request, "org_billing.html", ctx)


@app.post("/ui/orgs/{slug}/billing/checkout")
async def ui_billing_checkout(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    if member["role"] != "owner":
        return _toast_html("Only org owners can upgrade", error=True)
    if not billing.is_configured():
        return _toast_html("Billing is not configured on this server", error=True)
    success_url = f"{config.PUBLIC_URL}/ui/orgs/{slug}/billing?upgraded=1"
    cancel_url = f"{config.PUBLIC_URL}/ui/orgs/{slug}/billing"
    try:
        url = await billing.create_checkout_session(org, success_url, cancel_url)
    except Exception as e:
        log.exception("checkout session error")
        return _toast_html(f"Checkout error: {e}", error=True)
    return RedirectResponse(url, status_code=302)


@app.post("/ui/orgs/{slug}/billing/portal")
async def ui_billing_portal(request: Request, slug: str):
    result = await _require_org_membership(request, slug)
    if isinstance(result, JSONResponse):
        return result
    org, member = result
    if member["role"] != "owner":
        return _toast_html("Only org owners can manage billing", error=True)
    if not org.get("stripe_customer_id"):
        return _toast_html("No active subscription yet", error=True)
    return_url = f"{config.PUBLIC_URL}/ui/orgs/{slug}/billing"
    try:
        url = await billing.create_customer_portal_url(org, return_url)
    except Exception as e:
        log.exception("portal session error")
        return _toast_html(f"Portal error: {e}", error=True)
    return RedirectResponse(url, status_code=302)


# --- Stripe webhook (no auth — verified by signature) ---
# Webhook needs to bypass APIKeyMiddleware. Easiest path: register the URL
# as exact-skip in auth.py SKIP_EXACT and re-validate the signature here.
@app.post("/api/v1/stripe/webhook")
async def api_stripe_webhook(request: Request):
    if not config.STRIPE_WEBHOOK_SECRET:
        return JSONResponse({"error": "Webhook not configured"}, status_code=503)
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = billing.verify_webhook(payload, sig)
    except Exception as e:
        log.warning("[stripe] invalid webhook signature: %s", e)
        return JSONResponse({"error": "Invalid signature"}, status_code=400)
    await billing.handle_event(event)
    return {"received": True}


# Test-only endpoint (selfhosted-mode only) — accepts a synthetic event
# without signature so we can simulate Stripe webhooks in dev/CI.
@app.post("/api/v1/stripe/webhook/test")
async def api_stripe_webhook_test(request: Request):
    if config.IS_SAAS:
        return JSONResponse({"error": "Disabled in saas mode"}, status_code=403)
    payload = await request.json()
    await billing.handle_event(payload)
    return {"received": True}


# =============================================================================
# UI: GDPR (export + delete account)
# =============================================================================

@app.get("/ui/account/data", response_class=HTMLResponse)
async def ui_account_data(request: Request):
    ctx = await _base_context(request, "account")
    auth = request.state.auth
    ctx["only_owner_orgs"] = await db.list_user_owned_only_orgs(auth.user.id)
    return _render(request, "account_data.html", ctx)


@app.get("/ui/account/data/export")
async def ui_export_user_data(request: Request):
    auth = request.state.auth
    data = await gdpr.export_user_data(auth.user.id)
    content = json.dumps(data, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=selfmem-account-{auth.user.email}.json"},
    )


@app.post("/ui/account/data/delete")
async def ui_delete_account(request: Request, confirm_email: str = Form(...)):
    auth = request.state.auth
    if confirm_email.strip().lower() != auth.user.email.lower():
        ctx = await _base_context(request, "account")
        ctx["only_owner_orgs"] = await db.list_user_owned_only_orgs(auth.user.id)
        ctx["delete_error"] = "Email confirmation does not match"
        return _render(request, "account_data.html", ctx)
    summary = await gdpr.delete_account(auth.user.id)
    log.info("[gdpr] deleted user=%s summary=%s", auth.user.id, summary)
    response = RedirectResponse("/ui/login?deleted=1", status_code=302)
    response.delete_cookie("selfmem_session")
    response.delete_cookie("selfmem_project")
    return response


# =============================================================================
# Public marketing pages (rendered always; SaaS gating in templates)
# =============================================================================

@app.get("/pricing", response_class=HTMLResponse)
async def public_pricing(request: Request):
    return _render(request, "pricing.html", {
        "is_saas": config.IS_SAAS,
        "free_limits": {
            "memory": config.FREE_MEMORY_LIMIT,
            "project": config.FREE_PROJECT_LIMIT,
            "member": config.FREE_MEMBER_LIMIT,
            "api_daily": config.FREE_API_LIMIT_DAILY,
        },
    })


@app.get("/terms", response_class=HTMLResponse)
async def public_terms(request: Request):
    return _render(request, "terms.html", {})


@app.get("/privacy", response_class=HTMLResponse)
async def public_privacy(request: Request):
    return _render(request, "privacy.html", {})


# --- Settings ---

@app.get("/ui/settings", response_class=HTMLResponse)
async def ui_settings(request: Request):
    ctx = await _base_context(request, "settings")
    try:
        await db.get_user_by_id(request.state.auth.user.id)
        db_ok = True
    except Exception:
        db_ok = False
    ctx.update({
        "db_ok": db_ok,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
    })
    return _render(request, "settings.html", ctx)


# --- Public landing ---

@app.get("/", response_class=HTMLResponse)
async def public_root(request: Request):
    if not config.IS_SAAS:
        return RedirectResponse("/ui/login", status_code=302)
    return _render(request, "landing.html", {"is_saas": True})


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
