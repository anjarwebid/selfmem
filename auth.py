"""Auth middleware.

UI routes (/ui/*) require a cookie session that signs `user_id`.
REST + MCP routes require an `X-API-Key` header (SHA-256-hashed lookup).

Both paths populate `request.state.auth` with an `AuthContext` containing the
user, optional api_key_id, and the union of project IDs accessible to that
identity.
"""
from __future__ import annotations

import json
import logging

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

import config
import db
from models import AuthContext, User
from tokens import hash_api_key

log = logging.getLogger(__name__)

# Public paths that bypass auth entirely.
SKIP_EXACT = {
    "/health", "/health/",
    "/ui/login", "/ui/login/",
    "/ui/signup", "/ui/signup/",
    "/ui/forgot", "/ui/forgot/",
    "/", "/pricing", "/terms", "/privacy",
    "/api/v1/stripe/webhook", "/api/v1/stripe/webhook/test",
}
SKIP_PREFIXES = (
    "/static/",
    "/ui/reset/",
    "/ui/invite/",
)

_serializer: URLSafeTimedSerializer | None = None


def get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        _serializer = URLSafeTimedSerializer(config.SESSION_SECRET)
    return _serializer


def create_session_token(user_id: str) -> str:
    return get_serializer().dumps(user_id)


def validate_session_token(token: str) -> str | None:
    try:
        return get_serializer().loads(token, max_age=config.SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def _user_from_row(row: dict) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        email_verified=row.get("email_verified", False),
        created_at=row.get("created_at"),
        last_login_at=row.get("last_login_at"),
        deleted_at=row.get("deleted_at"),
    )


async def _build_user_auth_context(user_row: dict) -> AuthContext:
    """For UI sessions: user has access to all projects in orgs they own
    + projects where they're an explicit project_member."""
    accessible = await db.list_user_accessible_projects(user_row["id"])
    return AuthContext(
        user=_user_from_row(user_row),
        api_key_id=None,
        allowed_project_ids={p["id"] for p in accessible},
    )


async def _build_apikey_auth_context(user_row: dict, key_id: str, key_project_ids: list[str]) -> AuthContext:
    """For API keys: ACL is exactly the key's project list. We do NOT union
    with org-owner projects — the key's grant is intentional and explicit."""
    return AuthContext(
        user=_user_from_row(user_row),
        api_key_id=key_id,
        allowed_project_ids=set(key_project_ids),
    )


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in SKIP_EXACT or any(path.startswith(p) for p in SKIP_PREFIXES):
            return await call_next(request)

        # UI routes: cookie session
        if path.startswith("/ui/"):
            token = request.cookies.get("selfmem_session")
            if not token:
                return RedirectResponse("/ui/login", status_code=302)

            user_id = validate_session_token(token)
            if not user_id:
                response = RedirectResponse("/ui/login", status_code=302)
                response.delete_cookie("selfmem_session")
                return response

            user_row = await db.get_user_by_id(user_id)
            if not user_row:
                response = RedirectResponse("/ui/login", status_code=302)
                response.delete_cookie("selfmem_session")
                return response

            request.state.auth = await _build_user_auth_context(user_row)
            return await call_next(request)

        # API/MCP routes: X-API-Key header
        raw_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if not raw_key:
            return _json_error("Missing API key", 401)

        key_row = await db.get_api_key_by_hash(hash_api_key(raw_key))
        if not key_row:
            return _json_error("Invalid API key", 401)

        user_row = await db.get_user_by_id(key_row["user_id"])
        if not user_row:
            return _json_error("Invalid API key", 401)

        request.state.auth = await _build_apikey_auth_context(
            user_row, key_row["id"], key_row.get("project_ids") or []
        )
        # Touch last_used_at without blocking the response
        try:
            import asyncio
            asyncio.create_task(db.touch_api_key(key_row["id"]))
        except Exception:
            pass
        return await call_next(request)


def _json_error(message: str, status: int) -> Response:
    return Response(
        content=json.dumps({"error": message}),
        status_code=status,
        media_type="application/json",
    )


# --- helpers usable from route handlers ---

def get_auth(request: Request) -> AuthContext:
    return request.state.auth


def authorize_project(request: Request, project_id: str) -> bool:
    auth: AuthContext = request.state.auth
    return auth.can_access(project_id)


def require_project(request: Request, project_id: str) -> Response | None:
    """Return None if allowed, else a 403 Response."""
    if not authorize_project(request, project_id):
        return _json_error("Forbidden", 403)
    return None


async def get_org_for_user(user_id: str, slug: str) -> tuple[dict, dict] | None:
    """Return (org, member) if user is a member of this org, else None."""
    org = await db.get_org_by_slug(slug)
    if not org:
        return None
    member = await db.get_org_member(org["id"], user_id)
    if not member:
        return None
    return org, member


async def is_org_owner(user_id: str, org_id: str) -> bool:
    member = await db.get_org_member(org_id, user_id)
    return bool(member and member["role"] == "owner")
