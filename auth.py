import json
import logging

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

import config

log = logging.getLogger(__name__)

SKIP_PATHS = {"/health", "/health/", "/ui/login", "/ui/login/", "/static"}

_serializer: URLSafeTimedSerializer | None = None


def get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        _serializer = URLSafeTimedSerializer(config.SESSION_SECRET)
    return _serializer


def create_session_token(api_key: str) -> str:
    return get_serializer().dumps(api_key)


def validate_session_token(token: str) -> str | None:
    try:
        return get_serializer().loads(token, max_age=config.SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for health and login
        if path in SKIP_PATHS or path.startswith("/static/"):
            return await call_next(request)

        # UI routes: check session cookie
        if path.startswith("/ui/"):
            token = request.cookies.get("selfmem_session")
            if not token:
                return RedirectResponse("/ui/login", status_code=302)

            api_key = validate_session_token(token)
            if not api_key or (config.API_KEYS and api_key not in config.API_KEYS):
                response = RedirectResponse("/ui/login", status_code=302)
                response.delete_cookie("selfmem_session")
                return response

            return await call_next(request)

        # API/MCP routes: check X-API-Key header or query param
        api_key = request.headers.get("x-api-key") or request.query_params.get(
            "api_key"
        )

        if not config.API_KEYS:
            log.warning("No API keys configured — all requests allowed")
            return await call_next(request)

        if not api_key or api_key not in config.API_KEYS:
            return Response(
                content=json.dumps({"error": "Invalid or missing API key"}),
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
