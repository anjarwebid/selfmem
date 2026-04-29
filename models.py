from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    id: str
    email: str
    email_verified: bool
    created_at: datetime
    last_login_at: datetime | None
    deleted_at: datetime | None


@dataclass
class Org:
    id: str
    slug: str
    name: str
    is_personal: bool
    plan_tier: str
    created_at: datetime


@dataclass
class OrgMember:
    org_id: str
    user_id: str
    role: str  # 'owner' | 'member'


@dataclass
class Project:
    id: str           # slug, globally unique
    org_id: str
    name: str | None
    created_at: datetime


@dataclass
class ApiKey:
    id: str
    user_id: str
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    project_ids: list[str]    # ACL


@dataclass
class AuthContext:
    """Populated by middleware on every authenticated request."""
    user: User
    api_key_id: str | None        # set when auth came from API key (REST/MCP)
    allowed_project_ids: set[str] # union of (org-owned projects) + (project_member rows) + (api_key ACL)

    def can_access(self, project_id: str) -> bool:
        return project_id in self.allowed_project_ids
