import os

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://selfmem:changeme@db:5432/selfmem"
)

API_KEYS: set[str] = {
    k.strip()
    for k in os.environ.get("SELFMEM_API_KEYS", "").split(",")
    if k.strip()
}

EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "384"))

SESSION_SECRET: str = os.environ.get(
    "SELFMEM_SESSION_SECRET", "change-me-in-production"
)
SESSION_MAX_AGE: int = int(os.environ.get("SELFMEM_SESSION_MAX_AGE", "86400"))

HOST: str = os.environ.get("SELFMEM_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("SELFMEM_PORT", "8818"))
