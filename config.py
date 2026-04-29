import os

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://selfmem:changeme@db:5432/selfmem"
)

# --- Sessions ---
SESSION_SECRET: str = os.environ.get(
    "SELFMEM_SESSION_SECRET", "change-me-in-production"
)
SESSION_MAX_AGE: int = int(os.environ.get("SELFMEM_SESSION_MAX_AGE", "86400"))

# --- Hosting mode ---
# 'selfhosted' or 'saas'. Gates quotas, billing UI, rate limiting, marketing pages.
HOSTING_MODE: str = os.environ.get("SELFMEM_HOSTING_MODE", "selfhosted").lower()
IS_SAAS: bool = HOSTING_MODE == "saas"

# --- Free-tier limits (only enforced when IS_SAAS) ---
FREE_MEMORY_LIMIT: int = int(os.environ.get("SELFMEM_FREE_MEMORY_LIMIT", "100"))
FREE_PROJECT_LIMIT: int = int(os.environ.get("SELFMEM_FREE_PROJECT_LIMIT", "1"))
FREE_MEMBER_LIMIT: int = int(os.environ.get("SELFMEM_FREE_MEMBER_LIMIT", "1"))
FREE_API_LIMIT_DAILY: int = int(os.environ.get("SELFMEM_FREE_API_LIMIT_DAILY", "100"))
PAID_API_LIMIT_DAILY: int = int(os.environ.get("SELFMEM_PAID_API_LIMIT_DAILY", "10000"))

# --- Public URL (for invitation + password-reset email links) ---
PUBLIC_URL: str = os.environ.get("SELFMEM_PUBLIC_URL", "http://localhost:8818").rstrip("/")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "384"))

# --- Server ---
HOST: str = os.environ.get("SELFMEM_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("SELFMEM_PORT", "8818"))

# --- SMTP (phase 3) ---
SMTP_HOST: str = os.environ.get("SELFMEM_SMTP_HOST", "")
SMTP_PORT: int = int(os.environ.get("SELFMEM_SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SELFMEM_SMTP_USER", "")
SMTP_PASSWORD: str = os.environ.get("SELFMEM_SMTP_PASSWORD", "")
SMTP_FROM: str = os.environ.get("SELFMEM_SMTP_FROM", "noreply@selfmem.local")

# --- Stripe (phase 4) ---
STRIPE_SECRET_KEY: str = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET: str = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLIC_KEY: str = os.environ.get("STRIPE_PUBLIC_KEY", "")
STRIPE_UNLIMITED_PRICE_ID: str = os.environ.get("STRIPE_UNLIMITED_PRICE_ID", "")
