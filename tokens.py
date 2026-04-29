import hashlib
import secrets


API_KEY_PREFIX = "sm_"


def generate_token(nbytes: int = 32) -> str:
    """URL-safe random token. ~43 chars at 32 bytes."""
    return secrets.token_urlsafe(nbytes)


def generate_api_key() -> tuple[str, str, str]:
    """Generate a fresh API key.

    Returns (raw_key, key_prefix, key_hash). Raw key is shown once to the user;
    only the prefix and hash are stored. Hash uses SHA-256 for fast lookup
    (bcrypt is too slow for per-request authentication).
    """
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    prefix = raw[:12]
    digest = hash_api_key(raw)
    return raw, prefix, digest


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
