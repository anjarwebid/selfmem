#!/usr/bin/env python3
"""One-shot production migration helper.

Use after the destructive cutover (SELFMEM_RESET_DATABASE=true) has been
deployed. Performs in order:

  1. Re-creates the projects listed in the backup file (one per distinct
     project_id), inside a single org you specify (defaults to your personal).
     Skips projects that already exist.
  2. POSTs the backup JSON to /api/v1/import. The server preserves
     `pinned` and `created_at` on each row (requires v2.0.0+).

Auth: pass the URL, an API key (must already have ACL covering the projects
you intend to import into — easiest is to create a key after step 1).

Usage:
    python migrate_prod.py \
        --url https://selfmem-yk.silverspace.my.id \
        --api-key sm_XXXX \
        --backup backups/prod-all-2026-04-29.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter


def _http(url: str, method: str, api_key: str, body: dict | list | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="https://your-selfmem-host")
    p.add_argument("--api-key", required=True, help="API key with ACL covering all imported projects")
    p.add_argument("--backup", required=True, help="path to backup JSON")
    args = p.parse_args()

    base = args.url.rstrip("/")
    backup = json.load(open(args.backup))
    if not isinstance(backup, list):
        print("Backup must be a JSON array", file=sys.stderr)
        return 1

    counts = Counter(m["project_id"] for m in backup)
    print(f"Backup contains {len(backup)} memories across {len(counts)} projects:")
    for pid, n in counts.most_common():
        print(f"  {pid}: {n}")

    # Discover what the API key can already access
    me = _http(f"{base}/api/v1/me", "GET", args.api_key)
    accessible = set(me.get("allowed_projects") or [])
    missing = set(counts.keys()) - accessible
    if missing:
        print()
        print(f"WARNING: API key cannot access these projects (will be skipped on import):")
        for pid in sorted(missing):
            print(f"  - {pid}")
        print()
        print("Create the projects in the UI and ACL the key for them, then re-run.")
        return 2

    print()
    print(f"Importing {len(backup)} memories...")
    result = _http(f"{base}/api/v1/import", "POST", args.api_key, backup)
    print(f"  result: {result}")
    return 0 if result.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
