#!/usr/bin/env python3
"""Import all memories from OpenMemory to SelfMem."""

import json
import sys
import urllib.request

OPENMEMORY_BASE = "https://openmemory-yk.silverspace.my.id"
SELFMEM_BASE = "http://localhost:8818"
SELFMEM_API_KEY = sys.argv[1] if len(sys.argv) > 1 else ""

if not SELFMEM_API_KEY:
    print("Usage: python import_openmemory.py <SELFMEM_API_KEY>")
    sys.exit(1)


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def post_json(url: str, data: dict, api_key: str) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_users() -> list[dict]:
    data = fetch_json(f"{OPENMEMORY_BASE}/api/v1/stats/users")
    return data.get("users", [])


def get_memories(user_id: str) -> list[dict]:
    data = fetch_json(
        f"{OPENMEMORY_BASE}/api/v1/memories/?user_id={user_id}&size=100"
    )
    return data.get("items", [])


def extract_category(categories: list[str]) -> str:
    if not categories:
        return "general"
    # Use the first category as primary
    return categories[0].replace(" ", "-").lower()


def import_memory(user_id: str, memory: dict) -> dict | None:
    content = memory.get("content", "").strip()
    if not content:
        return None

    categories = memory.get("categories", [])
    category = extract_category(categories)
    tags = [c.replace(" ", "-").lower() for c in categories]

    payload = {
        "user_id": user_id,
        "content": content,
        "category": category,
        "tags": tags,
    }

    return post_json(f"{SELFMEM_BASE}/api/v1/memories", payload, SELFMEM_API_KEY)


def main():
    users = get_users()
    print(f"Found {len(users)} users in OpenMemory")

    total_imported = 0
    total_skipped = 0

    for user in users:
        user_id = user["user_id"]
        count = user["memory_count"]
        print(f"\n--- {user_id} ({count} memories) ---")

        memories = get_memories(user_id)
        print(f"  Fetched {len(memories)} active memories")

        for i, mem in enumerate(memories):
            state = mem.get("state", "active")
            if state != "active":
                print(f"  [{i+1}] Skipped (state={state})")
                total_skipped += 1
                continue

            result = import_memory(user_id, mem)
            if result:
                print(f"  [{i+1}] Imported: {result['id']} ({result['category']})")
                total_imported += 1
            else:
                print(f"  [{i+1}] Skipped (empty content)")
                total_skipped += 1

    print(f"\n=== Done: {total_imported} imported, {total_skipped} skipped ===")


if __name__ == "__main__":
    main()
