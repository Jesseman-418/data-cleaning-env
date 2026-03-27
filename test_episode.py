#!/usr/bin/env python3
"""
Test script for the Data Cleaning Environment.

Run against the live HF deployment or a local server.

Usage:
    python test_episode.py                    # Test against HF Spaces
    python test_episode.py --local            # Test against localhost:8000
    python test_episode.py --url URL          # Test against a custom URL
"""

import asyncio
import json
import sys

# ── Configuration ────────────────────────────────────────────────────────────

HF_URL = "jesse1811-data-cleaning-env.hf.space"
LOCAL_URL = "localhost:8000"


def get_urls():
    if "--local" in sys.argv:
        base = LOCAL_URL
        ws_scheme = "ws"
        http_scheme = "http"
    elif "--url" in sys.argv:
        idx = sys.argv.index("--url") + 1
        base = sys.argv[idx].replace("https://", "").replace("http://", "").rstrip("/")
        ws_scheme = "wss" if "hf.space" in base else "ws"
        http_scheme = "https" if "hf.space" in base else "http"
    else:
        base = HF_URL
        ws_scheme = "wss"
        http_scheme = "https"
    return base, ws_scheme, http_scheme


# ── HTTP Tests ───────────────────────────────────────────────────────────────

def test_http_endpoints(base, scheme):
    import urllib.request

    print("=" * 60)
    print("HTTP ENDPOINT TESTS")
    print("=" * 60)

    endpoints = [
        ("GET", "/health", "Health Check"),
        ("GET", "/tasks", "List Tasks"),
        ("GET", "/grader", "Grading Criteria"),
    ]

    for method, path, name in endpoints:
        url = f"{scheme}://{base}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                print(f"\n[PASS] {name} ({path})")
                if path == "/health":
                    print(f"       Status: {data['status']}")
                elif path == "/tasks":
                    print(f"       Tasks: {len(data['tasks'])}")
                    for t in data["tasks"]:
                        print(f"         - {t['id']} ({t['difficulty']}): {t['max_actions']} max actions")
                elif path == "/grader":
                    print(f"       Description: {data['description']}")
        except Exception as e:
            print(f"\n[FAIL] {name} ({path}): {e}")

    # Test baseline
    print(f"\n[....] Running baseline (may take a few seconds)...")
    try:
        url = f"{scheme}://{base}/baseline"
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            print(f"[PASS] Baseline Results:")
            for r in data["baseline_scores"]:
                print(f"         {r['task_id']}: score={r['score']:.4f}, actions={r['actions_taken']}")
    except Exception as e:
        print(f"[FAIL] Baseline: {e}")

    # Test web UI
    try:
        url = f"{scheme}://{base}/web/"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            print(f"\n[PASS] Web UI (/web/)")
            print(f"       Content-Type: {content_type}")
            print(f"       Status: {resp.status}")
    except Exception as e:
        print(f"\n[FAIL] Web UI: {e}")


# ── WebSocket Episode Test ───────────────────────────────────────────────────

async def test_websocket_episode(base, ws_scheme):
    try:
        import websockets
    except ImportError:
        print("\n[SKIP] WebSocket test requires 'websockets' package")
        print("       Install with: pip install websockets")
        return

    print("\n" + "=" * 60)
    print("WEBSOCKET EPISODE TEST (Easy Task)")
    print("=" * 60)

    uri = f"{ws_scheme}://{base}/ws"
    try:
        async with websockets.connect(uri) as ws:
            # ── Reset ────────────────────────────────────────────────
            await ws.send(json.dumps({"type": "reset"}))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]

            print(f"\n[RESET] Task: {obs['task_id']}")
            print(f"        Records: {len(obs['records'])}")
            print(f"        Issues to fix: {obs['total_issues']}")
            print(f"        Action budget: {obs['max_actions']}")
            print(f"\n        Dirty records:")
            for rec in obs["records"]:
                print(f"          ID {rec['id']}: {rec['name']} | {rec['email']} | {rec['phone']} | {rec['date_of_birth']}")

            # ── Fix email (record 1) ─────────────────────────────────
            await ws.send(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "fix_field",
                    "record_id": 1,
                    "field_name": "email",
                    "new_value": "john.smith@gmail.com",
                },
            }))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]
            print(f"\n[FIX]   {obs['last_action_result']}")
            print(f"        Score: {obs['current_score']:.4f} | Reward: {r['data']['reward']}")

            # ── Fix phone (record 1) ─────────────────────────────────
            await ws.send(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "fix_field",
                    "record_id": 1,
                    "field_name": "phone",
                    "new_value": "555-123-4567",
                },
            }))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]
            print(f"\n[FIX]   {obs['last_action_result']}")
            print(f"        Score: {obs['current_score']:.4f}")

            # ── Fix date (record 1) ──────────────────────────────────
            await ws.send(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "fix_field",
                    "record_id": 1,
                    "field_name": "date_of_birth",
                    "new_value": "1990-03-15",
                },
            }))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]
            print(f"\n[FIX]   {obs['last_action_result']}")
            print(f"        Score: {obs['current_score']:.4f}")

            # ── Fix email (record 2) ─────────────────────────────────
            await ws.send(json.dumps({
                "type": "step",
                "data": {
                    "action_type": "fix_field",
                    "record_id": 2,
                    "field_name": "email",
                    "new_value": "sarah.j@outlook.com",
                },
            }))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]
            print(f"\n[FIX]   {obs['last_action_result']}")
            print(f"        Score: {obs['current_score']:.4f}")

            # ── Submit ───────────────────────────────────────────────
            await ws.send(json.dumps({
                "type": "step",
                "data": {"action_type": "submit"},
            }))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]
            done = r["data"]["done"]

            print(f"\n[SUBMIT] {obs['last_action_result']}")
            print(f"         Final Score: {obs['current_score']:.4f}")
            print(f"         Done: {done}")

            if done:
                print("\n[PASS] Episode completed successfully!")
            else:
                print("\n[WARN] Episode did not mark as done")

    except Exception as e:
        print(f"\n[FAIL] WebSocket error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    base, ws_scheme, http_scheme = get_urls()

    print(f"\nTesting against: {http_scheme}://{base}")
    print("=" * 60)

    test_http_endpoints(base, http_scheme)
    asyncio.run(test_websocket_episode(base, ws_scheme))

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print(f"\nWeb UI: {http_scheme}://{base}/web/")
    print(f"Swagger: {http_scheme}://{base}/docs")
    print(f"WebSocket: {ws_scheme}://{base}/ws")


if __name__ == "__main__":
    main()
