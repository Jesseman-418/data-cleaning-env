"""
Inference script for the Data Cleaning Environment.

Runs an LLM agent against all 3 data cleaning tasks via the OpenEnv HTTP API.
Uses the OpenAI client with environment variables as required by the hackathon spec.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use for inference
    HF_TOKEN      — Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="your-key"
    python inference.py
"""

import asyncio
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

# ── Environment Variables (mandatory) ───────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
API_KEY = HF_TOKEN

# ── Environment Configuration ───────────────────────────────────────────────

ENV_URL = os.environ.get("ENV_URL", "https://jesse1811-data-cleaning-env.hf.space")
BENCHMARK = "data_cleaning_env"
TEMPERATURE = 0.0
MAX_TOKENS = 500

# Task configs: task_id -> (max_steps, max_total_reward)
TASKS = {
    "easy_format_standardization": {"max_steps": 30, "max_total_reward": 3.0},
    "medium_missing_and_typos": {"max_steps": 60, "max_total_reward": 5.0},
    "hard_full_pipeline": {"max_steps": 100, "max_total_reward": 8.0},
}

SUCCESS_SCORE_THRESHOLD = 0.3

# ── Structured Logging (mandatory format — matches sample spec exactly) ─────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_short = action[:200] if len(action) > 200 else action
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data cleaning agent. You receive messy tabular data and must clean it by issuing JSON actions.

Available actions:
- {"action_type": "fix_field", "record_id": <int>, "field_name": "<field>", "new_value": "<value>"}
- {"action_type": "mark_duplicate", "record_id": <int>, "duplicate_of": <int>}
- {"action_type": "delete_record", "record_id": <int>}
- {"action_type": "submit"}

Cleaning rules:
- Dates: YYYY-MM-DD format
- Phone numbers: XXX-XXX-XXXX format (10 digits, no country code)
- Emails: all lowercase
- State names: full name (e.g., "Maharashtra" not "MH" or "Maharshtra")
- Company names: fix typos (e.g., "Tata Consultany Services" -> "Tata Consultancy Services")
- Missing fields: infer from context when possible
- Duplicates: same person with slightly different data — mark_duplicate or delete_record
- Outliers: fix impossible dates, invalid zip codes

Strategy:
1. First scan all records and fix obvious formatting issues (dates, phones, emails)
2. Then fix typos in state and company names
3. Then look for duplicates (similar names, same email pattern, etc.)
4. Submit when done — don't waste actions

Respond with ONLY a single JSON action object. No explanation, no markdown, no extra text."""


# ── LLM Interaction ─────────────────────────────────────────────────────────


def build_user_prompt(obs: Dict, history: List[str]) -> str:
    records_str = json.dumps(obs.get("records", []), indent=2)
    history_block = "\n".join(history[-10:]) if history else "(none)"

    return (
        f"Task: {obs.get('task_description', '')}\n"
        f"Difficulty: {obs.get('difficulty', '')}\n"
        f"Total issues: {obs.get('total_issues', 0)}\n"
        f"Issues fixed: {obs.get('issues_fixed', 0)}\n"
        f"Actions taken: {obs.get('actions_taken', 0)}/{obs.get('max_actions', 50)}\n"
        f"Current score: {obs.get('current_score', 0.0):.4f}\n"
        f"Last result: {obs.get('last_action_result', '')}\n\n"
        f"Previous steps:\n{history_block}\n\n"
        f"Records:\n{records_str}\n\n"
        f"Respond with a single JSON action."
    )


def get_model_action(client: OpenAI, obs: Dict, history: List[str]) -> str:
    user_prompt = build_user_prompt(obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"action_type": "submit"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "submit"}'


def parse_action(content: str) -> Optional[Dict]:
    """Parse a JSON action from LLM output."""
    content = content.strip()

    # Remove markdown code block if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    match = re.search(r"\{[^}]+\}", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Environment HTTP Client ─────────────────────────────────────────────────


class EnvHTTPClient:
    """HTTP client for the OpenEnv environment with retry logic."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def _request_with_retry(self, method: str, path: str, **kwargs) -> Dict:
        """Make an HTTP request with retries for transient failures."""
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(5):
            try:
                if method == "GET":
                    resp = await self.client.get(url, **kwargs)
                else:
                    resp = await self.client.post(url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                wait = min(2 ** attempt * 2, 30)
                print(f"[DEBUG] Attempt {attempt+1} failed ({type(exc).__name__}), retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:
                    last_exc = exc
                    wait = min(2 ** attempt * 2, 30)
                    print(f"[DEBUG] Server error {exc.response.status_code}, retrying in {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

    async def health_check(self) -> bool:
        """Check if the environment server is reachable."""
        try:
            resp = await self.client.get(f"{self.base_url}/health", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def wait_for_server(self, max_wait: int = 120) -> bool:
        """Wait for the environment server to become available."""
        print(f"[DEBUG] Waiting for environment at {self.base_url}...", flush=True)
        start = time.time()
        while time.time() - start < max_wait:
            if await self.health_check():
                print(f"[DEBUG] Environment is ready ({time.time()-start:.1f}s)", flush=True)
                return True
            await asyncio.sleep(3)
        print(f"[DEBUG] Environment not ready after {max_wait}s", flush=True)
        return False

    async def reset(self, task_id: str = "easy_format_standardization") -> Dict:
        return await self._request_with_retry("POST", "/reset", json={"task_id": task_id})

    async def step(self, action: Dict) -> Dict:
        return await self._request_with_retry("POST", "/step", json={"action": action})

    async def close(self):
        await self.client.aclose()


# ── Main Runner ─────────────────────────────────────────────────────────────


async def run_task(client: OpenAI, env: EnvHTTPClient, task_id: str) -> Dict:
    """Run a single task and return results."""
    task_cfg = TASKS[task_id]
    max_steps = task_cfg["max_steps"]
    max_total_reward = task_cfg["max_total_reward"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.get("observation", result)
        done = result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            # Get action from LLM
            action_text = get_model_action(client, obs, history)
            action = parse_action(action_text)

            if action is None:
                action = {"action_type": "submit"}
                action_text = '{"action_type": "submit"}'

            # Execute action
            try:
                result = await env.step(action)
            except Exception as exc:
                print(f"[DEBUG] env.step failed: {exc}", flush=True)
                log_step(step=step, action=action_text, reward=0.0, done=True, error=str(exc))
                steps_taken = step
                rewards.append(0.0)
                break

            obs = result.get("observation", result)
            reward = result.get("reward", 0.0) or 0.0
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_text, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_text} -> reward {reward:+.2f}, "
                f"result: {str(obs.get('last_action_result', ''))[:100]}"
            )

            if done:
                break

        # Submit if not done
        if not done:
            action = {"action_type": "submit"}
            try:
                result = await env.step(action)
                obs = result.get("observation", result)
                reward = result.get("reward", 0.0) or 0.0
                done = result.get("done", False)
            except Exception as exc:
                print(f"[DEBUG] Final submit failed: {exc}", flush=True)
                reward = 0.0
                done = True
            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action='{"action_type": "submit"}', reward=reward, done=done, error=None)

        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.001
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]
        steps_taken = max(steps_taken, 1)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


async def main() -> None:
    if not API_KEY:
        print("[DEBUG] WARNING: No API key found (HF_TOKEN / API_KEY not set)", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    env = EnvHTTPClient(ENV_URL)

    print("=" * 60, flush=True)
    print("Data Cleaning Environment — Inference Script", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API Base URL: {API_BASE_URL}", flush=True)
    print(f"Environment URL: {ENV_URL}", flush=True)
    print("=" * 60, flush=True)

    # Wait for environment to be ready (HF Spaces can be sleeping)
    server_ready = await env.wait_for_server(max_wait=120)
    if not server_ready:
        print("[DEBUG] WARNING: Environment health check failed, attempting anyway...", flush=True)

    all_results = []

    try:
        for task_id in TASKS:
            print(f"\n--- Running task: {task_id} ---", flush=True)
            try:
                result = await run_task(client, env, task_id)
            except Exception as exc:
                print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)
                result = {"task_id": task_id, "score": 0.001, "steps": 0, "success": False, "rewards": [0.001]}
                log_end(success=False, steps=0, score=0.001, rewards=[0.001])
            all_results.append(result)
            print(f"    Score: {result['score']:.4f} | Steps: {result['steps']} | Success: {result['success']}", flush=True)
    finally:
        await env.close()

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        print(f"  {r['task_id']}: score={r['score']:.4f}, steps={r['steps']}, success={r['success']}", flush=True)

    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[DEBUG] Interrupted", flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        # Still exit 0 so the evaluator can parse whatever logs we emitted
        sys.exit(0)
