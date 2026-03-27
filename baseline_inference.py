"""
Baseline inference script using the OpenAI API client.

Runs an LLM agent against all 3 data cleaning tasks and reports scores.
Reads API credentials from OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-key"
    python baseline_inference.py

    # Or run against a remote server:
    python baseline_inference.py --base-url https://your-space.hf.space
"""

import argparse
import json
import os
import sys

from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import TASKS
from models import DataCleaningAction
from server.data_cleaning_env_environment import DataCleaningEnvironment


SYSTEM_PROMPT = """You are a data cleaning agent. You are given a messy dataset and must clean it.

Available actions (respond with JSON):
- {"action_type": "fix_field", "record_id": <int>, "field_name": "<field>", "new_value": "<value>"}
- {"action_type": "mark_duplicate", "record_id": <int>, "duplicate_of": <int>}
- {"action_type": "delete_record", "record_id": <int>}
- {"action_type": "submit"}

Rules:
- Dates should be in YYYY-MM-DD format
- Phone numbers should be in XXX-XXX-XXXX format
- Emails should be lowercase
- Fix typos in state names (use full state name, e.g., "Maharashtra" not "MH")
- Fix typos in company names
- Fill in missing fields when you can infer the correct value
- Identify and mark/delete duplicate records (same person, slightly different data)
- When done fixing, use the submit action

Respond with ONLY a JSON action object, no other text."""


def run_llm_agent(task_id: str, model: str = "gpt-4o-mini", seed: int = 42) -> dict:
    """Run an LLM-based agent on a single task."""
    client = OpenAI()
    env = DataCleaningEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]

    total_reward = 0.0
    max_steps = obs.max_actions

    for step in range(max_steps):
        if obs.done:
            break

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()

            # Parse action from LLM response
            action = parse_action(content)
            if action is None:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Invalid action format. Please respond with a valid JSON action.",
                })
                continue

            obs = env.step(action)
            total_reward += obs.reward if obs.reward else 0.0

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": format_observation(obs)})

        except Exception as e:
            print(f"  Error at step {step}: {e}")
            break

    # Submit if not already done
    if not obs.done:
        obs = env.step(DataCleaningAction(action_type="submit"))
        total_reward += obs.reward if obs.reward else 0.0

    return {
        "task_id": task_id,
        "score": obs.current_score,
        "total_reward": round(total_reward, 4),
        "actions_taken": obs.actions_taken,
        "last_result": obs.last_action_result,
    }


def format_observation(obs) -> str:
    """Format observation as a string for the LLM."""
    records_str = json.dumps(obs.records, indent=2)
    return (
        f"Task: {obs.task_description}\n"
        f"Difficulty: {obs.difficulty}\n"
        f"Total issues: {obs.total_issues}\n"
        f"Issues fixed: {obs.issues_fixed}\n"
        f"Actions taken: {obs.actions_taken}/{obs.max_actions}\n"
        f"Current score: {obs.current_score:.4f}\n"
        f"Last result: {obs.last_action_result}\n\n"
        f"Records:\n{records_str}\n\n"
        f"What action do you want to take? Respond with JSON."
    )


def parse_action(content: str):
    """Parse an action from LLM output."""
    # Try to extract JSON from the response
    content = content.strip()

    # Remove markdown code block if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        content = content.strip()

    try:
        data = json.loads(content)
        return DataCleaningAction(**data)
    except (json.JSONDecodeError, Exception):
        # Try to find JSON in the response
        import re
        match = re.search(r"\{[^}]+\}", content)
        if match:
            try:
                data = json.loads(match.group())
                return DataCleaningAction(**data)
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Data Cleaning Environment")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--task", default=None, help="Specific task ID to run (runs all if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Running heuristic baseline instead...")
        from server.baseline_runner import run_baseline_all_tasks
        results = run_baseline_all_tasks()
        for r in results:
            print(f"\nTask: {r['task_id']}")
            print(f"  Score: {r['score']:.4f}")
            print(f"  Actions: {r['actions_taken']}")
        return

    task_ids = [args.task] if args.task else list(TASKS.keys())

    print("=" * 60)
    print("Data Cleaning Environment - Baseline Inference")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    results = []
    for task_id in task_ids:
        print(f"\nRunning task: {task_id}...")
        result = run_llm_agent(task_id, model=args.model, seed=args.seed)
        results.append(result)
        print(f"  Score: {result['score']:.4f}")
        print(f"  Actions: {result['actions_taken']}")
        print(f"  Total reward: {result['total_reward']:.4f}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id']}: {r['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average: {avg_score:.4f}")


if __name__ == "__main__":
    main()
