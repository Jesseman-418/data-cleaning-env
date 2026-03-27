"""
Baseline runner for the /baseline endpoint.

Runs a simple heuristic agent against all 3 tasks and returns scores.
This does NOT require an API key — it uses rule-based logic.
"""

import re

try:
    from ..data_generator import TASKS
    from ..models import DataCleaningAction
    from .data_cleaning_env_environment import DataCleaningEnvironment
except ImportError:
    from data_generator import TASKS
    from models import DataCleaningAction
    from server.data_cleaning_env_environment import DataCleaningEnvironment


def _standardize_date(date_str: str) -> str:
    """Try to convert various date formats to YYYY-MM-DD."""
    date_str = date_str.strip()

    # Already correct
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # MM/DD/YYYY or MM.DD.YYYY
    m = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$", date_str)
    if m:
        month, day, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # M-D-YYYY (no padding)
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", date_str)
    if m:
        month, day, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # Mon DD, YYYY
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    m = re.match(r"^([A-Za-z]{3})\s+(\d{1,2}),?\s+(\d{4})$", date_str)
    if m:
        mon, day, year = m.groups()
        month_num = months.get(mon.lower()[:3], "01")
        return f"{year}-{month_num}-{int(day):02d}"

    return date_str


def _standardize_phone(phone_str: str) -> str:
    """Convert to XXX-XXX-XXXX format."""
    digits = re.sub(r"\D", "", phone_str)
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return phone_str


def run_heuristic_agent(task_id: str, seed: int = 42) -> dict:
    """Run a heuristic baseline agent on a single task."""
    env = DataCleaningEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    records = obs.records

    for rec in records:
        rid = rec["id"]

        # Fix date format
        dob = rec.get("date_of_birth", "")
        if dob and not re.match(r"^\d{4}-\d{2}-\d{2}$", dob):
            fixed = _standardize_date(dob)
            if fixed != dob:
                obs = env.step(DataCleaningAction(
                    action_type="fix_field",
                    record_id=rid,
                    field_name="date_of_birth",
                    new_value=fixed,
                ))
                if obs.done:
                    break

        # Fix phone format
        phone = rec.get("phone", "")
        if phone and not re.match(r"^\d{3}-\d{3}-\d{4}$", phone):
            fixed = _standardize_phone(phone)
            if fixed != phone:
                obs = env.step(DataCleaningAction(
                    action_type="fix_field",
                    record_id=rid,
                    field_name="phone",
                    new_value=fixed,
                ))
                if obs.done:
                    break

        # Fix email case
        email = rec.get("email", "")
        if email and email != email.lower():
            obs = env.step(DataCleaningAction(
                action_type="fix_field",
                record_id=rid,
                field_name="email",
                new_value=email.lower(),
            ))
            if obs.done:
                break

    # Submit
    if not obs.done:
        obs = env.step(DataCleaningAction(action_type="submit"))

    return {
        "task_id": task_id,
        "score": obs.current_score,
        "actions_taken": obs.actions_taken,
        "last_result": obs.last_action_result,
    }


def run_baseline_all_tasks() -> list:
    """Run baseline on all tasks."""
    results = []
    for task_id in TASKS:
        result = run_heuristic_agent(task_id)
        results.append(result)
    return results


if __name__ == "__main__":
    results = run_baseline_all_tasks()
    for r in results:
        print(f"Task: {r['task_id']}")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Actions: {r['actions_taken']}")
        print(f"  Result: {r['last_result']}")
        print()
