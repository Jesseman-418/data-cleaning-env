"""
Data Cleaning Environment Implementation.

A real-world environment where an AI agent must clean messy tabular data
through a series of actions: fixing fields, identifying duplicates, and
removing bad records.
"""

import copy
from typing import Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
    from data_generator import TASKS
    from grader import grade_episode
except ImportError:
    from ..models import DataCleaningAction, DataCleaningObservation, DataCleaningState
    from ..data_generator import TASKS
    from ..grader import grade_episode


class DataCleaningEnvironment(Environment):
    """
    Data Cleaning Environment.

    The agent receives a messy dataset and must clean it by fixing formatting
    issues, filling missing values, correcting typos, identifying duplicates,
    and removing outliers.

    Three tasks with increasing difficulty:
    - Easy: Format standardization (dates, phones, emails)
    - Medium: Missing values + typo correction
    - Hard: Full pipeline including duplicate detection
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = DataCleaningState(episode_id=str(uuid4()), step_count=0)
        self._current_task = None
        self._dirty_records = []
        self._clean_records = []
        self._issue_map = {}
        self._working_records = []
        self._marked_duplicates: Set[Tuple[int, int]] = set()
        self._deleted_records: Set[int] = set()
        self._last_grade = None

    def reset(self, seed=None, task_id=None, **kwargs) -> DataCleaningObservation:
        """Reset the environment with a specific task."""
        # Default to easy task
        if task_id is None:
            task_id = "easy_format_standardization"

        if task_id not in TASKS:
            task_id = "easy_format_standardization"

        task = TASKS[task_id]
        task_seed = seed if seed is not None else 42

        # Generate data
        dirty, clean, issue_map = task["generator"](seed=task_seed)

        self._current_task = task
        self._dirty_records = dirty
        self._clean_records = clean
        self._issue_map = issue_map
        self._working_records = copy.deepcopy(dirty)
        self._marked_duplicates = set()
        self._deleted_records = set()
        self._last_grade = None

        self._state = DataCleaningState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task["difficulty"],
            total_issues=issue_map["total"],
            issues_fixed=0,
            actions_taken=0,
            max_actions=task["max_actions"],
            current_score=0.0,
            duplicates_found=0,
            is_submitted=False,
        )

        return DataCleaningObservation(
            records=self._visible_records(),
            task_id=task_id,
            task_description=task["description"],
            difficulty=task["difficulty"],
            total_issues=issue_map["total"],
            issues_fixed=0,
            actions_taken=0,
            max_actions=task["max_actions"],
            last_action_result=f"Task '{task['name']}' loaded with {len(dirty)} records. {issue_map['total']} issues to fix.",
            current_score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Execute a cleaning action."""
        self._state.step_count += 1
        self._state.actions_taken += 1

        action_type = action.action_type.lower().strip()

        # Check if already submitted
        if self._state.is_submitted:
            return self._make_obs(
                "Episode already submitted. Call reset() to start a new episode.",
                reward=-0.1,
                done=True,
            )

        # Check action budget
        if self._state.actions_taken > self._state.max_actions:
            # Auto-submit on budget exhaustion
            return self._do_submit(auto=True)

        if action_type == "fix_field":
            return self._do_fix_field(action)
        elif action_type == "mark_duplicate":
            return self._do_mark_duplicate(action)
        elif action_type == "delete_record":
            return self._do_delete_record(action)
        elif action_type == "submit":
            return self._do_submit()
        else:
            return self._make_obs(
                f"Unknown action type: '{action_type}'. "
                "Valid actions: fix_field, mark_duplicate, delete_record, submit",
                reward=-0.05,
            )

    def _do_fix_field(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Fix a field value in a record."""
        rid = action.record_id
        field = action.field_name
        new_value = action.new_value

        if rid is None or field is None or new_value is None:
            return self._make_obs(
                "fix_field requires record_id, field_name, and new_value.",
                reward=-0.02,
            )

        # Find the record
        rec = self._find_record(rid)
        if rec is None:
            return self._make_obs(
                f"Record with id={rid} not found.",
                reward=-0.02,
            )

        if field not in rec:
            valid = [k for k in rec if k != "id"]
            return self._make_obs(
                f"Field '{field}' not found. Valid fields: {', '.join(valid)}",
                reward=-0.02,
            )

        old_value = rec[field]
        rec[field] = new_value

        # Check if this fix is correct
        reward = 0.0
        issues = self._issue_map.get("issues", {})
        record_issues = issues.get(rid, [])

        was_correct_fix = False
        for issue_field, expected_val in record_issues:
            if issue_field == field:
                from grader import _normalize
                if _normalize(new_value) == _normalize(expected_val):
                    was_correct_fix = True
                    reward = 0.1  # Partial reward for correct fix
                    break

        if was_correct_fix:
            msg = f"Fixed record {rid}, field '{field}': '{old_value}' -> '{new_value}' (correct)"
        else:
            # Check if we made it worse
            clean_rec = next((r for r in self._clean_records if r["id"] == rid), None)
            if clean_rec and field in clean_rec:
                from grader import _normalize
                if _normalize(old_value) == _normalize(clean_rec[field]):
                    reward = -0.05  # Penalty for making clean data dirty
                    msg = f"Changed record {rid}, field '{field}': '{old_value}' -> '{new_value}' (was already correct!)"
                else:
                    msg = f"Changed record {rid}, field '{field}': '{old_value}' -> '{new_value}' (not the expected value)"
            else:
                msg = f"Changed record {rid}, field '{field}': '{old_value}' -> '{new_value}'"

        # Recalculate score
        self._update_score()

        return self._make_obs(msg, reward=reward)

    def _do_mark_duplicate(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Mark two records as duplicates."""
        rid = action.record_id
        dup_of = action.duplicate_of

        if rid is None or dup_of is None:
            return self._make_obs(
                "mark_duplicate requires record_id and duplicate_of.",
                reward=-0.02,
            )

        if self._find_record(rid) is None:
            return self._make_obs(f"Record {rid} not found.", reward=-0.02)
        if self._find_record(dup_of) is None:
            return self._make_obs(f"Record {dup_of} not found.", reward=-0.02)

        self._marked_duplicates.add((rid, dup_of))

        # Check if correct
        expected_dups = self._issue_map.get("duplicates", [])
        is_correct = any(
            (d[0] == rid and d[1] == dup_of) or (d[0] == dup_of and d[1] == rid)
            for d in expected_dups
        )

        if is_correct:
            reward = 0.2
            msg = f"Marked record {rid} as duplicate of {dup_of} (correct!)"
        else:
            reward = -0.1
            msg = f"Marked record {rid} as duplicate of {dup_of} (these are not duplicates)"

        self._update_score()
        return self._make_obs(msg, reward=reward)

    def _do_delete_record(self, action: DataCleaningAction) -> DataCleaningObservation:
        """Delete a record from the dataset."""
        rid = action.record_id

        if rid is None:
            return self._make_obs("delete_record requires record_id.", reward=-0.02)

        if self._find_record(rid) is None:
            return self._make_obs(f"Record {rid} not found.", reward=-0.02)

        self._deleted_records.add(rid)

        # Check if this was a duplicate that should be deleted
        expected_dup_ids = self._issue_map.get("duplicate_ids", [])
        if rid in expected_dup_ids:
            reward = 0.15
            msg = f"Deleted record {rid} (correctly identified as duplicate)"
        else:
            reward = -0.15
            msg = f"Deleted record {rid} (this was not a duplicate — data lost!)"

        self._update_score()
        return self._make_obs(msg, reward=reward)

    def _do_submit(self, auto=False) -> DataCleaningObservation:
        """Submit the cleaned dataset for grading."""
        self._state.is_submitted = True

        # Grade the episode
        result = grade_episode(
            final_records=self._visible_records(),
            clean_records=self._clean_records,
            issue_map=self._issue_map,
            actions_taken=self._state.actions_taken,
            max_actions=self._state.max_actions,
            marked_duplicates=self._marked_duplicates,
            deleted_records=self._deleted_records,
        )

        self._last_grade = result
        self._state.current_score = result["score"]

        prefix = "Auto-submitted (action budget exhausted). " if auto else "Submitted! "
        msg = (
            f"{prefix}Final score: {result['score']:.4f} | "
            f"Field accuracy: {result['field_accuracy']:.2%} "
            f"({result['fields_correct']}/{result['fields_total']}) | "
            f"Duplicates: {result.get('duplicates_found', 0)}/{result.get('duplicates_expected', 0)} | "
            f"Efficiency: {result['efficiency']:.2%} | "
            f"False positives: {result['false_positives']}"
        )

        return self._make_obs(msg, reward=result["score"], done=True)

    def _find_record(self, rid: int):
        """Find a record by ID in the working set."""
        if rid in self._deleted_records:
            return None
        for rec in self._working_records:
            if rec["id"] == rid:
                return rec
        return None

    def _visible_records(self):
        """Get records visible to the agent (excluding deleted)."""
        return [r for r in self._working_records if r["id"] not in self._deleted_records]

    def _update_score(self):
        """Recalculate the current partial score."""
        result = grade_episode(
            final_records=self._visible_records(),
            clean_records=self._clean_records,
            issue_map=self._issue_map,
            actions_taken=self._state.actions_taken,
            max_actions=self._state.max_actions,
            marked_duplicates=self._marked_duplicates,
            deleted_records=self._deleted_records,
        )
        self._state.current_score = result["score"]
        self._state.issues_fixed = result.get("fields_correct", 0)
        self._state.duplicates_found = result.get("duplicates_found", 0)

    def _make_obs(self, msg: str, reward: float = 0.0, done: bool = False) -> DataCleaningObservation:
        """Create an observation."""
        return DataCleaningObservation(
            records=self._visible_records(),
            task_id=self._state.task_id,
            task_description=self._current_task["description"] if self._current_task else "",
            difficulty=self._state.difficulty,
            total_issues=self._state.total_issues,
            issues_fixed=self._state.issues_fixed,
            actions_taken=self._state.actions_taken,
            max_actions=self._state.max_actions,
            last_action_result=msg,
            current_score=self._state.current_score,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> DataCleaningState:
        return self._state

    def get_last_grade(self):
        """Get the last grading result (used by /grader endpoint)."""
        return self._last_grade
