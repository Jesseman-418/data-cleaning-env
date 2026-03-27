"""Data Cleaning Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class DataCleaningEnv(
    EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]
):
    """
    Client for the Data Cleaning Environment.

    Example:
        >>> with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.task_description)
        ...     result = env.step(DataCleaningAction(
        ...         action_type="fix_field",
        ...         record_id=1,
        ...         field_name="email",
        ...         new_value="rahul.sharma@gmail.com"
        ...     ))
        ...     print(result.observation.last_action_result)
    """

    def _step_payload(self, action: DataCleaningAction) -> Dict:
        payload = {"action_type": action.action_type}
        if action.record_id is not None:
            payload["record_id"] = action.record_id
        if action.field_name is not None:
            payload["field_name"] = action.field_name
        if action.new_value is not None:
            payload["new_value"] = action.new_value
        if action.duplicate_of is not None:
            payload["duplicate_of"] = action.duplicate_of
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DataCleaningObservation]:
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
            records=obs_data.get("records", []),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            total_issues=obs_data.get("total_issues", 0),
            issues_fixed=obs_data.get("issues_fixed", 0),
            actions_taken=obs_data.get("actions_taken", 0),
            max_actions=obs_data.get("max_actions", 50),
            last_action_result=obs_data.get("last_action_result", ""),
            current_score=obs_data.get("current_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DataCleaningState:
        return DataCleaningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            total_issues=payload.get("total_issues", 0),
            issues_fixed=payload.get("issues_fixed", 0),
            actions_taken=payload.get("actions_taken", 0),
            max_actions=payload.get("max_actions", 50),
            current_score=payload.get("current_score", 0.0),
            duplicates_found=payload.get("duplicates_found", 0),
            is_submitted=payload.get("is_submitted", False),
        )
