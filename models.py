"""
Data models for the Data Cleaning Environment.

Defines typed Action, Observation, and custom State models for a data cleaning
environment where an AI agent must fix messy tabular records.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class DataCleaningAction(Action):
    """Action the agent can take to clean data."""

    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'fix_field', 'mark_duplicate', "
            "'delete_record', or 'submit'"
        ),
    )
    record_id: Optional[int] = Field(
        default=None, description="ID of the record to modify"
    )
    field_name: Optional[str] = Field(
        default=None,
        description="Name of the field to fix (for fix_field action)",
    )
    new_value: Optional[str] = Field(
        default=None,
        description="New value for the field (for fix_field action)",
    )
    duplicate_of: Optional[int] = Field(
        default=None,
        description="ID of the record this is a duplicate of (for mark_duplicate)",
    )


class DataCleaningObservation(Observation):
    """Observation returned by the environment after each step."""

    records: List[Dict] = Field(
        default_factory=list,
        description="Current state of all records in the dataset",
    )
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(
        default="", description="Description of the cleaning task"
    )
    difficulty: str = Field(
        default="easy", description="Task difficulty: easy, medium, hard"
    )
    total_issues: int = Field(
        default=0, description="Total number of issues to fix in this dataset"
    )
    issues_fixed: int = Field(
        default=0, description="Number of issues correctly fixed so far"
    )
    actions_taken: int = Field(
        default=0, description="Number of actions taken so far"
    )
    max_actions: int = Field(
        default=50, description="Maximum actions allowed for this task"
    )
    last_action_result: str = Field(
        default="", description="Feedback from the last action taken"
    )
    current_score: float = Field(
        default=0.0, description="Current partial score (0.0-1.0)"
    )


class DataCleaningState(State):
    """Extended state for the data cleaning environment."""

    task_id: str = Field(default="", description="Current task ID")
    difficulty: str = Field(default="easy", description="Task difficulty")
    total_issues: int = Field(default=0, description="Total issues in dataset")
    issues_fixed: int = Field(default=0, description="Issues fixed so far")
    actions_taken: int = Field(default=0, description="Actions taken")
    max_actions: int = Field(default=50, description="Max actions allowed")
    current_score: float = Field(default=0.0, description="Current score")
    duplicates_found: int = Field(default=0, description="Duplicates correctly identified")
    is_submitted: bool = Field(default=False, description="Whether agent has submitted")
