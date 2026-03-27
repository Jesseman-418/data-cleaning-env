"""
FastAPI application for the Data Cleaning Environment.

Exposes the standard OpenEnv endpoints plus hackathon-required:
  - GET /tasks: List tasks and action schema
  - POST /baseline: Run baseline inference and return scores
  - GET /grader: Return grader score after episode completion
  - GET /web: Professional interactive web interface
"""

import os

# Enable the web interface
os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import DataCleaningAction, DataCleaningObservation
    from server.data_cleaning_env_environment import DataCleaningEnvironment
    from data_generator import TASKS
    from server.web_ui import build_custom_ui
except ImportError:
    from ..models import DataCleaningAction, DataCleaningObservation
    from .data_cleaning_env_environment import DataCleaningEnvironment
    from ..data_generator import TASKS
    from .web_ui import build_custom_ui


# Create the base OpenEnv app with custom Gradio UI
app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data_cleaning_env",
    max_concurrent_envs=10,
    gradio_builder=lambda wm, af, meta, is_chat, title, qs_md: build_custom_ui(),
)


# --- Additional hackathon endpoints ---

@app.get("/tasks")
async def get_tasks():
    """Return list of tasks and the action schema."""
    tasks = []
    for task_id, task in TASKS.items():
        tasks.append({
            "id": task["id"],
            "name": task["name"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "max_actions": task["max_actions"],
            "fields": task["fields"],
        })

    action_schema = {
        "action_type": {
            "type": "string",
            "enum": ["fix_field", "mark_duplicate", "delete_record", "submit"],
            "description": "Type of action to take",
        },
        "record_id": {
            "type": "integer",
            "description": "ID of the record to modify (required for fix_field, mark_duplicate, delete_record)",
        },
        "field_name": {
            "type": "string",
            "description": "Name of the field to fix (required for fix_field)",
            "enum": ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"],
        },
        "new_value": {
            "type": "string",
            "description": "New value for the field (required for fix_field)",
        },
        "duplicate_of": {
            "type": "integer",
            "description": "ID of the record this is a duplicate of (required for mark_duplicate)",
        },
    }

    return {
        "tasks": tasks,
        "action_schema": action_schema,
    }


@app.get("/grader")
async def get_grader():
    """Return grader score after an episode is completed."""
    return {
        "description": "Grader scores episodes from 0.0 to 1.0",
        "scoring": {
            "field_accuracy": "Percentage of dirty fields correctly fixed (0.0-1.0)",
            "duplicate_accuracy": "Percentage of duplicates correctly identified (0.0-1.0, hard task only)",
            "efficiency": "1.0 - (actions_used / max_actions), rewards fewer actions",
            "false_positive_penalty": "Penalty for incorrectly modifying clean fields (max 0.3)",
        },
        "weights": {
            "easy_medium": {
                "field_accuracy": 0.75,
                "efficiency": 0.15,
                "false_positive_penalty": 0.10,
            },
            "hard": {
                "field_accuracy": 0.60,
                "duplicate_accuracy": 0.25,
                "efficiency": 0.10,
                "false_positive_penalty": 0.05,
            },
        },
        "note": "Submit action during an episode to get your score. Score is returned in the observation.",
    }


@app.post("/baseline")
async def run_baseline():
    """Run baseline inference on all 3 tasks and return scores."""
    from server.baseline_runner import run_baseline_all_tasks
    results = run_baseline_all_tasks()
    return {"baseline_scores": results}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
