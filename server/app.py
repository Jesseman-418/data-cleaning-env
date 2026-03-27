"""
FastAPI application for the Data Cleaning Environment.

Exposes the standard OpenEnv endpoints plus hackathon-required:
  - GET /tasks: List tasks and action schema
  - POST /baseline: Run baseline inference and return scores
  - GET /grader: Return grader score after episode completion
  - GET /web: Professional interactive web interface
"""

import os
import sys

# Do NOT enable OpenEnv's default web interface — we mount our own Gradio app
# os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import DataCleaningAction, DataCleaningObservation
    from server.data_cleaning_env_environment import DataCleaningEnvironment
    from data_generator import TASKS
except ImportError:
    from ..models import DataCleaningAction, DataCleaningObservation
    from .data_cleaning_env_environment import DataCleaningEnvironment
    from ..data_generator import TASKS


# ── Create the base OpenEnv app (API only, no default Gradio) ────────────────
app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data_cleaning_env",
    max_concurrent_envs=10,
)

# ── Customise Swagger / OpenAPI docs ─────────────────────────────────────────
app.title = "Data Cleaning Environment API"
app.description = """
# Data Cleaning Environment

A real-world **OpenEnv** reinforcement learning environment where AI agents learn to clean messy tabular data.

Built by **Team Devgods** for the Scaler x Meta PyTorch OpenEnv Hackathon 2026.

---

## Quick Start

1. **Reset** an episode with a task:
   ```
   POST /reset  {"task_id": "easy_format_standardization"}
   ```
2. **Step** through actions:
   ```
   POST /step  {"action": {"action_type": "fix_field", "record_id": 1, "field_name": "email", "new_value": "john@gmail.com"}}
   ```
3. **Submit** when done:
   ```
   POST /step  {"action": {"action_type": "submit"}}
   ```

## Tasks

| Task | Difficulty | Records | Issues | Budget |
|------|-----------|---------|--------|--------|
| `easy_format_standardization` | Easy | 5 | ~15 | 30 |
| `medium_missing_and_typos` | Medium | 10 | ~31 | 60 |
| `hard_full_pipeline` | Hard | 15 | ~45 | 100 |

## Actions

| Action | Required Parameters |
|--------|-------------------|
| `fix_field` | `record_id`, `field_name`, `new_value` |
| `mark_duplicate` | `record_id`, `duplicate_of` |
| `delete_record` | `record_id` |
| `submit` | — |

## Links

- **Interactive UI**: [/web](/web)
- **WebSocket**: `ws://host/ws` (persistent sessions)
- **GitHub**: [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
"""
app.version = "1.0.0"
app.openapi_tags = [
    {"name": "Environment", "description": "Core reset / step / state operations"},
    {"name": "Hackathon", "description": "Task listing, grader details, baseline runner"},
    {"name": "Health", "description": "Service health checks"},
]
# Custom Swagger UI styling
app.swagger_ui_parameters = {
    "docExpansion": "list",
    "defaultModelsExpandDepth": 0,
    "syntaxHighlight.theme": "monokai",
    "tryItOutEnabled": True,
    "displayRequestDuration": True,
    "filter": True,
}

# Override OpenAPI schema to inject our custom title/description
_original_openapi = app.openapi

def custom_openapi():
    schema = _original_openapi()
    schema["info"]["title"] = "Data Cleaning Environment API"
    schema["info"]["version"] = "1.0.0"
    schema["info"]["description"] = app.description
    return schema

app.openapi = custom_openapi


# ── Mount our custom Gradio web UI at /web ───────────────────────────────────
import gradio as gr
from server.web_ui import build_custom_ui

_gradio_blocks = build_custom_ui()
app = gr.mount_gradio_app(app, _gradio_blocks, path="/web")


# ── Additional hackathon endpoints ───────────────────────────────────────────

@app.get("/tasks", tags=["Hackathon"])
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


@app.get("/grader", tags=["Hackathon"])
async def get_grader():
    """Return grader scoring criteria and weights."""
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


@app.post("/baseline", tags=["Hackathon"])
async def run_baseline():
    """Run heuristic baseline inference on all 3 tasks and return scores."""
    from server.baseline_runner import run_baseline_all_tasks
    results = run_baseline_all_tasks()
    return {"baseline_scores": results}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
