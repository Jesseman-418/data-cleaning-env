---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Data Cleaning Environment

A **real-world** OpenEnv environment where AI agents learn to clean messy tabular data. The agent receives dirty customer records with formatting errors, missing values, typos, outliers, and duplicate entries — and must fix them through a series of actions.

**Why this matters:** Data cleaning consumes ~80% of data scientists' time. This environment provides a standardized benchmark for training and evaluating AI agents on this critical task.

**Live Demo:** [https://jesse1811-data-cleaning-env.hf.space/web/](https://jesse1811-data-cleaning-env.hf.space/web/)

---

## Environment Description & Motivation

Data cleaning is one of the most common, time-consuming tasks in any data pipeline. This environment simulates realistic data quality issues found in customer/contact databases:

- **Format inconsistencies**: Dates in mixed formats (MM/DD/YYYY, DD.MM.YYYY, etc.), phone numbers without standard formatting, mixed-case emails
- **Missing values**: Empty fields that can be inferred from context (e.g., city from pin code)
- **Typos**: Misspelled state names ("Maharshtra"), company names ("Tata Consultany Services")
- **Outliers**: Impossible dates (birth year 1820), invalid zip codes
- **Duplicates**: Same person appearing multiple times with slight variations

The agent must diagnose and fix these issues efficiently, providing a rich signal for RL training.

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fix_field` | `record_id`, `field_name`, `new_value` | Correct a specific field in a record |
| `mark_duplicate` | `record_id`, `duplicate_of` | Flag two records as representing the same entity |
| `delete_record` | `record_id` | Remove a record from the dataset |
| `submit` | — | Finalize cleaning and get graded |

```python
DataCleaningAction(
    action_type="fix_field",
    record_id=1,
    field_name="date_of_birth",
    new_value="1990-03-15"
)
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `records` | `List[Dict]` | Current state of all records |
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | What needs to be cleaned |
| `difficulty` | `str` | easy / medium / hard |
| `total_issues` | `int` | Total issues in the dataset |
| `issues_fixed` | `int` | Issues correctly fixed so far |
| `actions_taken` | `int` | Actions used |
| `max_actions` | `int` | Action budget |
| `last_action_result` | `str` | Feedback from last action |
| `current_score` | `float` | Running score (0.0-1.0) |

Each record has fields: `id`, `name`, `email`, `phone`, `date_of_birth`, `city`, `state`, `zip_code`, `company`.

---

## Tasks

### Task 1: Format Standardization (Easy)
- **Records:** 5 | **Issues:** ~15 | **Action budget:** 30
- **Goal:** Standardize dates to YYYY-MM-DD, phones to XXX-XXX-XXXX, emails to lowercase
- **Challenge:** Straightforward pattern matching

### Task 2: Missing Values & Typo Correction (Medium)
- **Records:** 10 | **Issues:** ~31 | **Action budget:** 60
- **Goal:** Fill missing values, fix typos, standardize formats
- **Challenge:** Requires contextual reasoning — inferring city from pin code, recognizing misspelled state names

### Task 3: Full Data Pipeline (Hard)
- **Records:** 15 (including 3 duplicates) | **Issues:** ~45 | **Action budget:** 100
- **Goal:** Fix all issues, identify and remove duplicate records, handle outliers
- **Challenge:** Genuinely challenges frontier models — requires entity resolution, anomaly detection, and multi-step reasoning

---

## Reward Function

The reward provides **dense, partial progress signals** throughout the episode:

| Action | Reward |
|--------|--------|
| Correct field fix | +0.1 |
| Incorrect field change | -0.05 |
| Correct duplicate identification | +0.2 |
| Incorrect duplicate marking | -0.1 |
| Correct duplicate deletion | +0.15 |
| Deleting non-duplicate record | -0.15 |
| Invalid action | -0.02 |
| Submit (final score) | 0.0-1.0 |

**Final grading weights:**
- **Easy/Medium:** Field accuracy (75%) + Efficiency (15%) + No false positives (10%)
- **Hard:** Field accuracy (60%) + Duplicate detection (25%) + Efficiency (10%) + No false positives (5%)

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)

### Install
```bash
pip install openenv-core[core]
git clone <this-repo>
cd data_cleaning_env
pip install -e .
```

### Run Locally
```bash
# Start the server (includes web UI at /web)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn server.app:app --reload --port 8000
```

### Docker
```bash
docker build -t data-cleaning-env:latest -f server/Dockerfile .
docker run -p 8000:8000 data-cleaning-env:latest
```

### Deploy to Hugging Face Spaces
```bash
openenv push --repo-id your-username/data-cleaning-env
```

---

## Quick Start

### Python Client
```python
from data_cleaning_env import DataCleaningEnv, DataCleaningAction

with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    # Start easy task
    result = env.reset(task_id="easy_format_standardization")
    print(result.observation.task_description)

    # Fix a date
    result = env.step(DataCleaningAction(
        action_type="fix_field",
        record_id=1,
        field_name="date_of_birth",
        new_value="1990-03-15"
    ))
    print(result.observation.last_action_result)

    # Submit when done
    result = env.step(DataCleaningAction(action_type="submit"))
    print(f"Score: {result.observation.current_score}")
```

### WebSocket (Persistent Sessions)
```python
import asyncio, json, websockets

async def play_episode():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset"}))
        resp = json.loads(await ws.recv())
        records = resp["data"]["observation"]["records"]
        print(f"Loaded {len(records)} records")

        # Fix a field
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "action_type": "fix_field",
                "record_id": 1,
                "field_name": "email",
                "new_value": "rahul.sharma@gmail.com"
            }
        }))
        resp = json.loads(await ws.recv())
        print(resp["data"]["observation"]["last_action_result"])

        # Submit
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "submit"}
        }))
        resp = json.loads(await ws.recv())
        print(f"Score: {resp['data']['observation']['current_score']}")

asyncio.run(play_episode())
```

### cURL (Quick Test)
```bash
# Health check
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Grading criteria
curl http://localhost:8000/grader

# Run heuristic baseline on all tasks
curl -X POST http://localhost:8000/baseline
```

---

## Sample Test Output

### 1. Health Check
```
$ curl http://localhost:8000/health
{"status":"healthy"}
```

### 2. List Tasks
```
$ curl http://localhost:8000/tasks | python -m json.tool
{
    "tasks": [
        {
            "id": "easy_format_standardization",
            "name": "Format Standardization",
            "difficulty": "easy",
            "description": "Fix formatting issues in 5 customer records...",
            "max_actions": 30,
            "fields": ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"]
        },
        {
            "id": "medium_missing_and_typos",
            "name": "Missing Values & Typo Correction",
            "difficulty": "medium",
            ...
        },
        {
            "id": "hard_full_pipeline",
            "name": "Full Data Pipeline",
            "difficulty": "hard",
            ...
        }
    ],
    "action_schema": { ... }
}
```

### 3. Full Episode via WebSocket
```
$ python test_episode.py

=== RESET ===
Task: easy_format_standardization, Records: 5, Issues: 15
Record 1: {'id': 1, 'name': 'Rahul Sharma', 'email': 'RAHUL.SHARMA@gmail.com',
 'phone': '982.314.5670', 'date_of_birth': '03/15/1990', 'city': 'Mumbai',
 'state': 'Maharashtra', 'zip_code': '400001', 'company': 'Tata Consultancy Services'}

=== FIX EMAIL ===
Fixed record 1, field 'email': 'RAHUL.SHARMA@gmail.com' -> 'rahul.sharma@gmail.com' (correct)
Score: 0.495, Reward: 0.1

=== FIX PHONE ===
Fixed record 1, field 'phone': '982.314.5670' -> '982-314-5670' (correct)
Score: 0.54

=== FIX DATE ===
Fixed record 1, field 'date_of_birth': '03/15/1990' -> '1990-03-15' (correct)
Score: 0.585

=== SUBMIT ===
Submitted! Final score: 0.5800 | Field accuracy: 46.67% (7/15) |
 Duplicates: 0/0 | Efficiency: 86.67% | False positives: 0
Final Score: 0.58
Done: True
```

### 4. Heuristic Baseline (All Tasks)
```
$ curl -X POST http://localhost:8000/baseline | python -m json.tool
{
    "baseline_scores": [
        {
            "task_id": "easy_format_standardization",
            "score": 0.87,
            "actions_taken": 16,
            "last_result": "Submitted! Final score: 0.8700 | Field accuracy: 93.33% (14/15) | Duplicates: 0/0 | Efficiency: 46.67% | False positives: 0"
        },
        {
            "task_id": "medium_missing_and_typos",
            "score": 0.4427,
            "actions_taken": 10,
            "last_result": "Submitted! Final score: 0.4427 | Field accuracy: 29.03% (9/31) | Duplicates: 0/0 | Efficiency: 83.33% | False positives: 0"
        },
        {
            "task_id": "hard_full_pipeline",
            "score": 0.3709,
            "actions_taken": 22,
            "last_result": "Submitted! Final score: 0.3709 | Field accuracy: 40.48% (17/42) | Duplicates: 0/3 | Efficiency: 78.00% | False positives: 0"
        }
    ]
}
```

### 5. Grader Details
```
$ curl http://localhost:8000/grader | python -m json.tool
{
    "description": "Grader scores episodes from 0.0 to 1.0",
    "scoring": {
        "field_accuracy": "Percentage of dirty fields correctly fixed (0.0-1.0)",
        "duplicate_accuracy": "Percentage of duplicates correctly identified (0.0-1.0, hard task only)",
        "efficiency": "1.0 - (actions_used / max_actions), rewards fewer actions",
        "false_positive_penalty": "Penalty for incorrectly modifying clean fields (max 0.3)"
    },
    "weights": {
        "easy_medium": { "field_accuracy": 0.75, "efficiency": 0.15, "false_positive_penalty": 0.10 },
        "hard": { "field_accuracy": 0.60, "duplicate_accuracy": 0.25, "efficiency": 0.10, "false_positive_penalty": 0.05 }
    }
}
```

---

## Baseline Inference

### Heuristic Baseline (no API key needed)
```bash
python -m server.baseline_runner
```

### LLM Baseline (requires OpenAI API key)
```bash
export OPENAI_API_KEY="your-key"
python baseline_inference.py --model gpt-4o-mini
```

### Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | ~0.10 | ~0.05 | ~0.03 |
| Heuristic (regex only) | **0.87** | **0.44** | **0.37** |
| GPT-4o-mini | ~0.85 | ~0.70 | ~0.55 |
| Frontier LLM | ~0.95 | ~0.85 | ~0.70 |

The gap between heuristic and LLM performance on Medium/Hard tasks demonstrates that this environment **genuinely requires AI reasoning** — not just pattern matching.

---

## Interactive Web Interface

The environment includes a professional web UI at `/web` with four tabs:

| Tab | Description |
|-----|-------------|
| **Playground** | Interactive data table, action controls, real-time metrics, score progress chart |
| **Baselines & Training** | Run heuristic baseline, RL training simulation with learning curves |
| **Documentation** | Full API docs, connection examples, reward tables |
| **Grading Details** | Scoring formulas, component weights, difficulty analysis |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (accepts `task_id` parameter) |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | GET | Grading criteria and weights |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/ws` | WS | WebSocket for persistent sessions |
| `/web` | GET | Interactive web interface |
| `/docs` | GET | OpenAPI / Swagger documentation |

---

## Project Structure

```
data_cleaning_env/
├── __init__.py                 # Module exports
├── models.py                   # Action, Observation, State (Pydantic models)
├── client.py                   # DataCleaningEnv client (WebSocket)
├── data_generator.py           # Deterministic dirty data generation (seeded)
├── grader.py                   # Episode scoring (0.0-1.0, multi-dimensional)
├── baseline_inference.py       # LLM baseline using OpenAI API
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Dependencies
├── README.md                   # This file
└── server/
    ├── __init__.py
    ├── data_cleaning_env_environment.py  # Core environment (reset/step/state)
    ├── app.py                  # FastAPI app + custom endpoints
    ├── web_ui.py               # Professional Gradio web interface
    ├── baseline_runner.py      # Heuristic baseline agent
    ├── requirements.txt        # Server dependencies
    └── Dockerfile              # Container definition
```

---

## What Makes This Environment Unique

1. **Real-world task**: Data cleaning is a genuine industry problem, not a toy game
2. **Dense reward signal**: Every action gets immediate feedback, enabling effective RL training
3. **Progressive difficulty**: Easy (regex) -> Medium (reasoning) -> Hard (entity resolution) — clear skill ladder
4. **Multi-dimensional grading**: Accuracy + efficiency + false positive penalty rewards careful agents
5. **Deterministic generation**: Seeded data allows reproducible benchmarking across runs
6. **Interactive demo**: Judges and users can play the environment directly in the browser
7. **RL training curves**: Built-in simulation showing progressive skill acquisition over episodes

---

## Authors

**Team Devgods** — Scaler x Meta PyTorch OpenEnv Hackathon 2026
- Jesseman Devamirtham N (Team Lead)
- Karen Infanta Rozario
- Janani S
