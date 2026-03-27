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

## Environment Description & Motivation

Data cleaning is one of the most common, time-consuming tasks in any data pipeline. This environment simulates realistic data quality issues found in customer/contact databases:

- **Format inconsistencies**: Dates in mixed formats (MM/DD/YYYY, DD.MM.YYYY, etc.), phone numbers without standard formatting, mixed-case emails
- **Missing values**: Empty fields that can be inferred from context (e.g., city from zip code)
- **Typos**: Misspelled state names ("Califronia"), company names ("Acme Corportation")
- **Outliers**: Impossible dates (birth year 1820), invalid zip codes
- **Duplicates**: Same person appearing multiple times with slight variations

The agent must diagnose and fix these issues efficiently, providing a rich signal for RL training.

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

## Tasks

### Task 1: Format Standardization (Easy)
- **Records:** 5
- **Issues:** ~15 (dates, phones, emails need formatting)
- **Action budget:** 30
- **Goal:** Standardize dates to YYYY-MM-DD, phones to XXX-XXX-XXXX, emails to lowercase
- **Expected difficulty:** Straightforward pattern matching

### Task 2: Missing Values & Typo Correction (Medium)
- **Records:** 10
- **Issues:** ~15-25 (missing fields, misspelled states/companies, format issues)
- **Action budget:** 60
- **Goal:** Fill missing values, fix typos, standardize formats
- **Expected difficulty:** Requires contextual reasoning for missing values

### Task 3: Full Data Pipeline (Hard)
- **Records:** 15 (including 3 duplicates)
- **Issues:** ~25-35 (all above + duplicate detection + outliers)
- **Action budget:** 100
- **Goal:** Fix all issues, identify and remove duplicate records, handle outliers
- **Expected difficulty:** Genuinely challenges frontier models — requires entity resolution, anomaly detection, and multi-step reasoning

## Reward Function

The reward provides **partial progress signals** throughout the episode:

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

Final grading weights:
- **Easy/Medium:** Field accuracy (75%) + Efficiency (15%) + No false positives (10%)
- **Hard:** Field accuracy (60%) + Duplicate detection (25%) + Efficiency (10%) + No false positives (5%)

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
# Start the server
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

## Quick Start

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

| Task | Heuristic | GPT-4o-mini |
|------|-----------|-------------|
| Easy (Format) | ~0.65 | ~0.85 |
| Medium (Missing + Typos) | ~0.35 | ~0.70 |
| Hard (Full Pipeline) | ~0.25 | ~0.55 |

*Scores are approximate and depend on seed.*

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
| `/docs` | GET | OpenAPI documentation |

## Project Structure

```
data_cleaning_env/
├── __init__.py                # Module exports
├── models.py                  # Action, Observation, State (Pydantic models)
├── client.py                  # DataCleaningEnv client (WebSocket)
├── data_generator.py          # Deterministic dirty data generation
├── grader.py                  # Episode scoring (0.0-1.0)
├── baseline_inference.py      # LLM baseline using OpenAI API
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml             # Dependencies
├── README.md                  # This file
└── server/
    ├── __init__.py
    ├── data_cleaning_env_environment.py  # Core environment logic
    ├── app.py                 # FastAPI app + /tasks, /grader, /baseline
    ├── baseline_runner.py     # Heuristic baseline agent
    ├── requirements.txt       # Server dependencies
    └── Dockerfile             # Container definition
```

## Authors

**Team Devgods** — Scaler x Meta PyTorch OpenEnv Hackathon 2026
- Jesseman Devamirtham N (Team Lead)
- Karen Infanta Rozario
- Janani S
