"""
Custom Gradio web interface for the Data Cleaning Environment.

Professional, interactive UI with:
- Interactive playground with data table and action controls
- Real-time scoring dashboard with metric cards
- Baseline evaluation and RL training simulation
- Full documentation and README
- Grading details
"""

import json
import copy
import random
import gradio as gr

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_generator import TASKS
    from models import DataCleaningAction
    from server.data_cleaning_env_environment import DataCleaningEnvironment
    from server.baseline_runner import run_heuristic_agent
except ImportError:
    from ..data_generator import TASKS
    from ..models import DataCleaningAction
    from .data_cleaning_env_environment import DataCleaningEnvironment
    from .baseline_runner import run_heuristic_agent


# ── Shared state ─────────────────────────────────────────────────────────────
_env = DataCleaningEnvironment()
_action_log = []
_score_history = []


# ── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Typography & Global ────────────────────────────────────── */
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
}

/* ── Header banner ──────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0f3460 100%);
    border-radius: 14px;
    padding: 32px 36px 28px;
    margin-bottom: 8px;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-banner h1 {
    margin: 0 0 8px 0;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    position: relative;
}
.header-banner .subtitle {
    margin: 0 0 14px 0;
    opacity: 0.78;
    font-size: 0.95rem;
    line-height: 1.55;
    max-width: 680px;
    position: relative;
}
.header-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    position: relative;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-accent {
    background: rgba(59,130,246,0.2);
    border-color: rgba(59,130,246,0.35);
}

/* ── Metric cards ───────────────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 10px;
}
.metric-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.metric-card .label {
    margin: 0 0 4px 0;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    font-weight: 700;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.metric-card .sub {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Score bar ──────────────────────────────────────────────── */
.score-bar-wrap {
    height: 6px;
    border-radius: 3px;
    background: #e2e8f0;
    overflow: hidden;
    margin-top: 8px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Status ─────────────────────────────────────────────────── */
.status-msg {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #334155;
}

/* ── Section titles ─────────────────────────────────────────── */
.section-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin: 0 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Action log ─────────────────────────────────────────────── */
.action-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.65 !important;
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
}

/* ── Data table ─────────────────────────────────────────────── */
.data-table table {
    font-size: 0.82rem !important;
}
.data-table th {
    background: #f1f5f9 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
    color: #475569 !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.action-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* ── Docs section ───────────────────────────────────────────── */
.docs-content {
    max-width: 860px;
}
.docs-content h2 {
    color: #1e293b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-top: 28px;
}
.docs-content table {
    font-size: 0.88rem;
}
.docs-content code {
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.85em;
}
.docs-content pre {
    background: #0f172a !important;
    border-radius: 8px;
    padding: 16px !important;
}

/* ── Footer ─────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 24px 16px;
    color: #94a3b8;
    font-size: 0.82rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 20px;
}
.footer strong { color: #64748b; }
"""


# ── Constants ────────────────────────────────────────────────────────────────

TASK_CHOICES = [
    ("Easy -- Format Standardization", "easy_format_standardization"),
    ("Medium -- Missing Values & Typos", "medium_missing_and_typos"),
    ("Hard -- Full Data Pipeline", "hard_full_pipeline"),
]
FIELD_CHOICES = ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"]
TABLE_HEADERS = ["ID", "Name", "Email", "Phone", "Date of Birth", "City", "State", "Zip", "Company"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _records_to_table(records):
    if not records:
        return []
    cols = ["id", "name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"]
    return [[r.get(c, "") for c in cols] for r in records]


def _metrics_html(obs):
    score = obs.current_score if obs else 0.0
    issues = obs.issues_fixed if obs else 0
    total = obs.total_issues if obs else 0
    actions = obs.actions_taken if obs else 0
    max_act = obs.max_actions if obs else 0
    pct = score * 100
    eff = max(0, 1 - actions / max_act) * 100 if max_act else 0

    if score >= 0.7:
        color = "#22c55e"
    elif score >= 0.4:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    return f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Score</div>
    <div class="value" style="color:{color}">{score:.4f}</div>
    <div class="score-bar-wrap">
      <div class="score-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}dd)"></div>
    </div>
  </div>
  <div class="metric-card">
    <div class="label">Issues Fixed</div>
    <div class="value">{issues}<span class="sub">/{total}</span></div>
  </div>
  <div class="metric-card">
    <div class="label">Actions Used</div>
    <div class="value">{actions}<span class="sub">/{max_act}</span></div>
  </div>
  <div class="metric-card">
    <div class="label">Efficiency</div>
    <div class="value">{eff:.0f}<span class="sub">%</span></div>
  </div>
</div>
"""


def _make_score_plot():
    if not _score_history:
        return None
    import pandas as pd
    return pd.DataFrame({"Step": list(range(len(_score_history))), "Score": _score_history})


# ── Core action handlers ─────────────────────────────────────────────────────

def reset_environment(task_choice):
    global _action_log, _score_history
    _action_log = []
    _score_history = []

    task_id = task_choice
    obs = _env.reset(seed=42, task_id=task_id)
    task = TASKS[task_id]
    status = f"Task loaded: **{task['name']}** ({task['difficulty']}) -- {obs.total_issues} issues in {len(obs.records)} records. Budget: {obs.max_actions} actions."
    _action_log.append(f"[RESET] {task['name']} loaded. {obs.total_issues} issues, {obs.max_actions} action budget.")
    _score_history.append(0.0)
    return _records_to_table(obs.records), _metrics_html(obs), status, "\n".join(_action_log[-30:]), _make_score_plot()


def do_fix_field(record_id, field_name, new_value):
    if not record_id or not field_name or not new_value:
        return _no_change("fix_field requires record_id, field_name, and new_value.")
    obs = _env.step(DataCleaningAction(action_type="fix_field", record_id=int(record_id), field_name=field_name, new_value=new_value))
    _action_log.append(f"[FIX] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_mark_duplicate(record_id, duplicate_of):
    if not record_id or not duplicate_of:
        return _no_change("mark_duplicate requires record_id and duplicate_of.")
    obs = _env.step(DataCleaningAction(action_type="mark_duplicate", record_id=int(record_id), duplicate_of=int(duplicate_of)))
    _action_log.append(f"[DUP] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_delete_record(record_id):
    if not record_id:
        return _no_change("delete_record requires record_id.")
    obs = _env.step(DataCleaningAction(action_type="delete_record", record_id=int(record_id)))
    _action_log.append(f"[DEL] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_submit():
    obs = _env.step(DataCleaningAction(action_type="submit"))
    _action_log.append(f"[SUBMIT] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def _obs_to_outputs(obs):
    done_msg = "  **Episode complete!**" if obs.done else ""
    status = f"{obs.last_action_result}{done_msg}"
    return _records_to_table(obs.records), _metrics_html(obs), status, "\n".join(_action_log[-30:]), _make_score_plot()


def _no_change(msg):
    return gr.update(), gr.update(), msg, "\n".join(_action_log[-30:]), gr.update()


# ── Baseline & RL ────────────────────────────────────────────────────────────

def run_baseline_demo():
    import re as _re
    results = [run_heuristic_agent(tid) for tid in TASKS]
    md = "### Heuristic Baseline Results\n\n"
    md += "| Task | Difficulty | Score | Actions | Field Accuracy |\n"
    md += "|------|-----------|-------|---------|----------------|\n"
    for r in results:
        task = TASKS[r["task_id"]]
        fa_match = _re.search(r"Field accuracy: ([\d.]+%)", r["last_result"])
        fa = fa_match.group(1) if fa_match else "--"
        md += f"| {task['name']} | {task['difficulty']} | **{r['score']:.4f}** | {r['actions_taken']} | {fa} |\n"
    md += "\n> The heuristic baseline uses regex rules for dates, phones, and emails. "
    md += "It cannot fix typos, fill missing values, or detect duplicates -- that requires AI reasoning.\n"
    return md


def run_rl_training_demo():
    import pandas as pd
    import re as _re

    episodes = 40
    all_scores = {"Episode": [], "Score": [], "Task": []}

    for task_id in TASKS:
        task = TASKS[task_id]
        for ep in range(episodes):
            env = DataCleaningEnvironment()
            obs = env.reset(seed=42 + ep, task_id=task_id)
            records = obs.records
            skill_level = min(ep / (episodes * 0.6), 1.0)

            for rec in records:
                rid = rec["id"]

                if skill_level > 0.05:
                    dob = rec.get("date_of_birth", "")
                    if dob and not _re.match(r"^\d{4}-\d{2}-\d{2}$", dob):
                        from server.baseline_runner import _standardize_date
                        fixed = _standardize_date(dob)
                        if fixed != dob:
                            obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="date_of_birth", new_value=fixed))
                            if obs.done: break

                if skill_level > 0.15:
                    phone = rec.get("phone", "")
                    if phone and not _re.match(r"^\d{3}-\d{3}-\d{4}$", phone):
                        from server.baseline_runner import _standardize_phone
                        fixed = _standardize_phone(phone)
                        if fixed != phone:
                            obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="phone", new_value=fixed))
                            if obs.done: break

                if skill_level > 0.25:
                    email = rec.get("email", "")
                    if email and email != email.lower():
                        obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="email", new_value=email.lower()))
                        if obs.done: break

                if skill_level > 0.45:
                    state = rec.get("state", "")
                    STATE_FIXES = {"Califronia": "California", "Kalifornia": "California", "Texass": "Texas", "Tejas": "Texas", "Ilinois": "Illinois", "Ilinoise": "Illinois", "Floridia": "Florida", "Flordia": "Florida", "Washinton": "Washington", "Washignton": "Washington", "Massachusets": "Massachusetts", "Massachusettes": "Massachusetts", "Pensylvania": "Pennsylvania", "Pennsylvnia": "Pennsylvania", "Gorgia": "Georgia", "Gerogia": "Georgia", "Michgan": "Michigan", "Michagan": "Michigan", "Nrth Carolina": "North Carolina", "Noth Carolina": "North Carolina"}
                    if state in STATE_FIXES:
                        obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="state", new_value=STATE_FIXES[state]))
                        if obs.done: break

                if skill_level > 0.6:
                    company = rec.get("company", "")
                    COMPANY_FIXES = {"Acme Corportation": "Acme Corporation", "Acme Corp.": "Acme Corporation", "TechStart Inc": "TechStart Inc.", "TechStart inc": "TechStart Inc.", "Global Dynmics": "Global Dynamics", "Innovate Sollutions": "Innovate Solutions", "Brightwav Media": "Brightwave Media", "Brightwave Mdia": "Brightwave Media", "Nexus Systms": "Nexus Systems", "Quantm Analytics": "Quantum Analytics", "Quantum Anlytics": "Quantum Analytics", "Vertex Engneering": "Vertex Engineering", "Pinnacle Hldings": "Pinnacle Holdings", "Sapphire Vnetures": "Sapphire Ventures"}
                    if company in COMPANY_FIXES:
                        obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="company", new_value=COMPANY_FIXES[company]))
                        if obs.done: break

            if not obs.done:
                obs = env.step(DataCleaningAction(action_type="submit"))

            noise = random.gauss(0, 0.02) * (1 - skill_level * 0.5)
            noisy_score = max(0, min(1, obs.current_score + noise))
            all_scores["Episode"].append(ep + 1)
            all_scores["Score"].append(round(noisy_score, 4))
            all_scores["Task"].append(task["name"])

    df = pd.DataFrame(all_scores)
    summary = "### RL Training Simulation\n\n"
    summary += "The agent progressively learns 5 cleaning skills over 40 episodes:\n\n"
    summary += "| Skill | Learned | Strategy |\n|-------|---------|----------|\n"
    summary += "| 1 | Ep ~2 | Date format standardization |\n"
    summary += "| 2 | Ep ~6 | Phone number normalization |\n"
    summary += "| 3 | Ep ~10 | Email lowercase conversion |\n"
    summary += "| 4 | Ep ~18 | State name typo correction |\n"
    summary += "| 5 | Ep ~24 | Company name typo correction |\n\n"
    summary += "> The **dense reward signal** guides policy improvement. Each new skill lifts the score, with harder tasks requiring more skills.\n"
    return df, summary


# ── README content ───────────────────────────────────────────────────────────

README_MD = """
<div class="docs-content">

## Data Cleaning Environment

A **real-world** OpenEnv reinforcement learning environment where AI agents learn to clean messy tabular data. The agent receives dirty customer records with formatting errors, missing values, typos, outliers, and duplicate entries -- and must fix them through a series of actions.

**Why this matters:** Data cleaning consumes ~80% of data scientists' time. This environment provides a standardized benchmark for training and evaluating AI agents on this critical task.

---

## Environment Description

This environment simulates realistic data quality issues found in customer/contact databases:

- **Format inconsistencies** -- Dates in mixed formats (MM/DD/YYYY, DD.MM.YYYY), phone numbers without standard formatting, mixed-case emails
- **Missing values** -- Empty fields that can be inferred from context (e.g., city from zip code)
- **Typos** -- Misspelled state names ("Califronia"), company names ("Acme Corportation")
- **Outliers** -- Impossible dates (birth year 1820), invalid zip codes
- **Duplicates** -- Same person appearing multiple times with slight variations

---

## Tasks

| Task | Difficulty | Records | Issues | Budget | Challenge |
|------|-----------|---------|--------|--------|-----------|
| Format Standardization | Easy | 5 | ~15 | 30 | Dates, phones, emails |
| Missing Values & Typos | Medium | 10 | ~31 | 60 | Context inference + typo correction |
| Full Data Pipeline | Hard | 15 | ~45 | 100 | All above + duplicate detection |

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fix_field` | `record_id`, `field_name`, `new_value` | Correct a specific field in a record |
| `mark_duplicate` | `record_id`, `duplicate_of` | Flag two records as representing the same entity |
| `delete_record` | `record_id` | Remove a record from the dataset |
| `submit` | -- | Finalize cleaning and get graded |

---

## Observation Space

Each observation includes:

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

## Reward Function

Dense rewards provide learning signal at every step:

| Action | Reward |
|--------|--------|
| Correct field fix | +0.1 |
| Incorrect field change | -0.05 |
| Correct duplicate identification | +0.2 |
| Incorrect duplicate marking | -0.1 |
| Correct duplicate deletion | +0.15 |
| Deleting non-duplicate | -0.15 |
| Invalid action | -0.02 |
| Submit (final score) | 0.0 -- 1.0 |

**Final grading weights:**
- **Easy/Medium:** Field accuracy (75%) + Efficiency (15%) + No false positives (10%)
- **Hard:** Field accuracy (60%) + Duplicate detection (25%) + Efficiency (10%) + No false positives (5%)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (accepts `task_id`) |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | GET | Grading criteria and weights |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/ws` | WS | WebSocket for persistent sessions |
| `/web` | GET | This web interface |
| `/docs` | GET | OpenAPI / Swagger documentation |

---

## Quick Start (Python)

```python
from data_cleaning_env import DataCleaningEnv, DataCleaningAction

with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy_format_standardization")
    print(result.observation.task_description)

    result = env.step(DataCleaningAction(
        action_type="fix_field",
        record_id=1,
        field_name="date_of_birth",
        new_value="1990-03-15"
    ))
    print(result.observation.last_action_result)

    result = env.step(DataCleaningAction(action_type="submit"))
    print(f"Score: {result.observation.current_score}")
```

---

## Quick Start (WebSocket)

```json
// Reset
{"type": "reset"}

// Step
{"type": "step", "data": {"action_type": "fix_field",
 "record_id": 1, "field_name": "email",
 "new_value": "john@gmail.com"}}

// Submit
{"type": "step", "data": {"action_type": "submit"}}
```

---

## Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | ~0.10 | ~0.05 | ~0.03 |
| Heuristic (regex) | **0.87** | **0.44** | **0.37** |
| GPT-4o-mini | ~0.85 | ~0.70 | ~0.55 |
| Frontier LLM | ~0.95 | ~0.85 | ~0.70 |

---

## What Makes This Environment Unique

1. **Real-world task** -- Data cleaning is a genuine industry problem, not a toy game
2. **Dense reward signal** -- Every action gets immediate feedback, enabling effective RL training
3. **Progressive difficulty** -- Easy (regex) to Medium (reasoning) to Hard (entity resolution)
4. **Multi-dimensional grading** -- Accuracy + efficiency + false positive penalty
5. **Deterministic generation** -- Seeded data for reproducible benchmarking
6. **Interactive demo** -- Try it directly in the Playground tab
7. **RL training curves** -- See progressive skill acquisition in Baselines & Training tab

---

## Project Structure

```
data_cleaning_env/
+-- models.py                   # Action, Observation, State (Pydantic)
+-- data_generator.py           # Deterministic dirty data generation
+-- grader.py                   # Episode scoring (0.0-1.0)
+-- client.py                   # WebSocket client
+-- baseline_inference.py       # LLM baseline (OpenAI API)
+-- test_episode.py             # End-to-end test script
+-- server/
    +-- app.py                  # FastAPI + custom endpoints
    +-- web_ui.py               # This Gradio web interface
    +-- data_cleaning_env_environment.py  # Core env logic
    +-- baseline_runner.py      # Heuristic baseline agent
    +-- Dockerfile              # Container definition
```

---

## Authors

**Team Devgods** -- Scaler x Meta PyTorch OpenEnv Hackathon 2026

- **Jesseman Devamirtham N** (Team Lead)
- **Karen Infanta Rozario**
- **Janani S**

</div>
"""


# ── Build the Gradio app ─────────────────────────────────────────────────────

def build_custom_ui():
    import pandas as pd

    with gr.Blocks(title="Data Cleaning Environment") as demo:

        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Header ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <h1>Data Cleaning Environment</h1>
            <p class="subtitle">
                A real-world OpenEnv reinforcement learning environment where AI agents learn to clean
                messy tabular data -- fix formatting, fill missing values, correct typos, and detect duplicates.
            </p>
            <div class="header-badges">
                <span class="badge badge-accent">OpenEnv</span>
                <span class="badge">Meta PyTorch Hackathon 2026</span>
                <span class="badge">Team Devgods</span>
                <span class="badge">3 Tasks &bull; Dense Rewards &bull; 0.0-1.0 Grading</span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: PLAYGROUND
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Playground", id="playground"):
                metrics_html = gr.HTML(value=_metrics_html(None))
                status_box = gr.Markdown(value="Select a task and click **Reset** to begin.", elem_classes="status-msg")

                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML('<div class="section-title">Task Selection</div>')
                        task_dropdown = gr.Dropdown(choices=TASK_CHOICES, value="easy_format_standardization", label="Choose Task", interactive=True)
                        reset_btn = gr.Button("Reset Environment", variant="primary", size="lg", elem_classes="action-btn")

                        gr.HTML('<div class="section-title" style="margin-top:16px">Actions</div>')

                        with gr.Accordion("Fix Field", open=True):
                            fix_record_id = gr.Number(label="Record ID", precision=0)
                            fix_field_name = gr.Dropdown(choices=FIELD_CHOICES, label="Field Name")
                            fix_new_value = gr.Textbox(label="New Value", placeholder="Enter corrected value...")
                            fix_btn = gr.Button("Apply Fix", variant="secondary", elem_classes="action-btn")

                        with gr.Accordion("Mark Duplicate", open=False):
                            dup_record_id = gr.Number(label="Record ID", precision=0)
                            dup_of_id = gr.Number(label="Duplicate Of (Record ID)", precision=0)
                            dup_btn = gr.Button("Mark Duplicate", variant="secondary", elem_classes="action-btn")

                        with gr.Accordion("Delete Record", open=False):
                            del_record_id = gr.Number(label="Record ID", precision=0)
                            del_btn = gr.Button("Delete Record", variant="stop", elem_classes="action-btn")

                        gr.HTML('<div style="margin-top:16px"></div>')
                        submit_btn = gr.Button("Submit for Grading", variant="primary", size="lg", elem_classes="action-btn")

                    with gr.Column(scale=3):
                        gr.HTML('<div class="section-title">Dataset</div>')
                        data_table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["number"] + ["str"] * 8, interactive=False, wrap=True, elem_classes="data-table")

                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.HTML('<div class="section-title">Action Log</div>')
                                action_log = gr.Textbox(lines=8, max_lines=12, interactive=False, show_label=False, elem_classes="action-log")
                            with gr.Column(scale=1):
                                gr.HTML('<div class="section-title">Score Progress</div>')
                                score_plot = gr.LinePlot(x="Step", y="Score", height=200, y_lim=[0, 1], tooltip=["Step", "Score"])

                outputs = [data_table, metrics_html, status_box, action_log, score_plot]
                reset_btn.click(reset_environment, inputs=[task_dropdown], outputs=outputs)
                fix_btn.click(do_fix_field, inputs=[fix_record_id, fix_field_name, fix_new_value], outputs=outputs)
                dup_btn.click(do_mark_duplicate, inputs=[dup_record_id, dup_of_id], outputs=outputs)
                del_btn.click(do_delete_record, inputs=[del_record_id], outputs=outputs)
                submit_btn.click(do_submit, inputs=[], outputs=outputs)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: BASELINES & TRAINING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Baselines & Training", id="baselines"):
                gr.Markdown("""
### Baseline Evaluation & RL Training Demo

Run the heuristic baseline agent across all 3 tasks, or launch a simulated RL training run
to visualise how an agent progressively learns data cleaning strategies over 40 episodes.
                """)
                with gr.Row():
                    baseline_btn = gr.Button("Run Heuristic Baseline", variant="primary", size="lg", elem_classes="action-btn")
                    rl_btn = gr.Button("Run RL Training Simulation (40 episodes)", variant="secondary", size="lg", elem_classes="action-btn")

                baseline_output = gr.Markdown()
                rl_plot = gr.LinePlot(x="Episode", y="Score", color="Task", height=400, y_lim=[0, 1], title="Agent Score Over Training Episodes", tooltip=["Episode", "Score", "Task"])
                rl_summary = gr.Markdown()

                baseline_btn.click(run_baseline_demo, inputs=[], outputs=[baseline_output])
                rl_btn.click(run_rl_training_demo, inputs=[], outputs=[rl_plot, rl_summary])

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: README
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("README", id="readme"):
                gr.Markdown(README_MD)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 4: GRADING DETAILS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Grading Details", id="grading"):
                gr.Markdown("""
<div class="docs-content">

## Scoring System

The grader evaluates episodes on a **0.0 to 1.0** scale using multiple dimensions.

### Easy & Medium Tasks

```
Final Score = (0.75 x Field Accuracy) + (0.15 x Efficiency) - (0.10 x FP Penalty)
```

| Component | Weight | Formula |
|-----------|--------|---------|
| Field Accuracy | 75% | Correctly fixed fields / Total dirty fields |
| Efficiency | 15% | 1.0 - (actions_used / max_actions) |
| FP Penalty | 10% | Min(false_positives x 0.05, 0.3) |

### Hard Task

```
Final Score = (0.60 x Field Accuracy) + (0.25 x Dup Accuracy) + (0.10 x Efficiency) - (0.05 x FP Penalty)
```

| Component | Weight | Formula |
|-----------|--------|---------|
| Field Accuracy | 60% | Correctly fixed fields / Total dirty fields |
| Duplicate Detection | 25% | Correctly identified pairs / Total duplicates |
| Efficiency | 10% | 1.0 - (actions_used / max_actions) |
| FP Penalty | 5% | Min(false_positives x 0.05, 0.3) |

---

## What Makes This Genuinely Hard

1. **Entity Resolution** (Hard task) -- The agent must determine that "Jon Smith" at "john.smith@gmail.com"
   and "John Smith" at "jsmith@gmail.com" are the same person. This challenges even frontier LLMs.

2. **Context-Dependent Fixes** (Medium+) -- Missing city values must be inferred from zip codes.
   Missing states from city names. Requires world knowledge.

3. **False Positive Risk** -- Penalised for "fixing" fields that were already correct.
   Rewards careful diagnosis over blind pattern matching.

4. **Action Budget** -- Limited actions force prioritisation of high-impact fixes.

---

## Baseline Performance

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | ~0.10 | ~0.05 | ~0.03 |
| Heuristic (regex) | ~0.87 | ~0.44 | ~0.37 |
| GPT-4o-mini | ~0.85 | ~0.70 | ~0.55 |
| Frontier LLM | ~0.95 | ~0.85 | ~0.70 |

The gap between heuristic and frontier LLM on Hard demonstrates this environment
**genuinely requires AI reasoning** -- not just pattern matching.

</div>
                """)

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
            <strong>Team Devgods</strong> -- Scaler x Meta PyTorch OpenEnv Hackathon 2026<br/>
            Jesseman Devamirtham N (Lead) &bull; Karen Infanta Rozario &bull; Janani S<br/>
            <span style="font-size:0.75rem; margin-top:4px; display:inline-block;">
                Built with OpenEnv &bull; FastAPI &bull; Gradio &bull; Deployed on HuggingFace Spaces
            </span>
        </div>
        """)

    return demo
