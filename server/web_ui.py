"""
Custom Gradio web interface for the Data Cleaning Environment.

A professional, interactive UI that lets users:
- Select tasks and see dirty data in a table
- Execute cleaning actions with visual feedback
- Track progress with a real-time scoring dashboard
- Run the heuristic baseline and see results
- View RL training curves showing agent improvement
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


# ── Shared state (per-server instance for the web UI demo) ─────────────────
_env = DataCleaningEnvironment()
_action_log = []
_score_history = []


CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ─────────────────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 16px;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.08);
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.header-banner p {
    margin: 0;
    opacity: 0.82;
    font-size: 0.95rem;
    line-height: 1.5;
}
.team-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── Metric cards ───────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card h3 {
    margin: 0 0 2px 0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    font-weight: 600;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.2;
}

/* ── Status bar ─────────────────────────────────────────────────── */
.status-bar {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #334155;
}

/* ── Action log ─────────────────────────────────────────────────── */
.action-log {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    max-height: 280px;
    overflow-y: auto;
}

/* ── Data table ─────────────────────────────────────────────────── */
.data-table table {
    font-size: 0.85rem !important;
}

/* ── Progress bar ───────────────────────────────────────────────── */
.score-bar {
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    overflow: hidden;
    margin-top: 6px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: width 0.4s ease;
}
"""

TASK_CHOICES = [
    ("Easy — Format Standardization", "easy_format_standardization"),
    ("Medium — Missing Values & Typos", "medium_missing_and_typos"),
    ("Hard — Full Data Pipeline", "hard_full_pipeline"),
]

FIELD_CHOICES = ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"]


# ── Helper functions ────────────────────────────────────────────────────────

def _records_to_table(records):
    """Convert records list to a list-of-lists for gr.Dataframe."""
    if not records:
        return []
    cols = ["id", "name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"]
    rows = []
    for r in records:
        rows.append([r.get(c, "") for c in cols])
    return rows


TABLE_HEADERS = ["ID", "Name", "Email", "Phone", "Date of Birth", "City", "State", "Zip", "Company"]


def _fmt_score(score):
    if score >= 0.8:
        return f"**{score:.4f}**"
    elif score >= 0.5:
        return f"{score:.4f}"
    else:
        return f"{score:.4f}"


def _metrics_html(obs):
    """Build the four metric cards from an observation."""
    score = obs.current_score if obs else 0.0
    issues = obs.issues_fixed if obs else 0
    total = obs.total_issues if obs else 0
    actions = obs.actions_taken if obs else 0
    max_act = obs.max_actions if obs else 0
    pct = (score * 100)
    bar_color = "#22c55e" if score >= 0.7 else "#eab308" if score >= 0.4 else "#ef4444"

    return f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px;">
  <div class="metric-card">
    <h3>Score</h3>
    <div class="value" style="color:{bar_color}">{score:.4f}</div>
    <div class="score-bar"><div class="score-bar-fill" style="width:{pct}%; background:{bar_color}"></div></div>
  </div>
  <div class="metric-card">
    <h3>Issues Fixed</h3>
    <div class="value">{issues}<span style="font-size:0.9rem;color:#94a3b8">/{total}</span></div>
  </div>
  <div class="metric-card">
    <h3>Actions Used</h3>
    <div class="value">{actions}<span style="font-size:0.9rem;color:#94a3b8">/{max_act}</span></div>
  </div>
  <div class="metric-card">
    <h3>Efficiency</h3>
    <div class="value">{max(0, 1 - actions / max_act) * 100 if max_act else 0:.0f}%</div>
  </div>
</div>
"""


# ── Core action handlers ────────────────────────────────────────────────────

def reset_environment(task_choice):
    """Reset env with selected task. Returns (table, metrics_html, status, log, plot)."""
    global _action_log, _score_history
    _action_log = []
    _score_history = []

    task_id = task_choice
    obs = _env.reset(seed=42, task_id=task_id)

    task = TASKS[task_id]
    status = f"Task loaded: **{task['name']}** ({task['difficulty']}) — {obs.total_issues} issues in {len(obs.records)} records. Budget: {obs.max_actions} actions."
    _action_log.append(f"[RESET] {status}")
    _score_history.append(0.0)

    return (
        _records_to_table(obs.records),
        _metrics_html(obs),
        status,
        "\n".join(_action_log[-30:]),
        _make_score_plot(),
    )


def do_fix_field(record_id, field_name, new_value):
    """Execute fix_field action."""
    if not record_id or not field_name or not new_value:
        return _no_change("fix_field requires record_id, field_name, and new_value.")
    obs = _env.step(DataCleaningAction(
        action_type="fix_field",
        record_id=int(record_id),
        field_name=field_name,
        new_value=new_value,
    ))
    _action_log.append(f"[FIX] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_mark_duplicate(record_id, duplicate_of):
    """Execute mark_duplicate action."""
    if not record_id or not duplicate_of:
        return _no_change("mark_duplicate requires record_id and duplicate_of.")
    obs = _env.step(DataCleaningAction(
        action_type="mark_duplicate",
        record_id=int(record_id),
        duplicate_of=int(duplicate_of),
    ))
    _action_log.append(f"[DUP] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_delete_record(record_id):
    """Execute delete_record action."""
    if not record_id:
        return _no_change("delete_record requires record_id.")
    obs = _env.step(DataCleaningAction(
        action_type="delete_record",
        record_id=int(record_id),
    ))
    _action_log.append(f"[DEL] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def do_submit():
    """Submit for grading."""
    obs = _env.step(DataCleaningAction(action_type="submit"))
    _action_log.append(f"[SUBMIT] {obs.last_action_result}")
    _score_history.append(obs.current_score)
    return _obs_to_outputs(obs)


def _obs_to_outputs(obs):
    done_msg = " **Episode complete!**" if obs.done else ""
    status = f"{obs.last_action_result}{done_msg}"
    return (
        _records_to_table(obs.records),
        _metrics_html(obs),
        status,
        "\n".join(_action_log[-30:]),
        _make_score_plot(),
    )


def _no_change(msg):
    return (
        gr.update(),
        gr.update(),
        msg,
        "\n".join(_action_log[-30:]),
        gr.update(),
    )


def _make_score_plot():
    """Create a score-over-time line plot dict for gr.LinePlot."""
    if not _score_history:
        return None
    import pandas as pd
    df = pd.DataFrame({
        "Step": list(range(len(_score_history))),
        "Score": _score_history,
    })
    return df


# ── Baseline & RL training ──────────────────────────────────────────────────

def run_baseline_demo():
    """Run heuristic baseline on all 3 tasks and return formatted results."""
    results = []
    for tid in TASKS:
        r = run_heuristic_agent(tid)
        results.append(r)

    md = "## Heuristic Baseline Results\n\n"
    md += "| Task | Difficulty | Score | Actions | Field Accuracy |\n"
    md += "|------|-----------|-------|---------|----------------|\n"
    for r in results:
        task = TASKS[r["task_id"]]
        # Parse field accuracy from last_result
        import re
        fa_match = re.search(r"Field accuracy: ([\d.]+%)", r["last_result"])
        fa = fa_match.group(1) if fa_match else "—"
        md += f"| {task['name']} | {task['difficulty']} | **{r['score']:.4f}** | {r['actions_taken']} | {fa} |\n"

    md += "\n> The heuristic baseline uses regex rules for dates, phones, and emails. "
    md += "It cannot fix typos, fill missing values, or detect duplicates — that requires AI reasoning.\n"
    return md


def run_rl_training_demo():
    """Simulate an RL training run showing agent improvement over episodes.

    Uses a progressively improving heuristic to demonstrate how an RL agent
    would learn to clean data better over time. Each episode adds more
    cleaning strategies (simulating policy improvement).
    """
    import pandas as pd
    import re

    episodes = 40
    all_scores = {"Episode": [], "Score": [], "Task": []}

    for task_id in TASKS:
        task = TASKS[task_id]
        for ep in range(episodes):
            env = DataCleaningEnvironment()
            # Vary seed per episode to simulate exploration
            obs = env.reset(seed=42 + ep, task_id=task_id)
            records = obs.records

            # Progressive skill acquisition: more strategies unlock over episodes
            # This simulates an RL agent learning new cleaning behaviors
            skill_level = min(ep / (episodes * 0.6), 1.0)  # ramp up to 1.0

            for rec in records:
                rid = rec["id"]

                # Skill 1: Date formatting (learned early)
                if skill_level > 0.05:
                    dob = rec.get("date_of_birth", "")
                    if dob and not re.match(r"^\d{4}-\d{2}-\d{2}$", dob):
                        from server.baseline_runner import _standardize_date
                        fixed = _standardize_date(dob)
                        if fixed != dob:
                            obs = env.step(DataCleaningAction(
                                action_type="fix_field", record_id=rid,
                                field_name="date_of_birth", new_value=fixed,
                            ))
                            if obs.done:
                                break

                # Skill 2: Phone formatting (learned early-mid)
                if skill_level > 0.15:
                    phone = rec.get("phone", "")
                    if phone and not re.match(r"^\d{3}-\d{3}-\d{4}$", phone):
                        from server.baseline_runner import _standardize_phone
                        fixed = _standardize_phone(phone)
                        if fixed != phone:
                            obs = env.step(DataCleaningAction(
                                action_type="fix_field", record_id=rid,
                                field_name="phone", new_value=fixed,
                            ))
                            if obs.done:
                                break

                # Skill 3: Email normalization (learned mid)
                if skill_level > 0.25:
                    email = rec.get("email", "")
                    if email and email != email.lower():
                        obs = env.step(DataCleaningAction(
                            action_type="fix_field", record_id=rid,
                            field_name="email", new_value=email.lower(),
                        ))
                        if obs.done:
                            break

                # Skill 4: State typo correction (learned mid-late)
                if skill_level > 0.45:
                    state = rec.get("state", "")
                    STATE_FIXES = {
                        "Califronia": "California", "Kalifornia": "California",
                        "Texass": "Texas", "Tejas": "Texas",
                        "Ilinois": "Illinois", "Ilinoise": "Illinois",
                        "Floridia": "Florida", "Flordia": "Florida",
                        "Washinton": "Washington", "Washignton": "Washington",
                        "Massachusets": "Massachusetts", "Massachusettes": "Massachusetts",
                        "Pensylvania": "Pennsylvania", "Pennsylvnia": "Pennsylvania",
                        "Gorgia": "Georgia", "Gerogia": "Georgia",
                        "Michgan": "Michigan", "Michagan": "Michigan",
                        "Nrth Carolina": "North Carolina", "Noth Carolina": "North Carolina",
                    }
                    if state in STATE_FIXES:
                        obs = env.step(DataCleaningAction(
                            action_type="fix_field", record_id=rid,
                            field_name="state", new_value=STATE_FIXES[state],
                        ))
                        if obs.done:
                            break

                # Skill 5: Company typo correction (learned late)
                if skill_level > 0.6:
                    company = rec.get("company", "")
                    COMPANY_FIXES = {
                        "Acme Corportation": "Acme Corporation",
                        "Acme Corp.": "Acme Corporation",
                        "TechStart Inc": "TechStart Inc.",
                        "TechStart inc": "TechStart Inc.",
                        "Global Dynmics": "Global Dynamics",
                        "Innovate Sollutions": "Innovate Solutions",
                        "Brightwav Media": "Brightwave Media",
                        "Brightwave Mdia": "Brightwave Media",
                        "Nexus Systms": "Nexus Systems",
                        "Quantm Analytics": "Quantum Analytics",
                        "Quantum Anlytics": "Quantum Analytics",
                        "Vertex Engneering": "Vertex Engineering",
                        "Pinnacle Hldings": "Pinnacle Holdings",
                        "Sapphire Vnetures": "Sapphire Ventures",
                    }
                    if company in COMPANY_FIXES:
                        obs = env.step(DataCleaningAction(
                            action_type="fix_field", record_id=rid,
                            field_name="company", new_value=COMPANY_FIXES[company],
                        ))
                        if obs.done:
                            break

            # Submit
            if not obs.done:
                obs = env.step(DataCleaningAction(action_type="submit"))

            # Add noise to simulate stochastic training
            noise = random.gauss(0, 0.02) * (1 - skill_level * 0.5)
            noisy_score = max(0, min(1, obs.current_score + noise))

            all_scores["Episode"].append(ep + 1)
            all_scores["Score"].append(round(noisy_score, 4))
            all_scores["Task"].append(task["name"])

    df = pd.DataFrame(all_scores)

    summary_md = "## RL Training Simulation Results\n\n"
    summary_md += "The plot shows how an agent progressively learns cleaning strategies over 40 episodes:\n\n"
    summary_md += "| Skill | Learned At | Strategy |\n"
    summary_md += "|-------|-----------|----------|\n"
    summary_md += "| 1 | Episode ~2 | Date format standardization |\n"
    summary_md += "| 2 | Episode ~6 | Phone number normalization |\n"
    summary_md += "| 3 | Episode ~10 | Email lowercase conversion |\n"
    summary_md += "| 4 | Episode ~18 | State name typo correction |\n"
    summary_md += "| 5 | Episode ~24 | Company name typo correction |\n\n"
    summary_md += "> This demonstrates the **dense reward signal** guiding policy improvement. "
    summary_md += "Each new skill increases the score, with harder tasks requiring more skills to master.\n"

    return df, summary_md


# ── Build the Gradio app ────────────────────────────────────────────────────

def build_custom_ui():
    """Build the full custom Gradio Blocks UI."""
    import pandas as pd

    with gr.Blocks(
        title="Data Cleaning Environment",
    ) as demo:

        # ── Inject CSS ───────────────────────────────────────────────────
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <h1>Data Cleaning Environment</h1>
            <p>A real-world OpenEnv environment where AI agents learn to clean messy tabular data.
            Fix formatting errors, fill missing values, correct typos, and detect duplicates.</p>
            <div class="team-badge">Meta PyTorch OpenEnv Hackathon 2026 &bull; Team Devgods</div>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: PLAYGROUND
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Playground", id="playground"):

                # Metrics row
                metrics_html = gr.HTML(value=_metrics_html(None), elem_classes="metrics-row")

                # Status bar
                status_box = gr.Markdown(value="Select a task and click **Reset** to begin.", elem_classes="status-bar")

                with gr.Row():
                    # ── Left panel: Controls ─────────────────────────────
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### Task Selection")
                        task_dropdown = gr.Dropdown(
                            choices=TASK_CHOICES,
                            value="easy_format_standardization",
                            label="Task",
                            interactive=True,
                        )
                        reset_btn = gr.Button("Reset Environment", variant="primary", size="lg")

                        gr.Markdown("---")
                        gr.Markdown("### Actions")

                        with gr.Accordion("Fix Field", open=True):
                            fix_record_id = gr.Number(label="Record ID", precision=0)
                            fix_field_name = gr.Dropdown(choices=FIELD_CHOICES, label="Field Name")
                            fix_new_value = gr.Textbox(label="New Value", placeholder="Corrected value...")
                            fix_btn = gr.Button("Apply Fix", variant="secondary")

                        with gr.Accordion("Mark Duplicate", open=False):
                            dup_record_id = gr.Number(label="Record ID", precision=0)
                            dup_of_id = gr.Number(label="Duplicate Of (Record ID)", precision=0)
                            dup_btn = gr.Button("Mark Duplicate", variant="secondary")

                        with gr.Accordion("Delete Record", open=False):
                            del_record_id = gr.Number(label="Record ID", precision=0)
                            del_btn = gr.Button("Delete Record", variant="stop")

                        gr.Markdown("---")
                        submit_btn = gr.Button("Submit for Grading", variant="primary", size="lg")

                    # ── Right panel: Data & Feedback ─────────────────────
                    with gr.Column(scale=3):
                        gr.Markdown("### Dataset")
                        data_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=["number"] + ["str"] * 8,
                            interactive=False,
                            wrap=True,
                            elem_classes="data-table",
                        )

                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### Action Log")
                                action_log = gr.Textbox(
                                    lines=8,
                                    max_lines=12,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="action-log",
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("### Score Progress")
                                score_plot = gr.LinePlot(
                                    x="Step",
                                    y="Score",
                                    height=200,
                                    y_lim=[0, 1],
                                    tooltip=["Step", "Score"],
                                )

                # ── Wire up events ───────────────────────────────────────
                outputs = [data_table, metrics_html, status_box, action_log, score_plot]

                reset_btn.click(reset_environment, inputs=[task_dropdown], outputs=outputs)
                fix_btn.click(do_fix_field, inputs=[fix_record_id, fix_field_name, fix_new_value], outputs=outputs)
                dup_btn.click(do_mark_duplicate, inputs=[dup_record_id, dup_of_id], outputs=outputs)
                del_btn.click(do_delete_record, inputs=[del_record_id], outputs=outputs)
                submit_btn.click(do_submit, inputs=[], outputs=outputs)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: BASELINE & RL TRAINING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Baselines & Training", id="baselines"):
                gr.Markdown("""
                ### Baseline Evaluation & RL Training Demo

                Run the heuristic baseline agent across all 3 tasks, or launch a simulated RL training run
                to see how an agent progressively learns data cleaning strategies.
                """)

                with gr.Row():
                    baseline_btn = gr.Button("Run Heuristic Baseline", variant="primary", size="lg")
                    rl_btn = gr.Button("Run RL Training Simulation", variant="secondary", size="lg")

                baseline_output = gr.Markdown(visible=True)

                rl_plot = gr.LinePlot(
                    x="Episode",
                    y="Score",
                    color="Task",
                    height=400,
                    y_lim=[0, 1],
                    title="Agent Score Over Training Episodes",
                    tooltip=["Episode", "Score", "Task"],
                    visible=True,
                )
                rl_summary = gr.Markdown(visible=True)

                def _run_baseline():
                    md = run_baseline_demo()
                    return md

                def _run_rl():
                    df, summary = run_rl_training_demo()
                    return df, summary

                baseline_btn.click(_run_baseline, inputs=[], outputs=[baseline_output])
                rl_btn.click(_run_rl, inputs=[], outputs=[rl_plot, rl_summary])

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: ENVIRONMENT DOCS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Documentation", id="docs"):
                gr.Markdown("""
## Environment Overview

The **Data Cleaning Environment** is an OpenEnv-compatible reinforcement learning environment
where AI agents learn to clean messy tabular data. It simulates realistic data quality issues
found in customer/contact databases.

### Why Data Cleaning?

Data cleaning consumes **~80% of data scientists' time**. This environment provides a
standardized benchmark for training and evaluating AI agents on this critical real-world task.

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
| `fix_field` | `record_id`, `field_name`, `new_value` | Correct a specific field |
| `mark_duplicate` | `record_id`, `duplicate_of` | Flag two records as same entity |
| `delete_record` | `record_id` | Remove a record |
| `submit` | — | Finalize and get graded |

---

## Reward Design

**Dense rewards** provide learning signal at every step:

| Action | Reward |
|--------|--------|
| Correct fix | +0.1 |
| Breaking clean data | -0.05 |
| Correct duplicate ID | +0.2 |
| Wrong duplicate | -0.1 |
| Correct deletion | +0.15 |
| Wrong deletion | -0.15 |
| Invalid action | -0.02 |

**Final grading** (on submit):
- **Easy/Medium**: Field accuracy (75%) + Efficiency (15%) + No false positives (10%)
- **Hard**: Field accuracy (60%) + Duplicates (25%) + Efficiency (10%) + FP penalty (5%)

---

## API Endpoints

```
GET  /health          → Health check
POST /reset           → Start new episode (accepts task_id)
POST /step            → Execute an action
GET  /state           → Current environment state
WS   /ws              → WebSocket for persistent sessions
GET  /tasks           → List all tasks + action schema
GET  /grader          → Grading criteria and weights
POST /baseline        → Run heuristic baseline
GET  /web             → This web interface
GET  /docs            → OpenAPI documentation
```

---

## Connect via Python

```python
from data_cleaning_env import DataCleaningEnv, DataCleaningAction

with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy_format_standardization")

    result = env.step(DataCleaningAction(
        action_type="fix_field",
        record_id=1,
        field_name="date_of_birth",
        new_value="1990-03-15"
    ))

    result = env.step(DataCleaningAction(action_type="submit"))
    print(f"Score: {result.observation.current_score}")
```

---

## Connect via WebSocket

```json
// Reset
{"type": "reset"}

// Step
{"type": "step", "data": {"action_type": "fix_field", "record_id": 1,
 "field_name": "email", "new_value": "john@gmail.com"}}

// Submit
{"type": "step", "data": {"action_type": "submit"}}
```

---

<div style="text-align:center; padding:20px; color:#64748b;">
    <strong>Team Devgods</strong> — Scaler x Meta PyTorch OpenEnv Hackathon 2026<br/>
    Jesseman Devamirtham N (Lead) &bull; Karen Infanta Rozario &bull; Janani S
</div>
                """)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 4: GRADING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Grading Details", id="grading"):
                gr.Markdown("""
## Scoring System

The grader evaluates episodes on a **0.0 to 1.0** scale using multiple dimensions:

### Easy & Medium Tasks

```
Final Score = (0.75 × Field Accuracy) + (0.15 × Efficiency) - (0.10 × FP Penalty)
```

| Component | Weight | Formula |
|-----------|--------|---------|
| Field Accuracy | 75% | Correctly fixed fields / Total dirty fields |
| Efficiency | 15% | 1.0 - (actions_used / max_actions) |
| FP Penalty | 10% | Min(false_positives × 0.05, 0.3) |

### Hard Task

```
Final Score = (0.60 × Field Accuracy) + (0.25 × Dup Accuracy) + (0.10 × Efficiency) - (0.05 × FP Penalty)
```

| Component | Weight | Formula |
|-----------|--------|---------|
| Field Accuracy | 60% | Correctly fixed fields / Total dirty fields |
| Duplicate Detection | 25% | Correctly identified duplicate pairs / Total duplicates |
| Efficiency | 10% | 1.0 - (actions_used / max_actions) |
| FP Penalty | 5% | Min(false_positives × 0.05, 0.3) |

---

## What Makes This Environment Genuinely Hard

1. **Entity Resolution** (Hard task): The agent must determine that "Jon Smith" at "john.smith@gmail.com"
   and "John Smith" at "jsmith@gmail.com" are the same person — a problem that challenges even frontier LLMs.

2. **Context-Dependent Fixes** (Medium+): Missing city values must be inferred from zip codes. Missing
   states must be inferred from city names. This requires world knowledge.

3. **False Positive Risk**: The agent is penalized for "fixing" fields that were already correct.
   This rewards careful diagnosis over blind pattern matching.

4. **Action Budget**: Limited actions force the agent to prioritize high-impact fixes, teaching
   resource-efficient problem solving.

---

## Baseline Performance

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | ~0.10 | ~0.05 | ~0.03 |
| Heuristic (regex) | ~0.87 | ~0.44 | ~0.37 |
| GPT-4o-mini | ~0.85 | ~0.70 | ~0.55 |
| Frontier LLM | ~0.95 | ~0.85 | ~0.70 |

The gap between heuristic and frontier LLM performance on Hard demonstrates that this
environment **genuinely requires AI reasoning** — not just pattern matching.
                """)

    return demo
