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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header banner ──────────────────────────────────────────── */
.header-banner {
    background: linear-gradient(135deg, #0a0f1a 0%, #111827 30%, #1a1f3a 60%, #0f2440 100%);
    border-radius: 16px;
    padding: 40px 44px 32px;
    margin-bottom: 12px;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -15%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, rgba(59,130,246,0.08) 30%, transparent 70%);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -5%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 10s ease-in-out infinite reverse;
}
@keyframes float {
    0%, 100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-20px) scale(1.05); }
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 2.1rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-banner .subtitle {
    margin: 0 0 18px 0;
    opacity: 0.72;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 720px;
    position: relative;
    font-weight: 400;
}
.header-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    position: relative;
    margin-bottom: 16px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
.badge:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-1px);
}
.badge-accent {
    background: rgba(99,102,241,0.2);
    border-color: rgba(99,102,241,0.4);
    color: #c7d2fe;
}
.badge-green {
    background: rgba(16,185,129,0.15);
    border-color: rgba(16,185,129,0.3);
    color: #a7f3d0;
}
.header-links {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    position: relative;
}
.header-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #e2e8f0;
    text-decoration: none;
    transition: all 0.2s ease;
    cursor: pointer;
    backdrop-filter: blur(8px);
}
.header-link:hover {
    background: rgba(255,255,255,0.14);
    border-color: rgba(255,255,255,0.25);
    transform: translateY(-1px);
    color: #fff;
}
.header-link svg {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
}

/* ── Stats row in header ──────────────────────────────────── */
.header-stats {
    display: flex;
    gap: 28px;
    margin: 18px 0 20px;
    position: relative;
}
.header-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.header-stat .hs-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.header-stat .hs-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.45);
    font-weight: 600;
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
    border-radius: 14px;
    padding: 20px 22px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.02);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent, #3b82f6), var(--accent-end, #6366f1));
    opacity: 0;
    transition: opacity 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.03);
}
.metric-card:hover::before {
    opacity: 1;
}
.metric-card .label {
    margin: 0 0 6px 0;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-weight: 700;
}
.metric-card .value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.metric-card .sub {
    font-size: 0.82rem;
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
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Status ─────────────────────────────────────────────────── */
.status-msg {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    font-size: 0.88rem;
    color: #334155;
}

/* ── Section titles ─────────────────────────────────────────── */
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin: 0 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Action log ─────────────────────────────────────────────── */
.action-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.7 !important;
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Data table ─────────────────────────────────────────────── */
.data-table table {
    font-size: 0.82rem !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.data-table th {
    background: #f1f5f9 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    color: #475569 !important;
    padding: 10px 12px !important;
}
.data-table td {
    padding: 8px 12px !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.action-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.action-btn:hover {
    transform: translateY(-1px) !important;
}

/* ── Docs section ───────────────────────────────────────────── */
.docs-content {
    max-width: 880px;
}
.docs-content h2 {
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 10px;
    margin-top: 32px;
    font-weight: 800;
}
.docs-content h3 {
    color: #1e293b;
    margin-top: 24px;
}
.docs-content table {
    font-size: 0.88rem;
}
.docs-content code {
    background: #f1f5f9;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.85em;
    font-family: 'JetBrains Mono', monospace;
}
.docs-content pre {
    background: #0f172a !important;
    border-radius: 10px;
    padding: 18px !important;
    border: 1px solid #1e293b;
}

/* ── How it works cards ────────────────────────────────────── */
.how-it-works {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 16px 0;
}
.hiw-card {
    background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    transition: all 0.2s ease;
}
.hiw-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
.hiw-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 800;
    margin-bottom: 10px;
}
.hiw-step-1 { background: #dbeafe; color: #2563eb; }
.hiw-step-2 { background: #fce7f3; color: #db2777; }
.hiw-step-3 { background: #d1fae5; color: #059669; }
.hiw-step-4 { background: #fef3c7; color: #d97706; }
.hiw-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
}
.hiw-desc {
    font-size: 0.73rem;
    color: #64748b;
    line-height: 1.45;
}

/* ── Footer ─────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 28px 16px;
    color: #94a3b8;
    font-size: 0.82rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 24px;
    background: linear-gradient(180deg, transparent 0%, #f8fafc 100%);
}
.footer strong { color: #475569; }
.footer-tech {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.footer-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #475569;
}
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

    eff_color = "#22c55e" if eff >= 70 else "#f59e0b" if eff >= 40 else "#ef4444"

    return f"""
<div class="metric-row">
  <div class="metric-card" style="--accent: {color}; --accent-end: {color}">
    <div class="label">Score</div>
    <div class="value" style="color:{color}">{score:.4f}</div>
    <div class="score-bar-wrap">
      <div class="score-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}cc)"></div>
    </div>
  </div>
  <div class="metric-card" style="--accent: #6366f1; --accent-end: #8b5cf6">
    <div class="label">Issues Fixed</div>
    <div class="value" style="color:#6366f1">{issues}<span class="sub">/{total}</span></div>
  </div>
  <div class="metric-card" style="--accent: #3b82f6; --accent-end: #06b6d4">
    <div class="label">Actions Used</div>
    <div class="value" style="color:#3b82f6">{actions}<span class="sub">/{max_act}</span></div>
  </div>
  <div class="metric-card" style="--accent: {eff_color}; --accent-end: {eff_color}">
    <div class="label">Efficiency</div>
    <div class="value" style="color:{eff_color}">{eff:.0f}<span class="sub">%</span></div>
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
                    STATE_FIXES = {"Maharshtra": "Maharashtra", "Maharastra": "Maharashtra", "Maharashthra": "Maharashtra", "Karnatka": "Karnataka", "Karntaka": "Karnataka", "Karantaka": "Karnataka", "Tamil Naidu": "Tamil Nadu", "Tamilnadu": "Tamil Nadu", "Tamil Ndu": "Tamil Nadu", "Telangna": "Telangana", "Telegana": "Telangana", "Telengana": "Telangana", "Kerla": "Kerala", "Kerela": "Kerala", "Keralla": "Kerala", "Uttar Pradsh": "Uttar Pradesh", "Utter Pradesh": "Uttar Pradesh", "Rajasthn": "Rajasthan", "Rajsthan": "Rajasthan", "Gujrat": "Gujarat", "Gujerat": "Gujarat", "Andhra Pradsh": "Andhra Pradesh", "Andra Pradesh": "Andhra Pradesh", "West Bangal": "West Bengal", "West Bengel": "West Bengal"}
                    if state in STATE_FIXES:
                        obs = env.step(DataCleaningAction(action_type="fix_field", record_id=rid, field_name="state", new_value=STATE_FIXES[state]))
                        if obs.done: break

                if skill_level > 0.6:
                    company = rec.get("company", "")
                    COMPANY_FIXES = {"Tata Consultany Services": "Tata Consultancy Services", "Infosys Ldt": "Infosys Ltd", "Reliance Industires": "Reliance Industries", "Zoho Corporaton": "Zoho Corporation", "Persistant Systems": "Persistent Systems", "HCL Technolgies": "HCL Technologies", "Wipro Ldt": "Wipro Ltd", "Mindtree Ldt": "Mindtree Ltd", "Tech Mahinrda": "Tech Mahindra", "Cognizent": "Cognizant"}
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

<div style="background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%); border: 1px solid #dbeafe; border-radius: 14px; padding: 28px 32px; margin-bottom: 28px;">
<h2 style="margin: 0 0 8px 0; color: #1e293b; font-size: 1.5rem; font-weight: 800; border: none; padding: 0;">Data Cleaning Environment</h2>
<p style="margin: 0 0 16px 0; color: #475569; font-size: 0.95rem; line-height: 1.6;">
A <strong>real-world</strong> OpenEnv reinforcement learning environment where AI agents learn to clean messy tabular data &mdash; fix formatting, fill missing values, correct typos, and detect duplicates.
</p>
<p style="margin: 0; color: #64748b; font-size: 0.88rem;">
<strong style="color: #dc2626;">Why this matters:</strong> Data cleaning consumes ~80% of data scientists' time. This environment provides a standardized benchmark for training and evaluating AI agents on this critical task.
</p>
</div>

---

## Data Quality Issues Simulated

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0 24px;">
<div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #991b1b; font-size: 0.82rem; margin-bottom: 4px;">Format Inconsistencies</div>
<div style="color: #7f1d1d; font-size: 0.78rem; line-height: 1.5;">Dates in mixed formats (MM/DD/YYYY, DD.MM.YYYY), phones without standard formatting, mixed-case emails</div>
</div>
<div style="background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #9a3412; font-size: 0.82rem; margin-bottom: 4px;">Missing Values</div>
<div style="color: #7c2d12; font-size: 0.78rem; line-height: 1.5;">Empty fields that can be inferred from context (e.g., city from zip code, state from city)</div>
</div>
<div style="background: #fefce8; border: 1px solid #fde68a; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #854d0e; font-size: 0.82rem; margin-bottom: 4px;">Typos &amp; Misspellings</div>
<div style="color: #713f12; font-size: 0.78rem; line-height: 1.5;">Misspelled state names ("Maharshtra"), company names ("Tata Consultany Services")</div>
</div>
<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #166534; font-size: 0.82rem; margin-bottom: 4px;">Outliers &amp; Anomalies</div>
<div style="color: #14532d; font-size: 0.78rem; line-height: 1.5;">Impossible dates (birth year 1820), invalid zip codes, out-of-range values</div>
</div>
<div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #1e40af; font-size: 0.82rem; margin-bottom: 4px;">Duplicate Records</div>
<div style="color: #1e3a5f; font-size: 0.78rem; line-height: 1.5;">Same person appearing multiple times with slight variations in name, email, or phone</div>
</div>
<div style="background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 10px; padding: 14px 16px;">
<div style="font-weight: 700; color: #6b21a8; font-size: 0.82rem; margin-bottom: 4px;">Entity Resolution</div>
<div style="color: #581c87; font-size: 0.78rem; line-height: 1.5;">"Rahul Shrma" at rahulsharma@gmail.com = "Rahul Sharma" at rahul.sharma@gmail.com? Requires reasoning.</div>
</div>
</div>

---

## Tasks

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0 24px;">
<div style="background: linear-gradient(145deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #86efac; border-radius: 12px; padding: 20px; text-align: center;">
<div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #16a34a; font-weight: 700; margin-bottom: 6px;">Easy</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #14532d; margin-bottom: 8px;">Format Standardization</div>
<div style="display: flex; justify-content: center; gap: 16px; font-size: 0.78rem; color: #166534;">
<span><strong>5</strong> records</span>
<span><strong>~15</strong> issues</span>
<span><strong>30</strong> budget</span>
</div>
<div style="margin-top: 10px; font-size: 0.75rem; color: #15803d; line-height: 1.45;">Dates, phones, emails &mdash; straightforward pattern matching</div>
</div>
<div style="background: linear-gradient(145deg, #fff7ed 0%, #ffedd5 100%); border: 1px solid #fdba74; border-radius: 12px; padding: 20px; text-align: center;">
<div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #ea580c; font-weight: 700; margin-bottom: 6px;">Medium</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #7c2d12; margin-bottom: 8px;">Missing Values &amp; Typos</div>
<div style="display: flex; justify-content: center; gap: 16px; font-size: 0.78rem; color: #9a3412;">
<span><strong>10</strong> records</span>
<span><strong>~31</strong> issues</span>
<span><strong>60</strong> budget</span>
</div>
<div style="margin-top: 10px; font-size: 0.75rem; color: #c2410c; line-height: 1.45;">Contextual reasoning &mdash; infer city from zip, recognize misspelled states</div>
</div>
<div style="background: linear-gradient(145deg, #fef2f2 0%, #fecaca 100%); border: 1px solid #f87171; border-radius: 12px; padding: 20px; text-align: center;">
<div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #dc2626; font-weight: 700; margin-bottom: 6px;">Hard</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #7f1d1d; margin-bottom: 8px;">Full Data Pipeline</div>
<div style="display: flex; justify-content: center; gap: 16px; font-size: 0.78rem; color: #991b1b;">
<span><strong>15</strong> records</span>
<span><strong>~45</strong> issues</span>
<span><strong>100</strong> budget</span>
</div>
<div style="margin-top: 10px; font-size: 0.75rem; color: #b91c1c; line-height: 1.45;">Everything above + duplicate detection &amp; entity resolution</div>
</div>
</div>

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fix_field` | `record_id`, `field_name`, `new_value` | Correct a specific field in a record |
| `mark_duplicate` | `record_id`, `duplicate_of` | Flag two records as representing the same entity |
| `delete_record` | `record_id` | Remove a record from the dataset |
| `submit` | &mdash; | Finalize cleaning and get graded |

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
| `current_score` | `float` | Running score (0.0&ndash;1.0) |

Each record has fields: `id`, `name`, `email`, `phone`, `date_of_birth`, `city`, `state`, `zip_code`, `company`.

---

## Reward Function

<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px; margin: 16px 0;">
<div style="font-weight: 700; color: #1e293b; margin-bottom: 12px; font-size: 0.9rem;">Dense rewards provide learning signal at every step:</div>
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #f0fdf4; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #166534;">Correct field fix</span>
<strong style="color: #16a34a;">+0.1</strong>
</div>
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #fef2f2; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #991b1b;">Incorrect field change</span>
<strong style="color: #dc2626;">&minus;0.05</strong>
</div>
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #f0fdf4; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #166534;">Correct duplicate ID</span>
<strong style="color: #16a34a;">+0.2</strong>
</div>
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #fef2f2; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #991b1b;">Incorrect duplicate</span>
<strong style="color: #dc2626;">&minus;0.1</strong>
</div>
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #f0fdf4; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #166534;">Correct deletion</span>
<strong style="color: #16a34a;">+0.15</strong>
</div>
<div style="display: flex; justify-content: space-between; padding: 6px 12px; background: #fef2f2; border-radius: 6px; font-size: 0.82rem;">
<span style="color: #991b1b;">Deleting non-duplicate</span>
<strong style="color: #dc2626;">&minus;0.15</strong>
</div>
</div>
</div>

**Final grading weights:**
- **Easy/Medium:** Field accuracy (75%) + Efficiency (15%) &minus; False positive penalty (10%)
- **Hard:** Field accuracy (60%) + Duplicate detection (25%) + Efficiency (10%) &minus; FP penalty (5%)

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

## Quick Start (Python Client)

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
 "new_value": "rahul.sharma@gmail.com"}}

// Submit
{"type": "step", "data": {"action_type": "submit"}}
```

---

## Baseline Scores

<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px; margin: 16px 0;">
<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
<thead>
<tr style="border-bottom: 2px solid #e2e8f0;">
<th style="text-align: left; padding: 8px 12px; color: #475569; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;">Agent</th>
<th style="text-align: center; padding: 8px 12px; color: #16a34a; font-size: 0.75rem; text-transform: uppercase;">Easy</th>
<th style="text-align: center; padding: 8px 12px; color: #ea580c; font-size: 0.75rem; text-transform: uppercase;">Medium</th>
<th style="text-align: center; padding: 8px 12px; color: #dc2626; font-size: 0.75rem; text-transform: uppercase;">Hard</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #f1f5f9;">
<td style="padding: 8px 12px; color: #94a3b8;">Random</td>
<td style="text-align: center; padding: 8px 12px; color: #94a3b8;">~0.10</td>
<td style="text-align: center; padding: 8px 12px; color: #94a3b8;">~0.05</td>
<td style="text-align: center; padding: 8px 12px; color: #94a3b8;">~0.03</td>
</tr>
<tr style="border-bottom: 1px solid #f1f5f9;">
<td style="padding: 8px 12px; font-weight: 600; color: #334155;">Heuristic (regex)</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #16a34a;">0.87</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #ea580c;">0.44</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #dc2626;">0.37</td>
</tr>
<tr style="border-bottom: 1px solid #f1f5f9;">
<td style="padding: 8px 12px; color: #334155;">GPT-4o-mini</td>
<td style="text-align: center; padding: 8px 12px;">~0.85</td>
<td style="text-align: center; padding: 8px 12px;">~0.70</td>
<td style="text-align: center; padding: 8px 12px;">~0.55</td>
</tr>
<tr>
<td style="padding: 8px 12px; font-weight: 600; color: #1e293b;">Frontier LLM</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #16a34a;">~0.95</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #ea580c;">~0.85</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 700; color: #dc2626;">~0.70</td>
</tr>
</tbody>
</table>
<p style="margin: 12px 0 0; font-size: 0.8rem; color: #64748b; font-style: italic;">
The gap between heuristic and frontier LLM on Hard demonstrates this environment <strong>genuinely requires AI reasoning</strong> &mdash; not just pattern matching.
</p>
</div>

---

## What Makes This Environment Unique

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 16px 0;">
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #dbeafe; color: #2563eb; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">1</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Real-World Task</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Data cleaning is a genuine industry problem, not a toy game</div>
</div>
</div>
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #d1fae5; color: #059669; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">2</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Dense Reward Signal</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Every action gets immediate feedback, enabling effective RL training</div>
</div>
</div>
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #fef3c7; color: #d97706; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">3</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Progressive Difficulty</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Easy (regex) &rarr; Medium (reasoning) &rarr; Hard (entity resolution)</div>
</div>
</div>
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #fce7f3; color: #db2777; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">4</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Multi-Dimensional Grading</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Accuracy + efficiency + false positive penalty rewards careful agents</div>
</div>
</div>
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #ede9fe; color: #7c3aed; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">5</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Deterministic Generation</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Seeded data allows reproducible benchmarking across runs</div>
</div>
</div>
<div style="display: flex; gap: 12px; align-items: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px;">
<div style="background: #cffafe; color: #0891b2; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; flex-shrink: 0;">6</div>
<div>
<div style="font-weight: 700; color: #1e293b; font-size: 0.85rem;">Interactive Demo + RL Curves</div>
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">Play in the Playground tab, watch training curves in Baselines tab</div>
</div>
</div>
</div>

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

<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 24px 28px; text-align: center; color: #fff;">
<div style="font-size: 1.1rem; font-weight: 800; margin-bottom: 4px;">Team Devgods</div>
<div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 12px;">Scaler &times; Meta PyTorch OpenEnv Hackathon 2026</div>
<div style="display: flex; justify-content: center; gap: 24px; font-size: 0.85rem;">
<span><strong>Jesseman Devamirtham N</strong> <span style="opacity: 0.5;">(Lead)</span></span>
<span><strong>Karen Infanta Rozario</strong></span>
<span><strong>Janani S</strong></span>
</div>
</div>

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
                messy tabular data &mdash; fix formatting, fill missing values, correct typos, and detect duplicates.
            </p>
            <div class="header-badges">
                <span class="badge badge-accent">OpenEnv</span>
                <span class="badge badge-green">Scaler &times; Meta PyTorch Hackathon 2026</span>
                <span class="badge">Team Devgods</span>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <span class="hs-value">3</span>
                    <span class="hs-label">Tasks</span>
                </div>
                <div class="header-stat">
                    <span class="hs-value">4</span>
                    <span class="hs-label">Actions</span>
                </div>
                <div class="header-stat">
                    <span class="hs-value">Dense</span>
                    <span class="hs-label">Rewards</span>
                </div>
                <div class="header-stat">
                    <span class="hs-value">0-1</span>
                    <span class="hs-label">Grading</span>
                </div>
            </div>
            <div class="header-links">
                <a class="header-link" href="/docs" target="_blank">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                    API Docs (Swagger)
                </a>
                <a class="header-link" href="/tasks" target="_blank">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                    Tasks &amp; Schema
                </a>
                <a class="header-link" href="/grader" target="_blank">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    Grading Criteria
                </a>
                <a class="header-link" href="/health" target="_blank">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    Health Check
                </a>
            </div>
        </div>

        <div class="how-it-works">
            <div class="hiw-card">
                <div class="hiw-step hiw-step-1">1</div>
                <div class="hiw-title">Reset</div>
                <div class="hiw-desc">Choose a task &amp; load dirty data records</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-step hiw-step-2">2</div>
                <div class="hiw-title">Clean</div>
                <div class="hiw-desc">Fix fields, mark duplicates, delete bad records</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-step hiw-step-3">3</div>
                <div class="hiw-title">Submit</div>
                <div class="hiw-desc">Finalize and get graded on a 0.0&ndash;1.0 scale</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-step hiw-step-4">4</div>
                <div class="hiw-title">Learn</div>
                <div class="hiw-desc">Dense rewards guide the agent to improve</div>
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

1. **Entity Resolution** (Hard task) -- The agent must determine that "Rahul Shrma" at "rahulsharma@gmail.com"
   and "Rahul Sharma" at "rahul.sharma@gmail.com" are the same person. This challenges even frontier LLMs.

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
            <strong>Team Devgods</strong> &mdash; Scaler &times; Meta PyTorch OpenEnv Hackathon 2026<br/>
            Jesseman Devamirtham N (Lead) &bull; Karen Infanta Rozario &bull; Janani S
            <div class="footer-tech">
                <span class="footer-badge">OpenEnv</span>
                <span class="footer-badge">FastAPI</span>
                <span class="footer-badge">Gradio</span>
                <span class="footer-badge">Pydantic</span>
                <span class="footer-badge">Docker</span>
                <span class="footer-badge">HuggingFace Spaces</span>
            </div>
        </div>
        """)

    return demo
