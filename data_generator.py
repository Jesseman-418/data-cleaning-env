"""
Data generator for the Data Cleaning Environment.

Creates deterministic dirty datasets with known ground truth for each task.
Uses seeded randomness for reproducibility.
"""

import copy
import random
from typing import Dict, List, Tuple


# Base clean records pool
CLEAN_RECORDS = [
    {"id": 1, "name": "John Smith", "email": "john.smith@gmail.com", "phone": "555-123-4567", "date_of_birth": "1990-03-15", "city": "New York", "state": "New York", "zip_code": "10001", "company": "Acme Corporation"},
    {"id": 2, "name": "Sarah Johnson", "email": "sarah.j@outlook.com", "phone": "555-234-5678", "date_of_birth": "1985-07-22", "city": "Los Angeles", "state": "California", "zip_code": "90001", "company": "TechStart Inc"},
    {"id": 3, "name": "Michael Chen", "email": "mchen@yahoo.com", "phone": "555-345-6789", "date_of_birth": "1992-11-08", "city": "Chicago", "state": "Illinois", "zip_code": "60601", "company": "DataFlow Systems"},
    {"id": 4, "name": "Emily Davis", "email": "emily.davis@hotmail.com", "phone": "555-456-7890", "date_of_birth": "1988-01-30", "city": "Houston", "state": "Texas", "zip_code": "77001", "company": "Green Energy Co"},
    {"id": 5, "name": "Robert Wilson", "email": "rwilson@gmail.com", "phone": "555-567-8901", "date_of_birth": "1995-06-12", "city": "Phoenix", "state": "Arizona", "zip_code": "85001", "company": "Desert Solutions"},
    {"id": 6, "name": "Lisa Anderson", "email": "lisa.anderson@gmail.com", "phone": "555-678-9012", "date_of_birth": "1991-09-25", "city": "Philadelphia", "state": "Pennsylvania", "zip_code": "19101", "company": "HealthFirst Medical"},
    {"id": 7, "name": "David Martinez", "email": "dmartinez@outlook.com", "phone": "555-789-0123", "date_of_birth": "1987-04-18", "city": "San Antonio", "state": "Texas", "zip_code": "78201", "company": "BuildRight Construction"},
    {"id": 8, "name": "Jennifer Brown", "email": "jbrown@yahoo.com", "phone": "555-890-1234", "date_of_birth": "1993-12-05", "city": "San Diego", "state": "California", "zip_code": "92101", "company": "Pacific Ventures"},
    {"id": 9, "name": "James Taylor", "email": "jtaylor@gmail.com", "phone": "555-901-2345", "date_of_birth": "1989-08-14", "city": "Dallas", "state": "Texas", "zip_code": "75201", "company": "Lone Star Media"},
    {"id": 10, "name": "Amanda White", "email": "awhite@hotmail.com", "phone": "555-012-3456", "date_of_birth": "1994-02-28", "city": "San Jose", "state": "California", "zip_code": "95101", "company": "Silicon Valley Apps"},
    {"id": 11, "name": "Christopher Lee", "email": "clee@gmail.com", "phone": "555-111-2222", "date_of_birth": "1986-10-07", "city": "Austin", "state": "Texas", "zip_code": "73301", "company": "CodeCraft Labs"},
    {"id": 12, "name": "Jessica Garcia", "email": "jgarcia@outlook.com", "phone": "555-222-3333", "date_of_birth": "1990-05-19", "city": "Jacksonville", "state": "Florida", "zip_code": "32099", "company": "Sunshine Marketing"},
    {"id": 13, "name": "Daniel Robinson", "email": "drobinson@yahoo.com", "phone": "555-333-4444", "date_of_birth": "1983-07-31", "city": "Columbus", "state": "Ohio", "zip_code": "43085", "company": "Midwest Analytics"},
    {"id": 14, "name": "Michelle Clark", "email": "mclark@gmail.com", "phone": "555-444-5555", "date_of_birth": "1996-03-10", "city": "Charlotte", "state": "North Carolina", "zip_code": "28201", "company": "Southern Logistics"},
    {"id": 15, "name": "Kevin Thomas", "email": "kthomas@hotmail.com", "phone": "555-555-6666", "date_of_birth": "1991-11-22", "city": "San Francisco", "state": "California", "zip_code": "94101", "company": "Bay Area Consulting"},
    {"id": 16, "name": "Rachel Moore", "email": "rmoore@gmail.com", "phone": "555-666-7777", "date_of_birth": "1988-09-03", "city": "Indianapolis", "state": "Indiana", "zip_code": "46201", "company": "Hoosier Health"},
    {"id": 17, "name": "Brian Jackson", "email": "bjackson@outlook.com", "phone": "555-777-8888", "date_of_birth": "1984-06-15", "city": "Seattle", "state": "Washington", "zip_code": "98101", "company": "Rainy Day Software"},
    {"id": 18, "name": "Stephanie Harris", "email": "sharris@yahoo.com", "phone": "555-888-9999", "date_of_birth": "1992-01-27", "city": "Denver", "state": "Colorado", "zip_code": "80201", "company": "Mountain View Labs"},
    {"id": 19, "name": "Andrew Lewis", "email": "alewis@gmail.com", "phone": "555-999-0000", "date_of_birth": "1987-12-09", "city": "Nashville", "state": "Tennessee", "zip_code": "37201", "company": "Music City Media"},
    {"id": 20, "name": "Laura Walker", "email": "lwalker@hotmail.com", "phone": "555-100-2000", "date_of_birth": "1995-04-21", "city": "Portland", "state": "Oregon", "zip_code": "97201", "company": "Evergreen Designs"},
]

# State abbreviation mapping
STATE_ABBREVS = {
    "New York": "NY", "California": "CA", "Illinois": "IL", "Texas": "TX",
    "Arizona": "AZ", "Pennsylvania": "PA", "Florida": "FL", "Ohio": "OH",
    "North Carolina": "NC", "Indiana": "IN", "Washington": "WA",
    "Colorado": "CO", "Tennessee": "TN", "Oregon": "OR",
}

# Common typos for states
STATE_TYPOS = {
    "California": ["Califronia", "Californa", "Calfornia"],
    "Pennsylvania": ["Pensylvania", "Pennsylvnia", "Pennsilvania"],
    "Illinois": ["Illionis", "Illinios", "Illnois"],
    "Tennessee": ["Tennesee", "Tennesse", "Tenessee"],
    "North Carolina": ["North Carolna", "Noth Carolina"],
    "Indiana": ["Indana", "Indianna"],
    "New York": ["New Yrok", "New Yok"],
    "Texas": ["Texs", "Txas"],
    "Florida": ["Flordia", "Florda"],
    "Colorado": ["Colordo", "Colrado"],
}

COMPANY_TYPOS = {
    "Acme Corporation": "Acme Corportation",
    "TechStart Inc": "TechStart Ic",
    "DataFlow Systems": "DataFow Systems",
    "Green Energy Co": "Gren Energy Co",
    "HealthFirst Medical": "HeathFirst Medical",
    "Pacific Ventures": "Pacfic Ventures",
    "Silicon Valley Apps": "Silcon Valley Apps",
    "CodeCraft Labs": "CodCraft Labs",
    "Sunshine Marketing": "Sunshin Marketing",
    "Midwest Analytics": "Midwst Analytics",
}


def _dirty_date(date_str: str, rng: random.Random) -> str:
    """Convert YYYY-MM-DD to a messy format."""
    parts = date_str.split("-")
    year, month, day = parts[0], parts[1], parts[2]
    fmt = rng.choice(["slash_mdy", "slash_dmy", "dot", "no_pad", "verbose"])
    if fmt == "slash_mdy":
        return f"{month}/{day}/{year}"
    elif fmt == "slash_dmy":
        return f"{day}/{month}/{year}"
    elif fmt == "dot":
        return f"{month}.{day}.{year}"
    elif fmt == "no_pad":
        return f"{int(month)}-{int(day)}-{year}"
    else:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{months[int(month)-1]} {int(day)}, {year}"


def _dirty_phone(phone: str, rng: random.Random) -> str:
    """Mess up phone formatting."""
    digits = phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    fmt = rng.choice(["dots", "parens", "spaces", "plain", "plus1"])
    if fmt == "dots":
        return f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
    elif fmt == "parens":
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif fmt == "spaces":
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    elif fmt == "plain":
        return digits
    else:
        return f"+1{digits}"


def _dirty_email(email: str, rng: random.Random) -> str:
    """Mess up email casing."""
    fmt = rng.choice(["upper", "mixed", "caps_name"])
    if fmt == "upper":
        return email.upper()
    elif fmt == "mixed":
        return "".join(c.upper() if rng.random() > 0.5 else c for c in email)
    else:
        name, domain = email.split("@")
        return f"{name.upper()}@{domain}"


def generate_task_easy(seed: int = 42) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Task 1: Format Standardization (Easy).
    5 records with formatting issues in dates, phones, and emails.
    Returns: (dirty_records, clean_records, issue_map)
    """
    rng = random.Random(seed)
    records = [copy.deepcopy(r) for r in CLEAN_RECORDS[:5]]
    clean = [copy.deepcopy(r) for r in records]
    dirty = [copy.deepcopy(r) for r in records]
    issues = {}

    for rec in dirty:
        rid = rec["id"]
        issues[rid] = []

        # Dirty the date
        rec["date_of_birth"] = _dirty_date(rec["date_of_birth"], rng)
        issues[rid].append(("date_of_birth", clean[rid - 1]["date_of_birth"]))

        # Dirty the phone
        rec["phone"] = _dirty_phone(rec["phone"], rng)
        issues[rid].append(("phone", clean[rid - 1]["phone"]))

        # Dirty the email
        rec["email"] = _dirty_email(rec["email"], rng)
        issues[rid].append(("email", clean[rid - 1]["email"]))

    total_issues = sum(len(v) for v in issues.values())
    return dirty, clean, {"issues": issues, "total": total_issues, "duplicates": []}


def generate_task_medium(seed: int = 42) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Task 2: Missing Values + Typo Correction (Medium).
    10 records with missing values, typos, and some format issues.
    Returns: (dirty_records, clean_records, issue_map)
    """
    rng = random.Random(seed)
    records = [copy.deepcopy(r) for r in CLEAN_RECORDS[:10]]
    clean = [copy.deepcopy(r) for r in records]
    dirty = [copy.deepcopy(r) for r in records]
    issues = {}

    for i, rec in enumerate(dirty):
        rid = rec["id"]
        issues[rid] = []

        # Format issues on some records
        if rng.random() > 0.4:
            rec["date_of_birth"] = _dirty_date(rec["date_of_birth"], rng)
            issues[rid].append(("date_of_birth", clean[i]["date_of_birth"]))

        if rng.random() > 0.5:
            rec["phone"] = _dirty_phone(rec["phone"], rng)
            issues[rid].append(("phone", clean[i]["phone"]))

        # Missing values
        if rng.random() > 0.5:
            rec["city"] = ""
            issues[rid].append(("city", clean[i]["city"]))

        if rng.random() > 0.6:
            rec["email"] = ""
            issues[rid].append(("email", clean[i]["email"]))

        if rng.random() > 0.7:
            rec["company"] = ""
            issues[rid].append(("company", clean[i]["company"]))

        # State typos
        state = rec["state"]
        if state in STATE_TYPOS and rng.random() > 0.4:
            rec["state"] = rng.choice(STATE_TYPOS[state])
            issues[rid].append(("state", clean[i]["state"]))

        # Company typos
        company = rec["company"]
        if company in COMPANY_TYPOS and rec["company"] != "" and rng.random() > 0.5:
            rec["company"] = COMPANY_TYPOS[company]
            issues[rid].append(("company", clean[i]["company"]))

    total_issues = sum(len(v) for v in issues.values())
    return dirty, clean, {"issues": issues, "total": total_issues, "duplicates": []}


def generate_task_hard(seed: int = 42) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Task 3: Full Data Pipeline (Hard).
    15 records with ALL issue types + duplicate records + outliers.
    Returns: (dirty_records, clean_records, issue_map)
    """
    rng = random.Random(seed)
    records = [copy.deepcopy(r) for r in CLEAN_RECORDS[:12]]
    clean = [copy.deepcopy(r) for r in records]

    # Add 3 duplicate records (variations of existing ones)
    duplicates = []

    dup1 = copy.deepcopy(records[0])
    dup1["id"] = 13
    dup1["name"] = "Jon Smith"  # Typo in name
    dup1["email"] = "johnsmith@gmail.com"  # Slightly different email
    dup1["phone"] = "5551234567"  # Different format
    records.append(dup1)
    duplicates.append((13, 1))

    dup2 = copy.deepcopy(records[2])
    dup2["id"] = 14
    dup2["name"] = "Michael Chen"
    dup2["email"] = "michael.chen@yahoo.com"  # Different email
    dup2["phone"] = "(555) 345-6789"  # Different format
    dup2["city"] = "chicago"  # Lowercase
    records.append(dup2)
    duplicates.append((14, 3))

    dup3 = copy.deepcopy(records[4])
    dup3["id"] = 15
    dup3["name"] = "Rob Wilson"  # Shortened name
    dup3["email"] = "rwilson@gmail.com"
    dup3["date_of_birth"] = "06/12/1995"  # Different format
    records.append(dup3)
    duplicates.append((15, 5))

    dirty = [copy.deepcopy(r) for r in records]
    issues = {}

    for i, rec in enumerate(dirty):
        rid = rec["id"]
        issues[rid] = []

        # Skip duplicates for now (they have their own issues)
        if rid > 12:
            continue

        # Format issues
        if rng.random() > 0.3:
            rec["date_of_birth"] = _dirty_date(rec["date_of_birth"], rng)
            issues[rid].append(("date_of_birth", clean[i]["date_of_birth"]))

        if rng.random() > 0.4:
            rec["phone"] = _dirty_phone(rec["phone"], rng)
            issues[rid].append(("phone", clean[i]["phone"]))

        if rng.random() > 0.5:
            rec["email"] = _dirty_email(rec["email"], rng)
            issues[rid].append(("email", clean[i]["email"]))

        # Missing values
        if rng.random() > 0.6:
            rec["city"] = ""
            issues[rid].append(("city", clean[i]["city"]))

        if rng.random() > 0.7:
            rec["company"] = ""
            issues[rid].append(("company", clean[i]["company"]))

        # State typos
        state = rec["state"]
        if state in STATE_TYPOS and rng.random() > 0.3:
            rec["state"] = rng.choice(STATE_TYPOS[state])
            issues[rid].append(("state", clean[i]["state"]))

        # Company typos
        company = rec["company"]
        if company in COMPANY_TYPOS and rec["company"] != "" and rng.random() > 0.4:
            rec["company"] = COMPANY_TYPOS[company]
            issues[rid].append(("company", clean[i]["company"]))

        # Outlier injection: unrealistic zip codes or dates
        if rid == 6:
            rec["zip_code"] = "00000"
            issues[rid].append(("zip_code", clean[i]["zip_code"]))
        if rid == 9:
            rec["date_of_birth"] = "1820-08-14"  # Impossible age
            issues[rid].append(("date_of_birth", clean[i]["date_of_birth"]))

    total_field_issues = sum(len(v) for v in issues.values())
    total_issues = total_field_issues + len(duplicates)  # Duplicates count as issues

    return dirty, clean, {
        "issues": issues,
        "total": total_issues,
        "duplicates": duplicates,
        "duplicate_ids": [d[0] for d in duplicates],
    }


# Task registry
TASKS = {
    "easy_format_standardization": {
        "id": "easy_format_standardization",
        "name": "Format Standardization",
        "difficulty": "easy",
        "description": (
            "Fix formatting issues in 5 customer records. "
            "Dates should be YYYY-MM-DD format. "
            "Phone numbers should be XXX-XXX-XXXX format. "
            "Emails should be lowercase. "
            "Use fix_field action to correct each field, then submit when done."
        ),
        "generator": generate_task_easy,
        "max_actions": 30,
        "fields": ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"],
    },
    "medium_missing_and_typos": {
        "id": "medium_missing_and_typos",
        "name": "Missing Values & Typo Correction",
        "difficulty": "medium",
        "description": (
            "Fix 10 customer records with missing values, typos, and format issues. "
            "Fill in missing fields using context (e.g., zip code can help determine city). "
            "Fix misspelled state and company names. "
            "Standardize date and phone formats. "
            "Use fix_field action to correct each field, then submit when done."
        ),
        "generator": generate_task_medium,
        "max_actions": 60,
        "fields": ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"],
    },
    "hard_full_pipeline": {
        "id": "hard_full_pipeline",
        "name": "Full Data Cleaning Pipeline",
        "difficulty": "hard",
        "description": (
            "Clean 15 customer records with formatting issues, missing values, typos, "
            "outliers, AND duplicate records. "
            "Fix all field-level issues as before. "
            "Additionally, identify duplicate records using mark_duplicate action "
            "(records that represent the same person with slightly different data). "
            "Remove identified duplicates using delete_record. "
            "Watch for outlier values (impossible dates, invalid zip codes). "
            "Submit when done."
        ),
        "generator": generate_task_hard,
        "max_actions": 100,
        "fields": ["name", "email", "phone", "date_of_birth", "city", "state", "zip_code", "company"],
    },
}
