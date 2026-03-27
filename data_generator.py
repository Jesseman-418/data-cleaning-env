"""
Data generator for the Data Cleaning Environment.

Creates deterministic dirty datasets with known ground truth for each task.
Uses seeded randomness for reproducibility.
"""

import copy
import random
from typing import Dict, List, Tuple


# Base clean records pool — Indian customer database
CLEAN_RECORDS = [
    {"id": 1, "name": "Rahul Sharma", "email": "rahul.sharma@gmail.com", "phone": "982-314-5670", "date_of_birth": "1990-03-15", "city": "Mumbai", "state": "Maharashtra", "zip_code": "400001", "company": "Tata Consultancy Services"},
    {"id": 2, "name": "Priya Nair", "email": "priya.nair@outlook.com", "phone": "944-287-3456", "date_of_birth": "1985-07-22", "city": "Bangalore", "state": "Karnataka", "zip_code": "560001", "company": "Infosys Ltd"},
    {"id": 3, "name": "Amit Patel", "email": "amit.patel@yahoo.com", "phone": "879-456-1230", "date_of_birth": "1992-11-08", "city": "Ahmedabad", "state": "Gujarat", "zip_code": "380001", "company": "Reliance Industries"},
    {"id": 4, "name": "Sneha Iyer", "email": "sneha.iyer@hotmail.com", "phone": "900-712-8904", "date_of_birth": "1988-01-30", "city": "Chennai", "state": "Tamil Nadu", "zip_code": "600001", "company": "Zoho Corporation"},
    {"id": 5, "name": "Vikram Reddy", "email": "vreddy@gmail.com", "phone": "863-509-4123", "date_of_birth": "1995-06-12", "city": "Hyderabad", "state": "Telangana", "zip_code": "500001", "company": "Cyient Ltd"},
    {"id": 6, "name": "Ananya Deshmukh", "email": "ananya.d@gmail.com", "phone": "773-841-2056", "date_of_birth": "1991-09-25", "city": "Pune", "state": "Maharashtra", "zip_code": "411001", "company": "Persistent Systems"},
    {"id": 7, "name": "Karthik Menon", "email": "kmenon@outlook.com", "phone": "948-623-7081", "date_of_birth": "1987-04-18", "city": "Kochi", "state": "Kerala", "zip_code": "682001", "company": "UST Global"},
    {"id": 8, "name": "Divya Gupta", "email": "divya.g@yahoo.com", "phone": "981-035-6742", "date_of_birth": "1993-12-05", "city": "Delhi", "state": "Delhi", "zip_code": "110001", "company": "HCL Technologies"},
    {"id": 9, "name": "Suresh Kumar", "email": "skumar@gmail.com", "phone": "701-894-5230", "date_of_birth": "1989-08-14", "city": "Jaipur", "state": "Rajasthan", "zip_code": "302001", "company": "Genpact"},
    {"id": 10, "name": "Meera Joshi", "email": "meera.joshi@hotmail.com", "phone": "886-213-4790", "date_of_birth": "1994-02-28", "city": "Noida", "state": "Uttar Pradesh", "zip_code": "201301", "company": "Wipro Ltd"},
    {"id": 11, "name": "Arjun Bhat", "email": "arjun.bhat@gmail.com", "phone": "934-175-8602", "date_of_birth": "1986-10-07", "city": "Mysore", "state": "Karnataka", "zip_code": "570001", "company": "Mindtree Ltd"},
    {"id": 12, "name": "Lakshmi Rao", "email": "lrao@outlook.com", "phone": "809-362-1475", "date_of_birth": "1990-05-19", "city": "Visakhapatnam", "state": "Andhra Pradesh", "zip_code": "530001", "company": "Tech Mahindra"},
    {"id": 13, "name": "Rajesh Pillai", "email": "rpillai@yahoo.com", "phone": "956-048-3291", "date_of_birth": "1983-07-31", "city": "Thiruvananthapuram", "state": "Kerala", "zip_code": "695001", "company": "IBS Software"},
    {"id": 14, "name": "Nandini Singh", "email": "nsingh@gmail.com", "phone": "771-529-6843", "date_of_birth": "1996-03-10", "city": "Lucknow", "state": "Uttar Pradesh", "zip_code": "226001", "company": "Newgen Software"},
    {"id": 15, "name": "Sanjay Verma", "email": "sverma@hotmail.com", "phone": "832-461-7908", "date_of_birth": "1991-11-22", "city": "Chandigarh", "state": "Punjab", "zip_code": "160001", "company": "Nagarro"},
    {"id": 16, "name": "Pooja Kulkarni", "email": "pkulkarni@gmail.com", "phone": "902-783-5146", "date_of_birth": "1988-09-03", "city": "Nagpur", "state": "Maharashtra", "zip_code": "440001", "company": "KPIT Technologies"},
    {"id": 17, "name": "Deepak Choudhary", "email": "dchoudhary@outlook.com", "phone": "814-690-2357", "date_of_birth": "1984-06-15", "city": "Kolkata", "state": "West Bengal", "zip_code": "700001", "company": "Cognizant"},
    {"id": 18, "name": "Swati Mishra", "email": "smishra@yahoo.com", "phone": "936-127-8064", "date_of_birth": "1992-01-27", "city": "Bhopal", "state": "Madhya Pradesh", "zip_code": "462001", "company": "Mphasis Ltd"},
    {"id": 19, "name": "Arun Natarajan", "email": "arun.n@gmail.com", "phone": "844-058-9316", "date_of_birth": "1987-12-09", "city": "Coimbatore", "state": "Tamil Nadu", "zip_code": "641001", "company": "Hexaware Technologies"},
    {"id": 20, "name": "Kavitha Hegde", "email": "khegde@hotmail.com", "phone": "978-346-0152", "date_of_birth": "1995-04-21", "city": "Mangalore", "state": "Karnataka", "zip_code": "575001", "company": "Robosoft Technologies"},
]

# State abbreviation mapping (Indian states)
STATE_ABBREVS = {
    "Maharashtra": "MH", "Karnataka": "KA", "Gujarat": "GJ", "Tamil Nadu": "TN",
    "Telangana": "TS", "Kerala": "KL", "Delhi": "DL", "Rajasthan": "RJ",
    "Uttar Pradesh": "UP", "Andhra Pradesh": "AP", "Punjab": "PB",
    "West Bengal": "WB", "Madhya Pradesh": "MP",
}

# Common typos for Indian states
STATE_TYPOS = {
    "Maharashtra": ["Maharshtra", "Maharastra", "Maharashthra"],
    "Karnataka": ["Karnatka", "Karntaka", "Karantaka"],
    "Tamil Nadu": ["Tamil Naidu", "Tamilnadu", "Tamil Ndu"],
    "Telangana": ["Telangna", "Telegana", "Telengana"],
    "Kerala": ["Kerla", "Kerela", "Keralla"],
    "Uttar Pradesh": ["Uttar Pradsh", "Utter Pradesh"],
    "Rajasthan": ["Rajasthn", "Rajsthan"],
    "Gujarat": ["Gujrat", "Gujerat"],
    "Andhra Pradesh": ["Andhra Pradsh", "Andra Pradesh"],
    "West Bengal": ["West Bangal", "West Bengel"],
}

COMPANY_TYPOS = {
    "Tata Consultancy Services": "Tata Consultany Services",
    "Infosys Ltd": "Infosys Ldt",
    "Reliance Industries": "Reliance Industires",
    "Zoho Corporation": "Zoho Corporaton",
    "Persistent Systems": "Persistant Systems",
    "HCL Technologies": "HCL Technolgies",
    "Wipro Ltd": "Wipro Ldt",
    "Mindtree Ltd": "Mindtree Ldt",
    "Tech Mahindra": "Tech Mahinrda",
    "Cognizant": "Cognizent",
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
    dup1["name"] = "Rahul Shrma"  # Typo in name
    dup1["email"] = "rahulsharma@gmail.com"  # Slightly different email
    dup1["phone"] = "9823145670"  # Different format
    records.append(dup1)
    duplicates.append((13, 1))

    dup2 = copy.deepcopy(records[2])
    dup2["id"] = 14
    dup2["name"] = "Amit Patel"
    dup2["email"] = "amitpatel@yahoo.com"  # Different email
    dup2["phone"] = "(879) 456-1230"  # Different format
    dup2["city"] = "ahmedabad"  # Lowercase
    records.append(dup2)
    duplicates.append((14, 3))

    dup3 = copy.deepcopy(records[4])
    dup3["id"] = 15
    dup3["name"] = "Vikram R"  # Shortened name
    dup3["email"] = "vreddy@gmail.com"
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
