"""
Grader for the Data Cleaning Environment.

Scores agent performance on data cleaning tasks from 0.0 to 1.0.
Deterministic and reproducible — compares final dataset to ground truth.
"""

from typing import Dict, List, Set, Tuple


def grade_episode(
    final_records: List[Dict],
    clean_records: List[Dict],
    issue_map: Dict,
    actions_taken: int,
    max_actions: int,
    marked_duplicates: Set[Tuple[int, int]],
    deleted_records: Set[int],
) -> Dict:
    """
    Grade a completed episode.

    Returns a dict with:
    - score: float 0.0-1.0
    - field_accuracy: float (% of dirty fields correctly fixed)
    - false_positives: int (clean fields incorrectly changed)
    - duplicate_accuracy: float (% of duplicates correctly found)
    - efficiency: float (actions used / max actions, lower is better)
    - breakdown: dict with detailed scoring
    """
    issues = issue_map.get("issues", {})
    expected_duplicates = issue_map.get("duplicates", [])
    total_issues = issue_map.get("total", 0)

    if total_issues == 0:
        return {
            "score": 0.999,
            "field_accuracy": 1.0,
            "false_positives": 0,
            "duplicate_accuracy": 1.0,
            "efficiency": 1.0,
            "breakdown": {},
        }

    # Build lookup of final records by ID
    final_by_id = {r["id"]: r for r in final_records}
    clean_by_id = {r["id"]: r for r in clean_records}

    # Score field fixes
    fields_correct = 0
    fields_total = 0
    false_positives = 0

    for rid_str, field_issues in issues.items():
        rid = int(rid_str) if isinstance(rid_str, str) else rid_str
        if not field_issues:
            continue

        for field_name, expected_value in field_issues:
            fields_total += 1
            if rid in final_by_id:
                actual = final_by_id[rid].get(field_name, "")
                # Normalize comparison
                if _normalize(actual) == _normalize(expected_value):
                    fields_correct += 1

    # Check for false positives (clean fields that were incorrectly changed)
    for rid, clean_rec in clean_by_id.items():
        if rid in final_by_id:
            final_rec = final_by_id[rid]
            issue_fields = set()
            if rid in issues:
                issue_fields = {f for f, _ in issues[rid]}
            for field in clean_rec:
                if field == "id":
                    continue
                if field not in issue_fields:
                    if _normalize(final_rec.get(field, "")) != _normalize(clean_rec[field]):
                        false_positives += 1

    field_accuracy = fields_correct / fields_total if fields_total > 0 else 1.0

    # Score duplicate detection
    duplicate_accuracy = 1.0
    if expected_duplicates:
        correct_dups = 0
        for dup_id, original_id in expected_duplicates:
            # Check if the duplicate was marked or deleted
            pair_found = False
            for m1, m2 in marked_duplicates:
                if (m1 == dup_id and m2 == original_id) or (m1 == original_id and m2 == dup_id):
                    pair_found = True
                    break
            if pair_found or dup_id in deleted_records:
                correct_dups += 1
        duplicate_accuracy = correct_dups / len(expected_duplicates)

    # Efficiency score (using fewer actions is better)
    efficiency = max(0.0, 1.0 - (actions_taken / max_actions))

    # False positive penalty
    fp_penalty = min(false_positives * 0.05, 0.3)  # Max 30% penalty

    # Calculate total field issues vs duplicate issues weight
    field_issue_count = sum(len(v) for v in issues.values())
    dup_issue_count = len(expected_duplicates)

    if dup_issue_count > 0:
        # Hard task: 60% field accuracy, 25% duplicate detection, 10% efficiency, 5% penalty
        raw_score = (
            field_accuracy * 0.60
            + duplicate_accuracy * 0.25
            + efficiency * 0.10
            + (1.0 - fp_penalty) * 0.05
        )
    else:
        # Easy/Medium: 75% field accuracy, 15% efficiency, 10% penalty
        raw_score = (
            field_accuracy * 0.75
            + efficiency * 0.15
            + (1.0 - fp_penalty) * 0.10
        )

    score = max(0.001, min(0.999, raw_score))

    return {
        "score": round(score, 4),
        "field_accuracy": round(field_accuracy, 4),
        "fields_correct": fields_correct,
        "fields_total": fields_total,
        "false_positives": false_positives,
        "duplicate_accuracy": round(duplicate_accuracy, 4),
        "duplicates_found": sum(1 for d in expected_duplicates if d[0] in deleted_records or any(
            (m1 == d[0] and m2 == d[1]) or (m1 == d[1] and m2 == d[0])
            for m1, m2 in marked_duplicates
        )),
        "duplicates_expected": len(expected_duplicates),
        "efficiency": round(efficiency, 4),
        "actions_taken": actions_taken,
        "max_actions": max_actions,
        "breakdown": {
            "field_score": round(field_accuracy, 4),
            "duplicate_score": round(duplicate_accuracy, 4),
            "efficiency_score": round(efficiency, 4),
            "false_positive_penalty": round(fp_penalty, 4),
        },
    }


def _normalize(value) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()
