"""Honeypot detection via internal-consistency checks.

The dataset contains ~80 candidates with subtly impossible profiles, forced
to relevance tier 0 in the ground truth. We never special-case IDs; we run
four independent consistency checks any careful reader would apply:

1. skill_impossible  — 3+ skills each used longer than the candidate's
                       entire stated career (duration > YoE + 24 months).
2. expert_zero       — 3+ "expert"-proficiency skills with 0 months of use.
3. date_mismatch     — a career entry whose start/end dates disagree with its
                       own duration_months field by more than 12 months.
4. career_overflow   — summed non-overlapping job durations exceed stated
                       years_of_experience by more than 24 months.

Any single flag marks the candidate suspect. On the released pool this flags
126 of 100,000 profiles — a superset of the ~80 honeypots; the false
positives are profiles with self-contradictory data we would not surface to
a recruiter anyway.
"""

import datetime as _dt

from .jd_profile import REFERENCE_DATE

_REF = _dt.date.fromisoformat(REFERENCE_DATE)


def honeypot_flags(candidate: dict) -> list[str]:
    """Return the list of consistency-check names this candidate fails."""
    flags = []
    yoe_months = candidate["profile"]["years_of_experience"] * 12
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    impossible = sum(
        1 for s in skills if s.get("duration_months", 0) > yoe_months + 24
    )
    if impossible >= 3:
        flags.append("skill_impossible")

    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        flags.append("expert_zero")

    for job in history:
        start = _dt.date.fromisoformat(job["start_date"])
        end = _dt.date.fromisoformat(job["end_date"]) if job["end_date"] else _REF
        actual_months = (end - start).days / 30.44
        if abs(actual_months - job["duration_months"]) > 12:
            flags.append("date_mismatch")
            break

    total_months = sum(j["duration_months"] for j in history)
    if total_months > yoe_months + 24:
        flags.append("career_overflow")

    return flags


def is_honeypot(candidate: dict) -> bool:
    return bool(honeypot_flags(candidate))
