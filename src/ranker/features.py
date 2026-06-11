"""Per-candidate feature extraction.

Everything here reads only the candidate record; no network, no models.
Features feed scoring.py (weights) and reasoning.py (explanations).
"""

import datetime as _dt
import re

from . import jd_profile as jd

_REF = _dt.date.fromisoformat(jd.REFERENCE_DATE)
_WORD_RE = re.compile(r"[a-z][a-z0-9&+./-]*")


def _text_of(candidate: dict) -> str:
    """All narrative text: summary + headline + per-job titles/descriptions."""
    p = candidate["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def candidate_document(candidate: dict) -> str:
    """Document used by the semantic layer."""
    return _text_of(candidate)


def evidence_features(candidate: dict) -> dict:
    """Mine career-history descriptions for JD evidence, weighted by recency.

    A hit in the current role counts fully; each step back in history decays
    by 0.75. This rewards people doing retrieval/ranking work NOW over people
    who touched it once long ago.
    """
    fam_scores = {name: 0.0 for name in jd.EVIDENCE_FAMILIES}
    fam_hits = {name: [] for name in jd.EVIDENCE_FAMILIES}

    jobs = candidate.get("career_history", [])
    texts = [(candidate["profile"].get("summary", "").lower(), 0.8)]
    texts += [
        (
            (j.get("title", "") + " " + j.get("description", "")).lower(),
            0.75 ** i,
        )
        for i, j in enumerate(jobs)
    ]

    for name, fam in jd.EVIDENCE_FAMILIES.items():
        for text, recency in texts:
            hits = [t for t in fam["terms"] if t in text]
            if hits:
                # diminishing returns within a single text block
                fam_scores[name] += recency * min(1.0, 0.4 + 0.2 * len(hits))
                fam_hits[name].extend(hits)
        fam_scores[name] = min(1.5, fam_scores[name])

    full = _text_of(candidate)
    cv_hits = sum(1 for t in jd.CV_SPEECH_TERMS if t in full)
    research_hits = sum(1 for t in jd.RESEARCH_TERMS if t in full)

    return {
        "families": fam_scores,
        "hits": {k: sorted(set(v)) for k, v in fam_hits.items()},
        "cv_speech_hits": cv_hits,
        "research_hits": research_hits,
    }


def title_relevance(candidate: dict) -> tuple[float, float]:
    """(current-title relevance, best historical-title relevance)."""
    cur = jd.TITLE_RELEVANCE.get(
        candidate["profile"]["current_title"].lower(), jd.NON_TECHNICAL_FLOOR
    )
    past = max(
        (
            jd.TITLE_RELEVANCE.get(j["title"].lower(), 0.0)
            for j in candidate.get("career_history", [])
        ),
        default=0.0,
    )
    return cur, past


def corroborated_skills(candidate: dict) -> list[tuple[str, float, float]]:
    """JD-relevant skills with a trust factor in [0, 1].

    Trust is the anti-keyword-stuffing device: a listed skill earns weight
    only when corroborated by usage duration, endorsements, or a Redrob
    assessment score. 'Expert in Pinecone, 0 months used, 0 endorsements'
    contributes almost nothing.
    """
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})
    out = []
    for s in candidate.get("skills", []):
        name = s["name"].lower()
        rel = jd.RELEVANT_SKILLS.get(name)
        if rel is None:
            continue
        dur = s.get("duration_months", 0)
        endo = s.get("endorsements", 0)
        assess = assessments.get(s["name"], None)

        trust = 0.0
        trust += min(0.5, dur / 48.0)                      # up to 0.5 for 4y use
        trust += min(0.25, endo / 80.0)                    # up to 0.25
        if assess is not None:
            trust += 0.25 * (assess / 100.0)               # up to 0.25
        prof_mult = {
            "beginner": 0.4, "intermediate": 0.7,
            "advanced": 1.0, "expert": 1.1,
        }.get(s.get("proficiency", "intermediate"), 0.7)
        # uncorroborated claims are nearly worthless regardless of proficiency
        if dur == 0 and endo < 5 and assess is None:
            trust = 0.05
        out.append((s["name"], rel, min(1.0, trust) * prof_mult))
    return out


def company_profile(candidate: dict) -> dict:
    """Consulting-only careers, product-company exposure, ML-at-product."""
    jobs = candidate.get("career_history", [])
    n = len(jobs)
    services = 0
    product_months = 0
    for j in jobs:
        comp = j["company"].lower()
        ind = j["industry"].lower()
        if comp in jd.CONSULTING_COMPANIES or ind in jd.SERVICES_INDUSTRIES:
            services += 1
        elif ind in jd.PRODUCT_INDUSTRIES:
            product_months += j["duration_months"]
    return {
        "consulting_only": n > 0 and services == n,
        "product_months": product_months,
        "n_jobs": n,
    }


def tenure_pattern(candidate: dict) -> dict:
    """Job-hopping / title-chasing detection."""
    jobs = candidate.get("career_history", [])
    n = len(jobs)
    if n == 0:
        return {"avg_tenure": 0.0, "hopper": False}
    avg = sum(j["duration_months"] for j in jobs) / n
    hopper = n >= 4 and avg < 20
    return {"avg_tenure": avg, "hopper": hopper}


def behavioral(candidate: dict) -> dict:
    s = candidate["redrob_signals"]
    last = _dt.date.fromisoformat(s["last_active_date"])
    months_inactive = (_REF - last).days / 30.44
    return {
        "months_inactive": months_inactive,
        "response_rate": s["recruiter_response_rate"],
        "open_to_work": s["open_to_work_flag"],
        "interview_completion": s["interview_completion_rate"],
        "notice_days": s["notice_period_days"],
        "github": s["github_activity_score"],
        "willing_to_relocate": s["willing_to_relocate"],
        "work_mode": s["preferred_work_mode"],
    }


def location_score(candidate: dict, willing_to_relocate: bool) -> float:
    p = candidate["profile"]
    loc = p.get("location", "").lower()
    if p.get("country", "").lower() != "india":
        base = jd.ABROAD_DEFAULT
    else:
        base = jd.INDIA_DEFAULT
        for city, score in jd.LOCATION_SCORES.items():
            if city in loc:
                base = score
                break
    if willing_to_relocate and base < 0.92:
        base = min(0.92, base + jd.RELOCATION_BONUS)
    return base


def yoe_score(years: float) -> float:
    lo_acc, hi_acc = jd.YOE_ACCEPTABLE
    lo_ideal, hi_ideal = jd.YOE_IDEAL
    if lo_ideal <= years <= hi_ideal:
        return 1.0
    if lo_acc <= years <= hi_acc:
        return 0.92
    if years < lo_acc:  # ramp from 0.2 at 2y to 0.92 at 5y
        return max(0.2, 0.92 - (lo_acc - years) * 0.24)
    # beyond 9y: gentle decay — "seriously consider if signals are strong"
    return max(0.35, 0.92 - (years - hi_acc) * 0.08)
