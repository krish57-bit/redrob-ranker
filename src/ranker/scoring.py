"""Component scoring and final score combination.

final = base_fit × penalty_multiplier × behavioral_multiplier
        (honeypots are floored to ~0 by the consistency gate)

base_fit is a weighted sum of seven interpretable components; penalties
encode the JD's explicit disqualifiers; the behavioral multiplier encodes
"a perfect-on-paper candidate who hasn't logged in for 6 months and has a
5% response rate is not actually available".
"""

from dataclasses import dataclass, field

from . import features as F
from .honeypots import honeypot_flags

WEIGHTS = {
    "evidence": 0.32,   # what their career history shows they BUILT
    "title": 0.18,      # who they are now (kills keyword stuffers)
    "skills": 0.16,     # corroborated, trust-weighted skills
    "semantic": 0.12,   # JD <-> profile narrative similarity
    "yoe": 0.10,
    "location": 0.08,
    "logistics": 0.04,  # notice period
}


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float = 0.0
    components: dict = field(default_factory=dict)
    penalties: list = field(default_factory=list)
    behavioral_mult: float = 1.0
    honeypot: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)


def _evidence_component(ev: dict) -> float:
    fam = ev["families"]
    core = (fam["retrieval"] + fam["ranking"]) / 2 / 1.5     # normalise to ~[0,1]
    nlp = fam["nlp_llm"] / 1.5
    prod = fam["production"] / 1.5
    # production multiplies core evidence: built it AND shipped it
    score = core * (0.55 + 0.45 * min(1.0, prod * 1.5)) + 0.25 * nlp
    return min(1.0, score)


def _skills_component(skills: list) -> float:
    total = sum(rel * trust for _, rel, trust in skills)
    return min(1.0, total / 4.0)  # ~4 strong corroborated core skills => 1.0


def _logistics_component(notice_days: int) -> float:
    if notice_days <= 30:
        return 1.0
    if notice_days <= 60:
        return 0.8
    if notice_days <= 90:
        return 0.6
    return 0.45


def _penalties(candidate: dict, ev: dict, comp: dict, tenure: dict,
               beh: dict, title_cur: float) -> tuple[float, list[str]]:
    mult, reasons = 1.0, []

    if title_cur == 0.0:
        # Non-technical current title (HR/Marketing/Sales/...) — the explicit
        # keyword-stuffer trap. Skills cannot rescue this.
        mult *= 0.05
        reasons.append("non_technical_title")

    if comp["consulting_only"]:
        mult *= 0.15
        reasons.append("consulting_only_career")

    fam = ev["families"]
    technical_evidence = fam["retrieval"] + fam["ranking"] + fam["nlp_llm"]
    if ev["cv_speech_hits"] >= 3 and technical_evidence < 0.6:
        mult *= 0.40
        reasons.append("cv_speech_only")

    if ev["research_hits"] >= 4 and fam["production"] < 0.3:
        mult *= 0.40
        reasons.append("research_no_production")

    if tenure["hopper"]:
        mult *= 0.70
        reasons.append("job_hopper")

    # "Hasn't written production code recently": no current technical role
    history = candidate.get("career_history", [])
    if history and not history[0]["is_current"] and beh["months_inactive"] > 3:
        mult *= 0.85
        reasons.append("no_current_role")

    return mult, reasons


def _behavioral_multiplier(beh: dict) -> float:
    m = 1.0
    mi = beh["months_inactive"]
    if mi > 6:
        m *= 0.55
    elif mi > 3:
        m *= 0.75
    elif mi > 1:
        m *= 0.92

    rr = beh["response_rate"]
    if rr < 0.15:
        m *= 0.55
    elif rr < 0.40:
        m *= 0.75
    elif rr < 0.70:
        m *= 0.90

    if not beh["open_to_work"]:
        m *= 0.85
    if beh["interview_completion"] < 0.5:
        m *= 0.85
    return max(0.30, m)


def score_candidate(candidate: dict, semantic_sim: float) -> ScoredCandidate:
    cid = candidate["candidate_id"]
    sc = ScoredCandidate(candidate_id=cid)

    sc.honeypot = honeypot_flags(candidate)

    ev = F.evidence_features(candidate)
    title_cur, title_past = F.title_relevance(candidate)
    skills = F.corroborated_skills(candidate)
    comp = F.company_profile(candidate)
    tenure = F.tenure_pattern(candidate)
    beh = F.behavioral(candidate)

    components = {
        "evidence": _evidence_component(ev),
        "title": min(1.0, 0.8 * title_cur + 0.2 * title_past),
        "skills": _skills_component(skills),
        "semantic": semantic_sim,
        "yoe": F.yoe_score(candidate["profile"]["years_of_experience"]),
        "location": F.location_score(candidate, beh["willing_to_relocate"]),
        "logistics": _logistics_component(beh["notice_days"]),
    }

    base = sum(WEIGHTS[k] * v for k, v in components.items())

    # product-company ML experience bonus (the JD's ideal profile);
    # uncapped — rank.py normalises by the pool maximum, preserving
    # resolution among the very top candidates instead of saturating at 1.0
    if comp["product_months"] >= 24 and components["evidence"] > 0.3:
        base += 0.05
    # external validation nice-to-have
    if beh["github"] >= 40:
        base += 0.02

    pen_mult, pen_reasons = _penalties(candidate, ev, comp, tenure, beh, title_cur)
    beh_mult = _behavioral_multiplier(beh)

    score = base * pen_mult * beh_mult
    if sc.honeypot:
        score *= 0.01  # consistency-gate: impossible profiles fall to the floor

    sc.score = score
    sc.components = components
    sc.penalties = pen_reasons
    sc.behavioral_mult = beh_mult
    sc.facts = {
        "evidence_hits": ev["hits"],
        "skills": skills,
        "tenure": tenure,
        "behavioral": beh,
        "company": comp,
        "title_cur": title_cur,
    }
    return sc
