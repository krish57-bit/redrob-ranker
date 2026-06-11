"""Minimal sanity tests: honeypot gates, keyword-stuffer gate, determinism.

Run: python -m pytest tests/ -q
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ranker.honeypots import honeypot_flags
from ranker.scoring import score_candidate

BASE_SIGNALS = {
    "profile_completeness_score": 90.0,
    "signup_date": "2024-01-01",
    "last_active_date": "2026-06-01",
    "open_to_work_flag": True,
    "profile_views_received_30d": 10,
    "applications_submitted_30d": 2,
    "recruiter_response_rate": 0.9,
    "avg_response_time_hours": 5.0,
    "skill_assessment_scores": {},
    "connection_count": 200,
    "endorsements_received": 50,
    "notice_period_days": 30,
    "expected_salary_range_inr_lpa": {"min": 30, "max": 40},
    "preferred_work_mode": "hybrid",
    "willing_to_relocate": True,
    "github_activity_score": 50.0,
    "search_appearance_30d": 20,
    "saved_by_recruiters_30d": 3,
    "interview_completion_rate": 0.9,
    "offer_acceptance_rate": 0.8,
    "verified_email": True,
    "verified_phone": True,
    "linkedin_connected": True,
}


def make_candidate(**overrides):
    c = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "ML Engineer",
            "summary": "Built embedding-based retrieval and ranking systems "
                       "shipped to production with A/B tested NDCG gains.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Machine Learning Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "SaaS",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": "Machine Learning Engineer",
                "start_date": "2021-06-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "SaaS",
                "company_size": "201-500",
                "description": "Shipped semantic search with FAISS embeddings "
                               "and learning-to-rank re-ranking in production.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Embeddings", "proficiency": "advanced",
             "endorsements": 30, "duration_months": 48},
            {"name": "Python", "proficiency": "expert",
             "endorsements": 40, "duration_months": 80},
        ],
        "redrob_signals": dict(BASE_SIGNALS),
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in c and isinstance(c[key], dict):
            c[key].update(val)
        else:
            c[key] = val
    return c


def test_clean_candidate_has_no_honeypot_flags():
    assert honeypot_flags(make_candidate()) == []


def test_impossible_skill_durations_flagged():
    c = make_candidate()
    c["skills"] = [
        {"name": "Python", "proficiency": "expert",
         "endorsements": 5, "duration_months": 200},
        {"name": "NLP", "proficiency": "expert",
         "endorsements": 5, "duration_months": 220},
        {"name": "RAG", "proficiency": "expert",
         "endorsements": 5, "duration_months": 210},
    ]
    assert "skill_impossible" in honeypot_flags(c)
    assert score_candidate(c, 0.5).score < 0.05


def test_keyword_stuffer_gated_by_title():
    stuffer = make_candidate()
    stuffer["profile"]["current_title"] = "HR Manager"
    stuffer["career_history"][0]["title"] = "HR Manager"
    stuffer["career_history"][0]["description"] = "HR operations and hiring."
    stuffer["profile"]["summary"] = "HR professional."
    real = make_candidate()
    s_stuffer = score_candidate(stuffer, 0.5).score
    s_real = score_candidate(real, 0.5).score
    assert s_stuffer < 0.1 * s_real


def test_inactive_candidate_downweighted():
    stale = make_candidate(
        redrob_signals=dict(BASE_SIGNALS,
                            last_active_date="2025-10-01",
                            recruiter_response_rate=0.05),
    )
    fresh = make_candidate()
    assert score_candidate(stale, 0.5).score < 0.6 * score_candidate(fresh, 0.5).score


def test_deterministic():
    c = make_candidate()
    assert score_candidate(c, 0.5).score == score_candidate(c, 0.5).score
