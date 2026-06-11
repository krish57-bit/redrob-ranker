"""Streamlit sandbox demo — upload a small candidate sample, get a ranking.

Deploy on Streamlit Cloud / HuggingFace Spaces (free tier):
    streamlit run app.py

Satisfies submission_spec.md section 10.5: accepts a ≤100-candidate JSONL
upload, runs the full ranking pipeline end-to-end on CPU, and returns the
ranked CSV.
"""

import io
import json
import sys

import streamlit as st

sys.path.insert(0, "src")

from ranker import jd_profile
from ranker.features import candidate_document
from ranker.reasoning import build_reasoning
from ranker.scoring import score_candidate
from ranker.semantic import lsa_similarities

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Intelligent Candidate Discovery — demo sandbox")
st.markdown(
    "Upload a JSONL sample (≤100 candidates, same schema as "
    "`candidates.jsonl`) and the ranker scores them against the "
    "Senior AI Engineer JD. Fully offline, CPU-only."
)

uploaded = st.file_uploader("candidates.jsonl sample", type=["jsonl", "json"])

if uploaded:
    lines = uploaded.read().decode("utf-8").strip().splitlines()
    candidates = [json.loads(line) for line in lines if line.strip()][:100]
    st.write(f"Loaded **{len(candidates)}** candidates.")

    docs = [candidate_document(c) for c in candidates]
    sims = lsa_similarities(jd_profile.JD_TEXT.lower(), docs, dims=min(64, len(docs)))

    scored = [score_candidate(c, float(sims[i])) for i, c in enumerate(candidates)]
    mx = max(s.score for s in scored) or 1.0
    for s in scored:
        s.score /= mx
    by_id = {c["candidate_id"]: c for c in candidates}
    scored.sort(key=lambda s: (-round(s.score, 4), s.candidate_id))

    rows = []
    for rank, sc in enumerate(scored, start=1):
        rows.append(
            {
                "candidate_id": sc.candidate_id,
                "rank": rank,
                "score": round(sc.score, 4),
                "reasoning": build_reasoning(by_id[sc.candidate_id], sc, rank),
                "honeypot_flags": ", ".join(sc.honeypot) or "—",
                "penalties": ", ".join(sc.penalties) or "—",
            }
        )
    st.dataframe(rows, use_container_width=True)

    buf = io.StringIO()
    import csv as _csv

    w = _csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
    st.download_button("Download ranked CSV", buf.getvalue(), "ranked_sample.csv")
