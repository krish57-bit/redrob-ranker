# Redrob Intelligent Candidate Discovery — Ranking System

Ranks the 100,000-candidate pool against the **Senior AI Engineer — Founding
Team** JD the way a careful recruiter would: by reading career histories for
real evidence, trusting skills only when corroborated, weighing behavioral
availability, and rejecting internally inconsistent (honeypot) profiles.

## Reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs fully **offline, CPU-only**, ~2.5 minutes end-to-end on the 100K pool
(2-core laptop; faster with more cores), < 3 GB RAM peak. Accepts `.jsonl`
or `.jsonl.gz`. Validate with:

```bash
python validate_submission.py submission.csv
```

## How it works

```
candidates.jsonl
   │
   ├─► honeypots.py    4 internal-consistency checks (impossible skill
   │                   durations, expert-with-zero-use, date mismatches,
   │                   career overflow) → flagged profiles floored to ~0
   │
   ├─► features.py     evidence mining over career-history DESCRIPTIONS
   │                   (retrieval / ranking / NLP-LLM / production families,
   │                   recency-decayed), title relevance, trust-weighted
   │                   skills, company profile, tenure pattern, behavioral
   │                   signals, location & experience fit
   │
   ├─► semantic.py     TF-IDF + LSA (256-dim, seeded) cosine similarity
   │                   between the JD text and each candidate's narrative
   │                   (optional: precomputed MiniLM embeddings backend)
   │
   ├─► scoring.py      final = Σ(weighted components) × penalties × behavior
   │                   weights: evidence .32 · title .18 · skills .16 ·
   │                   semantic .12 · yoe .10 · location .08 · logistics .04
   │
   └─► reasoning.py    per-candidate, fact-grounded 1-2 sentence reasoning
                       (cites real titles, years, evidence terms, signal
                       values; names the top concern; varies by candidate)
```

### Design decisions (the JD's traps, addressed)

| Trap | Defense |
|---|---|
| Keyword stuffers (perfect skill list, wrong title) | Current title gates the score (×0.05 for non-technical titles); skills earn weight only when corroborated by duration, endorsements, or assessment scores |
| Honeypots (~80 impossible profiles) | Four independent consistency checks; flags 126/100K, **0 honeypots in our top 100** |
| Perfect-on-paper but unavailable | Behavioral multiplier: months inactive, recruiter response rate, open-to-work, interview completion — floors at 0.30 |
| Consulting-only careers | ×0.15 when every job is at a services firm/industry (prior product experience exempts, per the JD) |
| Research-without-production | ×0.40 when research language dominates and production evidence is absent |
| CV/speech-only specialists | ×0.40 when vision/speech terms dominate without NLP/IR evidence |
| Title-chasers / job-hoppers | ×0.70 for 4+ jobs averaging <20 months |

### Why no LLM in the ranking step

The compute budget (5 min, CPU, no network) rules out per-candidate LLM
calls by design. We treat that as the point of the exercise: a production
recruiting system needs a fast, cheap, *auditable* ranker. Every score here
decomposes into named components with explicit weights, so every ranking
decision can be explained to a recruiter — which is also what generates the
reasoning column (no generation, no hallucination: only facts extracted
from the profile being scored).

### Semantic layer backends

- **`--semantic lsa`** (default): TF-IDF (uni+bigrams) + truncated SVD fit
  on the candidate corpus at ranking time. Deterministic (seeded),
  no model files, no network. This is what produced the submission.
- **`--semantic minilm`** (optional): sentence-transformers
  `all-MiniLM-L6-v2` over embeddings precomputed offline by
  `scripts/precompute_embeddings.py`. The ranking step only loads the cache
  and encodes the JD locally.

## Repo layout

```
rank.py                      entrypoint (single reproduce command)
src/ranker/jd_profile.py     declarative JD interpretation (all knobs live here)
src/ranker/honeypots.py      consistency checks
src/ranker/features.py       feature extraction
src/ranker/semantic.py       LSA / MiniLM similarity
src/ranker/scoring.py        component weights, penalties, behavioral multiplier
src/ranker/reasoning.py      fact-grounded reasoning strings
scripts/precompute_embeddings.py   optional MiniLM cache builder
app.py                       Streamlit sandbox demo (≤100-candidate sample)
tests/test_pipeline.py       sanity tests (pytest)
```

## Sandbox

`streamlit run app.py` — upload a ≤100-candidate JSONL sample, get the
ranked CSV with reasoning, honeypot flags, and penalty breakdown per row.
Deployable on Streamlit Cloud / HuggingFace Spaces free tier.

## Tests

```bash
python -m pytest tests/ -q
```
