#!/usr/bin/env python3
"""Redrob Intelligent Candidate Discovery — ranking entrypoint.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs fully offline on CPU. End-to-end runtime on the 100K pool is well
inside the 5-minute budget (~2.5 min on a 2-core laptop CPU, <3GB RAM).
"""

import argparse
import csv
import gzip
import json
import sys
import time

sys.path.insert(0, "src")

from ranker import jd_profile  # noqa: E402
from ranker.features import candidate_document  # noqa: E402
from ranker.reasoning import build_reasoning  # noqa: E402
from ranker.scoring import score_candidate  # noqa: E402
from ranker.semantic import lsa_similarities, minilm_similarities  # noqa: E402

TOP_N = 100


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--semantic", choices=["lsa", "minilm"], default="lsa")
    ap.add_argument("--minilm-cache", default="artifacts/minilm_embeddings.npz")
    args = ap.parse_args()

    t0 = time.time()
    candidates = load_candidates(args.candidates)
    print(f"[{time.time() - t0:6.1f}s] loaded {len(candidates)} candidates")

    docs = [candidate_document(c) for c in candidates]
    if args.semantic == "minilm":
        sims = minilm_similarities(jd_profile.JD_TEXT, args.minilm_cache)
    else:
        sims = lsa_similarities(jd_profile.JD_TEXT.lower(), docs)
    print(f"[{time.time() - t0:6.1f}s] semantic similarities ({args.semantic})")

    scored = [
        score_candidate(c, float(sims[i])) for i, c in enumerate(candidates)
    ]
    print(f"[{time.time() - t0:6.1f}s] scored all candidates")

    by_id = {c["candidate_id"]: c for c in candidates}
    # Normalise by the pool maximum so scores land in (0, 1] without
    # saturating ties at the top (order is preserved).
    max_score = max(s.score for s in scored) or 1.0
    for s in scored:
        s.score /= max_score
    # Deterministic order: score desc, candidate_id asc — matches the
    # tie-break rule required by the submission spec. Sort on the ROUNDED
    # score so the written file is internally consistent.
    scored.sort(key=lambda s: (-round(s.score, 4), s.candidate_id))
    top = scored[:TOP_N]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, sc in enumerate(top, start=1):
            reasoning = build_reasoning(by_id[sc.candidate_id], sc, rank)
            w.writerow([sc.candidate_id, rank, f"{round(sc.score, 4):.4f}", reasoning])

    n_honeypots = sum(1 for sc in top if sc.honeypot)
    print(f"[{time.time() - t0:6.1f}s] wrote {args.out}")
    print(f"           honeypot-flagged candidates in top {TOP_N}: {n_honeypots}")
    print(f"           score range: {top[0].score:.4f} .. {top[-1].score:.4f}")


if __name__ == "__main__":
    main()
