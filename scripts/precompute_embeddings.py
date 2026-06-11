#!/usr/bin/env python3
"""OPTIONAL offline pre-computation of MiniLM embeddings.

Not required for the default (LSA) pipeline. Run this once, with network
access, if you want `rank.py --semantic minilm`:

    pip install sentence-transformers
    python scripts/precompute_embeddings.py \
        --candidates ./candidates.jsonl \
        --out artifacts/minilm_embeddings.npz

The ranking step then loads the cache and encodes only the JD — no network,
CPU-only, within the compute budget. Per submission_spec.md section 10.3,
pre-computation may exceed the 5-minute window as long as it is documented.
"""

import argparse
import gzip
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from ranker.features import candidate_document  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="artifacts/minilm_embeddings.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    opener = gzip.open if args.candidates.endswith(".gz") else open
    with opener(args.candidates, "rt", encoding="utf-8") as f:
        docs = [candidate_document(json.loads(line)) for line in f if line.strip()]

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(
        docs,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float16)

    np.savez_compressed(args.out, embeddings=emb)
    print(f"saved {emb.shape} embeddings to {args.out}")


if __name__ == "__main__":
    main()
