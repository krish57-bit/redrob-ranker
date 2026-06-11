"""Semantic similarity layer.

Default backend: TF-IDF + truncated SVD (LSA) dense vectors, fit on the
candidate corpus at ranking time. Fully offline, deterministic (fixed seed),
and fast: ~40s for 100K candidates on a laptop CPU — comfortably inside the
5-minute, no-network ranking budget.

Optional backend ("minilm"): sentence-transformers all-MiniLM-L6-v2 with
embeddings precomputed offline by scripts/precompute_embeddings.py and
loaded from cache during ranking. Use --semantic minilm only if the cache
exists; the ranking step itself never downloads anything.
"""

import numpy as np


def lsa_similarities(jd_text: str, docs: list[str], dims: int = 256,
                     seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        stop_words="english",
    )
    X = vec.fit_transform(docs + [jd_text])
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    Z = svd.fit_transform(X)
    Z = normalize(Z)
    jd_vec = Z[-1]
    sims = Z[:-1] @ jd_vec
    # map cosine from [-1, 1] to [0, 1]
    return (sims + 1.0) / 2.0


def minilm_similarities(jd_text: str, cache_path: str) -> np.ndarray:
    """Load precomputed candidate embeddings; encode only the JD (CPU, local)."""
    from sentence_transformers import SentenceTransformer

    data = np.load(cache_path)
    emb = data["embeddings"]  # (N, 384) float16, L2-normalised
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    jd_vec = model.encode([jd_text], normalize_embeddings=True)[0]
    sims = emb.astype(np.float32) @ jd_vec
    return (sims + 1.0) / 2.0
