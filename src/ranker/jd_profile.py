"""Structured representation of what the Senior AI Engineer JD actually asks for.

The JD is deliberately written in prose with explicit "read between the lines"
guidance. This module encodes that reading as data: keyword evidence families,
title relevance, location preferences, experience band, and disqualifier
patterns. Keeping it declarative makes every scoring decision auditable.
"""

REFERENCE_DATE = "2026-06-11"  # ranking date; used for activity recency

# ---------------------------------------------------------------------------
# Evidence families mined from career-history DESCRIPTIONS (not skill lists).
# The JD: "A candidate may not use the words 'RAG' or 'Pinecone'... but if
# their career history shows they built a recommendation system at a product
# company, they're a fit."
# ---------------------------------------------------------------------------
EVIDENCE_FAMILIES = {
    # Core must-have: embeddings-based retrieval / vector search in production
    "retrieval": {
        "weight": 1.0,
        "terms": [
            "embedding", "vector search", "vector database", "vector db",
            "semantic search", "retrieval", "faiss", "pinecone", "weaviate",
            "qdrant", "milvus", "opensearch", "elasticsearch", "bm25",
            "hybrid search", "ann index", "hnsw", "two-tower", "dense retrieval",
            "sentence transformer", "sentence-transformer",
        ],
    },
    # Core must-have: ranking / recommendation systems + rigorous evaluation
    "ranking": {
        "weight": 1.0,
        "terms": [
            "ranking model", "ranking system", "learning-to-rank", "ltr",
            "re-rank", "rerank", "recommendation", "recsys", "discovery feed",
            "personalization", "personalisation", "search relevance",
            "ndcg", "mrr", "map@", "offline-online", "offline metrics",
            "a/b test", "ab test", "interleaving", "click-through",
        ],
    },
    # Modern NLP / LLM depth (wanted, but not sufficient alone)
    "nlp_llm": {
        "weight": 0.6,
        "terms": [
            "nlp", "natural language", "llm", "large language model",
            "fine-tun", "lora", "qlora", "peft", "rag", "transformer",
            "bert", "gpt", "text classification", "named entity",
            "hugging face", "huggingface", "tokeniz",
        ],
    },
    # Production shipping signals — the JD tilts "shipper over researcher"
    "production": {
        "weight": 0.8,
        "terms": [
            "shipped", "deployed", "in production", "to production",
            "production deployment", "real users", "served", "serving",
            "latency", "p99", "qps", "at scale", "rolled out", "launched",
            "monitoring", "drift", "index refresh", "online metrics",
        ],
    },
}

# Negative-evidence family: careers dominated by vision/speech/robotics
# without NLP/IR exposure are an explicit "do NOT want".
CV_SPEECH_TERMS = [
    "computer vision", "image classification", "object detection", "yolo",
    "image segmentation", "gan", "speech recognition", "asr", "text-to-speech",
    "robotics", "slam", "autonomous", "lidar", "opencv", "video analytics",
]

# Research-only careers (academic labs, no production deployment) are a
# stated hard disqualifier.
RESEARCH_TERMS = [
    "research", "paper", "publication", "phd", "thesis", "academic",
    "laboratory", "lab ", "novel architecture", "state-of-the-art",
]

# ---------------------------------------------------------------------------
# Title relevance. current_title is the decisive signal against the
# keyword-stuffer trap ("Marketing Manager with 9 AI skills is not a fit").
# ---------------------------------------------------------------------------
TITLE_RELEVANCE = {
    # direct fits
    "senior ai engineer": 1.00,
    "lead ai engineer": 0.95,
    "staff machine learning engineer": 0.95,
    "senior machine learning engineer": 1.00,
    "senior nlp engineer": 1.00,
    "senior applied scientist": 0.90,
    "machine learning engineer": 0.92,
    "applied ml engineer": 0.92,
    "ml engineer": 0.90,
    "ai engineer": 0.90,
    "nlp engineer": 0.92,
    "search engineer": 0.95,
    "recommendation systems engineer": 0.95,
    "senior data scientist": 0.82,
    "ai specialist": 0.75,
    "data scientist": 0.75,
    "ai research engineer": 0.70,   # research tilt — JD wants shippers
    "senior software engineer (ml)": 0.88,
    "junior ml engineer": 0.55,     # seniority gap for a 5-9y senior role
    "computer vision engineer": 0.40,  # explicit "not without NLP/IR"
    # adjacent: can hide Tier-5 "plain language" gems — description evidence
    # must do the lifting, title alone earns a moderate base
    "senior software engineer": 0.55,
    "senior data engineer": 0.50,
    "data engineer": 0.45,
    "backend engineer": 0.45,
    "software engineer": 0.45,
    "analytics engineer": 0.40,
    "data analyst": 0.30,
    "full stack developer": 0.30,
    "frontend engineer": 0.20,
    "java developer": 0.25,
    ".net developer": 0.25,
    "mobile developer": 0.20,
    "cloud engineer": 0.30,
    "devops engineer": 0.30,
    "qa engineer": 0.15,
}
# Anything not listed (HR Manager, Marketing Manager, Accountant, ...) → 0.0
NON_TECHNICAL_FLOOR = 0.0

# ---------------------------------------------------------------------------
# Skills the JD cares about, mined from the candidate `skills` array but only
# when corroborated (duration / endorsements / assessment) — see scoring.py.
# ---------------------------------------------------------------------------
RELEVANT_SKILLS = {
    # core retrieval/ranking
    "embeddings": 1.0, "vector search": 1.0, "semantic search": 1.0,
    "information retrieval": 1.0, "faiss": 0.9, "pinecone": 0.9,
    "sentence transformers": 0.9, "recommendation systems": 1.0,
    "elasticsearch": 0.8, "opensearch": 0.8,
    # llm / nlp
    "rag": 0.7, "llms": 0.7, "fine-tuning llms": 0.7, "nlp": 0.8,
    "hugging face transformers": 0.7, "prompt engineering": 0.3,
    "langchain": 0.2,  # explicitly "framework enthusiast" territory
    # engineering foundation
    "python": 0.8, "pytorch": 0.6, "tensorflow": 0.4, "xgboost": 0.6,
    "lightgbm": 0.6, "feature engineering": 0.5, "mlflow": 0.5,
    "weights & biases": 0.4, "mlops": 0.5, "sql": 0.3, "spark": 0.3,
    "kafka": 0.2, "airflow": 0.2, "docker": 0.2, "kubernetes": 0.2,
    "fastapi": 0.3, "aws": 0.2, "gcp": 0.2,
}

# ---------------------------------------------------------------------------
# Companies / industries. "Only consulting firms entire career" is a stated
# disqualifier; product-company ML experience is the ideal profile.
# ---------------------------------------------------------------------------
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "mindtree", "lti", "l&t infotech",
}
SERVICES_INDUSTRIES = {"it services", "consulting"}
PRODUCT_INDUSTRIES = {
    "ai/ml", "saas", "e-commerce", "fintech", "food delivery", "edtech",
    "adtech", "software", "insurance tech", "transportation", "healthtech",
    "gaming", "media", "social", "marketplace",
}

# ---------------------------------------------------------------------------
# Experience band: 5-9 stated, 6-8 ideal, soft outside.
# ---------------------------------------------------------------------------
YOE_IDEAL = (6.0, 8.0)
YOE_ACCEPTABLE = (5.0, 9.0)

# ---------------------------------------------------------------------------
# Location: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR welcome;
# other Tier-1 India okay with relocation; outside India case-by-case.
# ---------------------------------------------------------------------------
LOCATION_SCORES = {
    "pune": 1.00, "noida": 1.00,
    "delhi": 0.90, "hyderabad": 0.90, "mumbai": 0.90, "gurgaon": 0.90,
    "bangalore": 0.78, "chennai": 0.70, "kolkata": 0.70, "ahmedabad": 0.65,
    "jaipur": 0.62, "indore": 0.62, "chandigarh": 0.62, "coimbatore": 0.60,
    "trivandrum": 0.60, "bhubaneswar": 0.60, "vizag": 0.60,
}
INDIA_DEFAULT = 0.55
ABROAD_DEFAULT = 0.18
RELOCATION_BONUS = 0.15  # added (capped at 0.92) when willing_to_relocate

# JD text used by the semantic layer (condensed to the substantive parts).
JD_TEXT = """
Senior AI Engineer founding team. Own the intelligence layer: ranking,
retrieval, and matching systems for candidate and job search. Production
experience with embeddings-based retrieval systems (sentence-transformers,
BGE, E5) deployed to real users; embedding drift, index refresh, retrieval
quality regression in production. Vector databases and hybrid search:
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python and code quality. Evaluation frameworks for ranking systems:
NDCG, MRR, MAP, offline-to-online correlation, A/B testing. Hybrid retrieval,
LLM-based re-ranking, BM25. Nice to have: LLM fine-tuning LoRA QLoRA PEFT,
learning-to-rank XGBoost, HR-tech recruiting marketplace products,
distributed systems, inference optimization, open source contributions.
Shipped end-to-end ranking, search, or recommendation systems to real users
at meaningful scale at product companies. Applied ML at product companies,
not pure research, not consulting services. Scrappy product engineering,
ships fast, writes production code.
"""
