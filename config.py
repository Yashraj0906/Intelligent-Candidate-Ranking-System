"""
config.py — All weights, thresholds, and constants for the InternDedo ranking system.
Central configuration so every tunable parameter lives in one place.
"""

# =============================================================================
# Embedding Model
# =============================================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 33M params, 384-dim, fast on CPU
EMBEDDING_DIM = 384

# =============================================================================
# Pipeline Parameters
# =============================================================================
RETRIEVAL_TOP_K = 2000          # Stage 2: candidates to shortlist via hybrid retrieval
FINAL_TOP_K = 100               # Stage 3: final ranking output
RRF_K = 60                      # Reciprocal Rank Fusion constant

# =============================================================================
# Stage 3: Reranker Weights (must sum to 1.0)
# =============================================================================
W_CAREER_FIT = 0.35             # Career trajectory, title, product vs services
W_SKILLS_FIT = 0.25             # Skill match, authenticity, assessments
W_BEHAVIORAL = 0.20             # Availability, engagement, recruiter signals
W_LOGISTICS = 0.10              # Location, notice period, salary, work mode
W_BACKGROUND = 0.10             # Education, experience sweet spot, certifications

# =============================================================================
# Experience Parameters
# =============================================================================
EXP_IDEAL_MIN = 5.0             # JD says 5-9 years
EXP_IDEAL_MAX = 9.0
EXP_HARD_MIN = 2.0              # Absolute minimum (generous filter)
EXP_HARD_MAX = 20.0             # Absolute maximum
EXP_SWEET_SPOT_CENTER = 7.0     # Gaussian center
EXP_SWEET_SPOT_SIGMA = 3.0      # Gaussian spread

# =============================================================================
# Notice Period Scoring
# =============================================================================
NOTICE_IDEAL_MAX = 30            # JD prefers sub-30 day, can buy out up to 30
NOTICE_OK_MAX = 60               # Acceptable
NOTICE_PENALTY_THRESHOLD = 90    # Starts getting penalized heavily

# =============================================================================
# Location Preferences
# =============================================================================
PREFERRED_CITIES = {"pune", "noida", "delhi", "delhi ncr", "new delhi", "gurgaon",
                    "gurugram", "greater noida", "faridabad", "ghaziabad"}
TIER1_CITIES = {"mumbai", "bangalore", "bengaluru", "hyderabad", "chennai",
                "kolkata", "ahmedabad", "jaipur"}
PREFERRED_COUNTRY = "india"

# =============================================================================
# Consulting / Services Companies (JD explicitly penalizes entire-career here)
# =============================================================================
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "mindtree", "l&t infotech", "lti", "ltimindtree",
    "hexaware", "cyient", "zensar", "persistent systems", "niit technologies",
    "birlasoft", "coforge", "sonata software", "mastek",
}

# =============================================================================
# AI/ML Related Skill Keywords (for matching against JD requirements)
# =============================================================================
# Must-have skill buckets from the JD
MUST_HAVE_SKILLS = {
    "embeddings_retrieval": {
        "sentence-transformers", "sentence transformers", "embeddings", "embedding",
        "bge", "e5", "openai embeddings", "dense retrieval", "semantic search",
        "vector search", "vector retrieval", "retrieval", "information retrieval",
        "ir", "rag", "retrieval augmented", "retrieval-augmented",
    },
    "vector_db_search": {
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
        "opensearch", "vector database", "vector db", "hybrid search",
        "elastic search", "solr", "lucene", "annoy", "chromadb", "chroma",
        "pgvector",
    },
    "python_coding": {
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "numpy", "pandas", "flask", "fastapi", "django",
    },
    "ranking_evaluation": {
        "ndcg", "mrr", "map", "ranking", "learning to rank", "learning-to-rank",
        "l2r", "a/b testing", "ab testing", "evaluation", "recommendation",
        "recommendation system", "search ranking", "re-ranking", "reranking",
        "xgboost", "lightgbm", "catboost",
    },
}

# Nice-to-have skill keywords
NICE_TO_HAVE_SKILLS = {
    "llm_finetuning": {
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
        "llm", "large language model", "gpt", "bert", "transformer",
        "huggingface", "hugging face",
    },
    "mlops_infra": {
        "mlflow", "kubeflow", "airflow", "docker", "kubernetes", "k8s",
        "ci/cd", "mlops", "model serving", "triton", "bentoml", "ray",
        "distributed", "spark", "kafka",
    },
    "open_source": {
        "github", "open source", "open-source", "contributor", "contributions",
    },
}

# Non-technical titles that should NOT have heavy AI skill lists
NON_TECH_TITLES = {
    "marketing manager", "hr manager", "human resources", "accountant",
    "content writer", "graphic designer", "sales executive", "sales manager",
    "customer support", "operations manager", "civil engineer",
    "mechanical engineer", "business analyst", "project manager",
    "administrative", "receptionist", "executive assistant",
}

# Technical AI/ML titles (positive signal)
AI_ML_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "research scientist", "applied scientist",
    "nlp engineer", "deep learning engineer", "senior ai engineer",
    "senior ml engineer", "senior machine learning engineer",
    "ai/ml engineer", "ml/ai engineer", "ai researcher",
    "machine learning", "data engineer", "backend engineer",
    "software engineer", "senior software engineer", "sde",
    "full stack", "platform engineer",
}

# =============================================================================
# Keyword stuffer detection thresholds
# =============================================================================
STUFFER_AI_SKILL_THRESHOLD = 5          # Non-tech title + >5 AI skills = suspicious
STUFFER_SKILL_MENTION_RATE_MIN = 0.10   # At least 10% of AI skills should appear in career descriptions
STUFFER_PENALTY = -0.40                 # Heavy penalty for detected stuffers
AUTHENTICITY_PENALTY = -0.30            # Skills never mentioned in work descriptions

# =============================================================================
# Honeypot Detection Thresholds
# =============================================================================
HONEYPOT_EXP_CAREER_MISMATCH_MONTHS = 36  # Tolerate up to 3yr discrepancy
HONEYPOT_EXPERT_ZERO_DURATION = True       # Flag expert skills with 0 months
HONEYPOT_PERFECT_SKILL_COUNT = 8           # Flag if >= 8 advanced/expert skills + no assessments
HONEYPOT_SAME_PROFICIENCY_MIN = 6          # Flag if >= 6 skills all at same proficiency

# =============================================================================
# Salary Range (JD implies Series A startup, India market)
# =============================================================================
SALARY_RANGE_MIN_LPA = 15.0    # Reasonable minimum for 5-9 yr AI engineer
SALARY_RANGE_MAX_LPA = 60.0    # Reasonable maximum

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = "data"
CANDIDATES_FILE = "candidates.jsonl"
OUTPUT_FILE = "submission.csv"
