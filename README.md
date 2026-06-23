# 🎯 Intelligent Candidate Discovery & Ranking System

> **Team InternDedo** — Redrob Hackathon: India Runs Data & AI Challenge 2026

An AI-powered candidate ranking system that finds the **top 100 best-fit candidates** from a pool of **100,000 profiles** for a Senior AI Engineer role — not by keyword matching, but by genuinely understanding who fits the role.

---

## 🏆 Key Results

| Metric | Value |
|:---|:---|
| **Total Candidates Processed** | 100,000 |
| **Honeypots Detected & Filtered** | 528 |
| **Ranking Runtime** | **2.2 seconds** _(limit: 5 min)_ |
| **Honeypots in Top 100** | **0** _(disqualified if >10)_ |
| **Keyword Stuffers in Top 100** | **0** |
| **AI/ML Titles in Top 100** | **93/100** |
| **Validation** | ✅ **PASSED** |

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        JD["Job Description\nSenior AI Engineer"]
        CAND["candidates.jsonl\n100,000 profiles"]
    end

    subgraph PRECOMPUTE["⚙️ Pre-Computation Phase (Offline, ~20 min)"]
        direction TB
        FE["Feature Extraction\n100K candidates → structured features"]
        HP["Honeypot Detection\n7 heuristics → 528 flagged"]
        PF["Pre-Filter\nFeature scoring → top 5,000"]
        EMB["BGE-small-en-v1.5\nDense embeddings (5000 × 384)"]
        JDE["JD Embedding\nSingle vector (1 × 384)"]
        
        FE --> HP --> PF --> EMB
        JD --> JDE
    end

    subgraph RANK["🚀 Ranking Phase (2.2 seconds, CPU only)"]
        direction TB
        
        subgraph S2["Stage 2: Hybrid Retrieval"]
            DENSE["Dense Retrieval\nBGE cosine similarity"]
            SPARSE["Sparse Retrieval\nFeature-based BM25 proxy"]
            RRF["Reciprocal Rank Fusion\n1/(k+rank_dense) + 1/(k+rank_sparse)"]
            DENSE --> RRF
            SPARSE --> RRF
        end
        
        subgraph S3["Stage 3: Multi-Signal Reranker"]
            CAREER["Career Fit (35%)\nTitle trajectory, product vs consulting,\nproduction signals, semantic similarity"]
            SKILLS["Skills Fit (25%)\nMust-have coverage, authenticity,\nassessments, anti-stuffer"]
            BEHAV["Behavioral (20%)\nAvailability, engagement,\nrecruiter signals"]
            LOGIS["Logistics (10%)\nLocation, notice period,\nsalary, work mode"]
            BACKG["Background (10%)\nEducation, experience\nsweet spot, certs"]
        end
        
        REASON["Reasoning Generator\nFact-based, per-candidate"]
    end

    subgraph OUTPUT["📤 Output"]
        CSV["submission.csv\nTop 100 ranked candidates\nwith reasoning"]
    end

    CAND --> PRECOMPUTE
    PRECOMPUTE --> RANK
    RRF -->|"Top 2,000"| S3
    S3 -->|"Top 100"| REASON
    REASON --> CSV

    style INPUT fill:#1a1a2e,stroke:#e94560,color:#fff
    style PRECOMPUTE fill:#16213e,stroke:#0f3460,color:#fff
    style RANK fill:#0f3460,stroke:#533483,color:#fff
    style OUTPUT fill:#1a1a2e,stroke:#e94560,color:#fff
```

---

## 🔬 How It Works

### The Problem
Recruiters go through hundreds of profiles and miss the right person — not because talent isn't there, but because keyword filters can't see what actually matters. This system ranks candidates **the way a great recruiter would**.

### The Solution: 4-Stage Pipeline

#### Stage 1: Hard Filters
Eliminates obvious non-fits before any expensive computation:
- **Honeypot detection** — 7 independent heuristics catch impossible profiles
- **Experience band filter** — generous 2-20 year range
- **Result**: 100K → ~92K candidates

#### Stage 2: Hybrid Retrieval (Dense + Sparse + RRF)
```mermaid
flowchart LR
    subgraph Dense["Dense Retrieval"]
        BGE["BGE-small-en-v1.5\n384-dim embeddings"]
        COS["Cosine Similarity\nto JD embedding"]
        BGE --> COS
    end
    
    subgraph Sparse["Sparse Retrieval"]
        FEAT["Feature-based scoring"]
        SKILL["Skill bucket coverage"]
        TITLE["Title signal strength"]
        PROD["Production signal density"]
        FEAT --> SKILL
        FEAT --> TITLE
        FEAT --> PROD
    end
    
    COS -->|"rank_dense"| RRF["Reciprocal Rank\nFusion (k=60)"]
    SKILL -->|"rank_sparse"| RRF
    TITLE -->|"rank_sparse"| RRF
    PROD -->|"rank_sparse"| RRF
    RRF -->|"Top 2,000"| OUT["Shortlist"]
    
    style Dense fill:#1e3a5f,stroke:#4a9eff,color:#fff
    style Sparse fill:#3a1e5f,stroke:#9a4aff,color:#fff
```

**Why hybrid?** Dense retrieval catches semantic meaning (a "search infrastructure engineer" matches "vector DB experience"), while sparse catches exact keyword matches. RRF merges both without tuning weights.

#### Stage 3: Multi-Signal Reranker

Five weighted scoring components, each capturing a different dimension of candidate fit:

```mermaid
pie title Scoring Weight Distribution
    "Career Fit" : 35
    "Skills Fit" : 25
    "Behavioral Signals" : 20
    "Logistics" : 10
    "Background" : 10
```

| Component | Weight | Sub-signals |
|:---|:---:|:---|
| **Career Fit** | 35% | Title trajectory (AI/ML roles), product vs consulting ratio, production deployment NLP signals, semantic similarity to JD, tenure stability |
| **Skills Fit** | 25% | Must-have skill bucket coverage (4 buckets), skill authenticity score, assessment scores, nice-to-have bonus, anti-keyword-stuffer penalty |
| **Behavioral** | 20% | Recency of activity, recruiter response rate, open-to-work flag, GitHub activity, interview completion rate, profile completeness, verification status |
| **Logistics** | 10% | Location match (Pune/Noida preferred), notice period (≤30d ideal), salary alignment, work mode, willingness to relocate |
| **Background** | 10% | Education tier + field relevance, experience sweet spot (Gaussian around 7 yrs), relevant certifications, industry fit |

#### Stage 4: Reasoning Generation
Every candidate gets a **specific, fact-based reasoning** — no templates, no hallucination:
- References actual skills, titles, companies from the profile
- Connects to specific JD requirements
- Honestly notes concerns (long notice periods, location mismatches, etc.)

---

## 🛡️ Anti-Gaming Defenses

### Honeypot Detection (7 Heuristics)
The dataset contains ~80 honeypots with subtly impossible profiles. Our detector catches them:

| # | Heuristic | What it catches |
|:---:|:---|:---|
| H1 | Experience-career mismatch | Stated 8 yrs but career history sums to 3 yrs |
| H2 | Expert-zero-duration | "Expert" in skills with 0 months usage |
| H3 | Perfect skill breadth | 8+ advanced/expert skills with zero assessments |
| H4 | Uniform proficiency | All skills at exact same level (synthetic pattern) |
| H5 | Title-description mismatch | Title says "Mechanical Engineer" but description is about SEO |
| H6 | Impossible tenure | 20+ year tenure or >6 month overlap at different companies |
| H7 | Endorsement anomaly | Expert skills with 50+ endorsements but only 2 months duration |

**Result**: 528 honeypots detected, **0 in our top 100**.

### Anti-Keyword-Stuffer Logic
The JD explicitly warns that candidates with perfect AI keyword lists but non-technical titles (Marketing Manager, HR Manager, etc.) are traps:

```python
# If title is non-technical BUT has 5+ AI skills AND skills are barely
# mentioned in career descriptions → keyword stuffer
if is_non_tech_title and ai_skill_count >= 5 and skill_mention_rate < 0.15:
    score *= 0.15  # Reduce to 15% of original score
```

**Result**: 0 keyword stuffers in our top 100.

### Consulting Company Detection
The JD explicitly penalizes entire-career-at-consulting-firms:
- Detects TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, etc.
- Entire career at such firms → strong penalty
- Mixed (consulting + product) → no penalty

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Pre-compute (run once, ~20 min on CPU)
```bash
python precompute.py --candidates ./candidates.jsonl
```

This extracts features, detects honeypots, pre-filters to top 5000 candidates, and computes BGE-small embeddings.

### Step 2: Rank (produces submission.csv in ~2 seconds)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Step 3: Validate
```bash
python validate_submission.py submission.csv
# Output: Submission is valid.
```

### Single-Command Reproduction
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
> Assumes `precompute.py` has been run once and artifacts exist in `./data/`.

---

## 📁 Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies (pinned)
├── submission_metadata.yaml     # Team metadata for portal
│
├── config.py                    # All tunable parameters & constants
├── utils.py                     # Shared utility functions
├── features.py                  # Feature extraction engine
├── honeypot.py                  # 7-heuristic honeypot detector
├── scoring.py                   # 5-component scoring engine
├── reasoning.py                 # Fact-based reasoning generator
│
├── precompute.py                # Offline: features + embeddings
├── rank.py                      # Online: ranking pipeline (≤5 min)
├── app.py                       # HuggingFace Spaces Gradio demo
│
├── validate_submission.py       # Official format validator
├── submission.csv               # Final ranked output (100 candidates)
│
└── data/                        # Pre-computed artifacts
    ├── features.pkl             # Extracted features (all 100K)
    ├── candidate_ids.pkl        # Ordered candidate IDs
    ├── honeypot_flags.pkl       # Honeypot detection results
    ├── shortlist_ids.pkl        # Pre-filtered 5000 IDs
    ├── shortlist_embeddings.npy # BGE embeddings (5000 × 384)
    └── jd_embedding.npy         # JD embedding (1 × 384)
```

---

## ⚙️ Technical Stack

| Component | Technology | Why |
|:---|:---|:---|
| **Embeddings** | BAAI/bge-small-en-v1.5 (33M params) | Top-tier on MTEB for its size, 384-dim, CPU-friendly, 3x faster than bge-base |
| **Hybrid Retrieval** | Reciprocal Rank Fusion (RRF) | Parameter-free rank merging — used in production by Elasticsearch and Vespa |
| **Feature Engineering** | NumPy + Pandas | Vectorized ops for 100K candidates |
| **NLP Signals** | Regex + keyword matching | Production/ranking signal detection in career narratives |
| **Vector Similarity** | NumPy dot product | L2-normalized embeddings → dot product = cosine similarity |
| **Serialization** | Pickle + NumPy .npy | Fast load of pre-computed artifacts |
| **Sandbox** | Gradio on HuggingFace Spaces | Free tier, Python-native, instant deployment |

---

## 📊 Understanding the JD (What We're Ranking For)

```mermaid
mindmap
  root((Senior AI Engineer<br/>Redrob AI))
    Must Have
      Embeddings-based retrieval
        sentence-transformers
        BGE, E5
      Vector DBs / Hybrid Search
        Pinecone, Weaviate
        FAISS, Elasticsearch
      Strong Python
      Ranking evaluation
        NDCG, MRR, MAP
        A/B testing
    Nice to Have
      LLM fine-tuning
        LoRA, QLoRA, PEFT
      Learning-to-rank
        XGBoost, neural L2R
      HR-tech experience
      Open-source contributions
    Disqualifiers
      Pure research, no production
      Only recent LangChain projects
      Stopped writing code
      Entire career at consulting
      CV/Speech/Robotics only
    Profile
      5-9 years experience
      Product company background
      India - Pune/Noida preferred
      Hybrid work mode
      Ships fast, thinks deep
```

---

## 🧪 Compute Environment

| Spec | Value |
|:---|:---|
| **Platform** | Windows 11, Python 3.11 |
| **CPU** | Standard multi-core |
| **RAM** | 16 GB |
| **GPU** | Not used (CPU only) |
| **Network during ranking** | Off (no API calls) |
| **Ranking runtime** | 2.2 seconds |
| **Pre-computation runtime** | ~20 minutes |

---

## 🔄 End-to-End Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant PC as precompute.py
    participant R as rank.py
    participant V as validate_submission.py
    
    U->>PC: python precompute.py --candidates candidates.jsonl
    activate PC
    PC->>PC: Load 100K candidates
    PC->>PC: Extract features (all 100K)
    PC->>PC: Detect honeypots (528 found)
    PC->>PC: Pre-filter → 5,000 shortlist
    PC->>PC: BGE-small embeddings (5000 × 384)
    PC->>PC: JD embedding (1 × 384)
    PC-->>U: Artifacts saved to data/
    deactivate PC
    
    U->>R: python rank.py --candidates candidates.jsonl --out submission.csv
    activate R
    R->>R: Load pre-computed artifacts (2.1s)
    R->>R: Hybrid retrieval: Dense + Sparse + RRF → top 2,000
    R->>R: Multi-signal reranking → top 100
    R->>R: Generate fact-based reasoning
    R-->>U: submission.csv (100 ranked candidates)
    deactivate R
    
    U->>V: python validate_submission.py submission.csv
    activate V
    V-->>U: ✅ Submission is valid.
    deactivate V
```

---

## 📝 AI Tools Declaration

| Tool | Usage |
|:---|:---|
| Claude (via Antigravity IDE) | Architecture discussion, code generation, iterative refinement |

All design decisions, scoring weights, honeypot heuristics, and anti-stuffer logic were collaboratively developed with AI assistance. **No candidate data was fed to any hosted LLM during ranking.** The ranking step runs 100% offline on CPU with no network.

---

## 📄 License

Built for the **Redrob Hackathon — India Runs Data & AI Challenge 2026**.

---

<p align="center">
  <b>Team InternDedo</b> · Built with 🧠 and ☕
</p>
