"""
utils.py — Shared utility functions for the InternDedo ranking system.
"""

import json
import os
import re
import pickle
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional


def load_candidates_jsonl(filepath: str, max_candidates: int = None) -> List[Dict]:
    """Load candidates from a JSONL file (one JSON object per line)."""
    candidates = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
            if max_candidates and len(candidates) >= max_candidates:
                break
    return candidates


def load_candidates_json(filepath: str) -> List[Dict]:
    """Load candidates from a JSON array file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_numpy(arr: np.ndarray, filepath: str):
    """Save numpy array to .npy file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    np.save(filepath, arr)


def load_numpy(filepath: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(filepath)


def days_since(date_str: str, reference_date: str = "2026-06-01") -> int:
    """Calculate days between a date string and a reference date.
    
    The reference date is set to approximately when this hackathon is happening,
    so 'recency' scores are computed relative to a fixed point.
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
        delta = (ref - d).days
        return max(0, delta)
    except (ValueError, TypeError):
        return 365  # Default to 1 year old if parsing fails


def normalize_text(text: str) -> str:
    """Lowercase, strip, and normalize whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def build_candidate_text(candidate: Dict) -> str:
    """Build a comprehensive text representation of a candidate for embedding.
    
    Concatenates headline, summary, career descriptions, skills, and education
    into a single string that captures the full professional profile.
    """
    parts = []
    
    profile = candidate.get("profile", {})
    
    # Headline and summary
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    
    # Current title and company
    if profile.get("current_title"):
        parts.append(f"Current role: {profile['current_title']} at {profile.get('current_company', 'unknown')}")
    
    # Career history descriptions
    for role in candidate.get("career_history", []):
        role_text = f"{role.get('title', '')} at {role.get('company', '')} ({role.get('industry', '')})"
        if role.get("description"):
            role_text += f": {role['description']}"
        parts.append(role_text)
    
    # Skills with proficiency
    skills = candidate.get("skills", [])
    if skills:
        skill_strs = [f"{s['name']} ({s.get('proficiency', 'unknown')})" for s in skills]
        parts.append("Skills: " + ", ".join(skill_strs))
    
    # Education
    for edu in candidate.get("education", []):
        edu_text = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
        parts.append(edu_text)
    
    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} by {cert.get('issuer', '')}")
    
    return " . ".join(parts)


def build_jd_text() -> str:
    """Build the job description text for embedding comparison.
    
    This is a curated summary of what the JD actually wants, emphasizing
    the real requirements over the filler text.
    """
    return """
    Senior AI Engineer at a Series A AI-native talent intelligence platform.
    Building ranking, retrieval, and matching systems for recruiter search.
    
    Must have: Production experience with embeddings-based retrieval systems 
    like sentence-transformers, BGE, E5 deployed to real users. Production 
    experience with vector databases or hybrid search infrastructure like 
    Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch. Strong Python.
    Evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing.
    
    Building and shipping ranking systems, search systems, recommendation 
    systems to production at scale. Hybrid retrieval combining dense embeddings 
    and sparse keyword matching. LLM-based re-ranking. Fine-tuning language 
    models. Applied ML at product companies, not consulting.
    
    5-9 years experience, preferably at product companies. Must write code.
    Located in India, Pune or Noida preferred. Hybrid work mode.
    
    Nice to have: LLM fine-tuning with LoRA QLoRA PEFT, learning-to-rank 
    models with XGBoost, HR-tech or marketplace products, distributed systems,
    open-source contributions in AI/ML.
    
    Disqualifiers: Pure research no production, only recent LangChain projects,
    stopped writing code, entire career at consulting firms like TCS Infosys Wipro,
    primarily computer vision speech or robotics without NLP/IR.
    """


def gaussian_score(value: float, center: float, sigma: float) -> float:
    """Compute a Gaussian-shaped score centered at 'center' with spread 'sigma'.
    Returns 1.0 at center, decaying smoothly to 0 as distance increases.
    """
    return np.exp(-0.5 * ((value - center) / sigma) ** 2)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))
