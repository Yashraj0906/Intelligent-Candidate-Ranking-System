"""
features.py — Feature extraction from candidate profiles.

Extracts structured features from raw candidate JSON into a flat dict
that can be used by the scoring module. This is called during precomputation.
"""

import re
from typing import Dict, List, Set
from config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, NON_TECH_TITLES, AI_ML_TITLES,
    CONSULTING_COMPANIES, PREFERRED_CITIES, TIER1_CITIES, PREFERRED_COUNTRY,
    STUFFER_AI_SKILL_THRESHOLD,
)
from utils import normalize_text, days_since


def extract_features(candidate: Dict) -> Dict:
    """Extract all structured features from a candidate profile.
    
    Returns a flat dictionary of features used by the scoring module.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    languages = candidate.get("languages", [])
    signals = candidate.get("redrob_signals", {})
    
    features = {}
    
    # =========================================================================
    # Basic profile features
    # =========================================================================
    features["candidate_id"] = candidate.get("candidate_id", "")
    features["years_of_experience"] = profile.get("years_of_experience", 0)
    features["current_title"] = profile.get("current_title", "")
    features["current_title_lower"] = normalize_text(profile.get("current_title", ""))
    features["current_company"] = profile.get("current_company", "")
    features["current_industry"] = profile.get("current_industry", "")
    features["location"] = normalize_text(profile.get("location", ""))
    features["country"] = normalize_text(profile.get("country", ""))
    features["headline"] = profile.get("headline", "")
    features["summary"] = profile.get("summary", "")
    
    # =========================================================================
    # Career features
    # =========================================================================
    features["num_roles"] = len(career)
    features["career_descriptions"] = " ".join(
        r.get("description", "") for r in career
    ).lower()
    
    # Title trajectory: what titles has this person held?
    all_titles = [normalize_text(r.get("title", "")) for r in career]
    features["all_titles"] = all_titles
    
    # Is current title a technical AI/ML role?
    features["is_ai_ml_title"] = _is_ai_ml_title(features["current_title_lower"])
    
    # Has any AI/ML title in career history?
    features["has_ai_ml_title_in_history"] = any(
        _is_ai_ml_title(t) for t in all_titles
    )
    
    # Is current title non-technical?
    features["is_non_tech_title"] = features["current_title_lower"] in NON_TECH_TITLES
    
    # Product vs consulting company analysis
    all_companies = [normalize_text(r.get("company", "")) for r in career]
    consulting_count = sum(1 for c in all_companies if _is_consulting_company(c))
    features["consulting_company_count"] = consulting_count
    features["total_company_count"] = len(all_companies)
    features["all_consulting"] = (consulting_count == len(all_companies) and len(all_companies) > 0)
    features["consulting_ratio"] = consulting_count / max(len(all_companies), 1)
    
    # Average tenure (for title-chaser detection)
    tenures = [r.get("duration_months", 0) for r in career]
    features["avg_tenure_months"] = sum(tenures) / max(len(tenures), 1)
    features["min_tenure_months"] = min(tenures) if tenures else 0
    features["max_tenure_months"] = max(tenures) if tenures else 0
    
    # Production deployment signals in career descriptions
    features["production_signal_count"] = _count_production_signals(
        features["career_descriptions"]
    )
    
    # NLP/IR/Search/Ranking signals in career descriptions
    features["ranking_search_signal_count"] = _count_ranking_search_signals(
        features["career_descriptions"]
    )
    
    # =========================================================================
    # Skills features
    # =========================================================================
    skill_names_lower = {normalize_text(s.get("name", "")) for s in skills}
    features["skill_names_lower"] = skill_names_lower
    features["total_skills"] = len(skills)
    
    # Match against JD must-have skill buckets
    for bucket_name, bucket_keywords in MUST_HAVE_SKILLS.items():
        matched = _match_skills(skill_names_lower, bucket_keywords)
        features[f"must_have_{bucket_name}_count"] = len(matched)
        features[f"must_have_{bucket_name}_matched"] = matched
    
    # Match against JD nice-to-have skill buckets
    for bucket_name, bucket_keywords in NICE_TO_HAVE_SKILLS.items():
        matched = _match_skills(skill_names_lower, bucket_keywords)
        features[f"nice_to_have_{bucket_name}_count"] = len(matched)
    
    # Total AI-relevant skill count (for stuffer detection)
    all_ai_keywords = set()
    for bucket in MUST_HAVE_SKILLS.values():
        all_ai_keywords.update(bucket)
    for bucket in NICE_TO_HAVE_SKILLS.values():
        all_ai_keywords.update(bucket)
    features["ai_skill_count"] = len(_match_skills(skill_names_lower, all_ai_keywords))
    
    # Skill authenticity: cross-reference skills with career descriptions
    features["skill_mention_rate"] = _skill_mention_rate(skills, features["career_descriptions"])
    
    # Skill proficiency distribution
    proficiencies = [s.get("proficiency", "").lower() for s in skills]
    features["advanced_expert_skill_count"] = sum(
        1 for p in proficiencies if p in ("advanced", "expert")
    )
    
    # Average skill endorsements and duration for AI skills
    ai_skills = [s for s in skills 
                 if normalize_text(s.get("name", "")) in all_ai_keywords
                 or any(kw in normalize_text(s.get("name", "")) for kw in all_ai_keywords)]
    if ai_skills:
        features["avg_ai_skill_endorsements"] = sum(
            s.get("endorsements", 0) for s in ai_skills
        ) / len(ai_skills)
        features["avg_ai_skill_duration"] = sum(
            s.get("duration_months", 0) for s in ai_skills
        ) / len(ai_skills)
    else:
        features["avg_ai_skill_endorsements"] = 0
        features["avg_ai_skill_duration"] = 0
    
    # Keyword stuffer detection flag
    features["is_likely_stuffer"] = (
        features["is_non_tech_title"] 
        and features["ai_skill_count"] >= STUFFER_AI_SKILL_THRESHOLD
        and features["skill_mention_rate"] < 0.15
    )
    
    # =========================================================================
    # Assessment scores
    # =========================================================================
    assessment_scores = signals.get("skill_assessment_scores", {})
    features["has_assessments"] = len(assessment_scores) > 0
    features["num_assessments"] = len(assessment_scores)
    features["avg_assessment_score"] = (
        sum(assessment_scores.values()) / len(assessment_scores)
        if assessment_scores else 0
    )
    
    # Check if assessments are for relevant skills
    relevant_assessment_count = 0
    relevant_assessment_total = 0
    for skill_name, score in assessment_scores.items():
        skill_lower = normalize_text(skill_name)
        if any(kw in skill_lower or skill_lower in kw 
               for bucket in MUST_HAVE_SKILLS.values() for kw in bucket):
            relevant_assessment_count += 1
            relevant_assessment_total += score
        elif any(kw in skill_lower or skill_lower in kw 
                 for bucket in NICE_TO_HAVE_SKILLS.values() for kw in bucket):
            relevant_assessment_count += 1
            relevant_assessment_total += score
    features["relevant_assessment_count"] = relevant_assessment_count
    features["relevant_assessment_avg"] = (
        relevant_assessment_total / relevant_assessment_count
        if relevant_assessment_count > 0 else 0
    )
    
    # =========================================================================
    # Education features
    # =========================================================================
    tiers = [e.get("tier", "unknown") for e in education]
    fields = [normalize_text(e.get("field_of_study", "")) for e in education]
    degrees = [normalize_text(e.get("degree", "")) for e in education]
    
    features["best_edu_tier"] = _best_tier(tiers)
    features["has_cs_ai_education"] = any(
        f in ("computer science", "artificial intelligence", "machine learning",
              "data science", "information technology", "software engineering",
              "electronics", "electrical engineering", "mathematics", "statistics")
        for f in fields
    )
    features["has_masters_phd"] = any(
        d in ("m.tech", "m.sc", "m.e.", "ms", "msc", "mtech", "ph.d", "phd", "m.s.")
        for d in degrees
    )
    
    # =========================================================================
    # Behavioral signal features
    # =========================================================================
    features["profile_completeness"] = signals.get("profile_completeness_score", 0)
    features["last_active_days_ago"] = days_since(signals.get("last_active_date", "2020-01-01"))
    features["open_to_work"] = signals.get("open_to_work_flag", False)
    features["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0)
    features["avg_response_time_hours"] = signals.get("avg_response_time_hours", 999)
    features["github_activity_score"] = signals.get("github_activity_score", -1)
    features["interview_completion_rate"] = signals.get("interview_completion_rate", 0)
    features["offer_acceptance_rate"] = signals.get("offer_acceptance_rate", -1)
    features["profile_views_30d"] = signals.get("profile_views_received_30d", 0)
    features["applications_30d"] = signals.get("applications_submitted_30d", 0)
    features["search_appearance_30d"] = signals.get("search_appearance_30d", 0)
    features["saved_by_recruiters_30d"] = signals.get("saved_by_recruiters_30d", 0)
    features["connection_count"] = signals.get("connection_count", 0)
    features["endorsements_received"] = signals.get("endorsements_received", 0)
    
    # Logistics
    features["notice_period_days"] = signals.get("notice_period_days", 90)
    salary = signals.get("expected_salary_range_inr_lpa", {})
    features["salary_min_lpa"] = salary.get("min", 0)
    features["salary_max_lpa"] = salary.get("max", 0)
    features["preferred_work_mode"] = signals.get("preferred_work_mode", "")
    features["willing_to_relocate"] = signals.get("willing_to_relocate", False)
    
    # Verification
    features["verified_email"] = signals.get("verified_email", False)
    features["verified_phone"] = signals.get("verified_phone", False)
    features["linkedin_connected"] = signals.get("linkedin_connected", False)
    
    # Location matching
    features["is_preferred_city"] = any(
        city in features["location"] for city in PREFERRED_CITIES
    )
    features["is_tier1_city"] = any(
        city in features["location"] for city in TIER1_CITIES
    )
    features["is_india"] = features["country"] == "india"
    
    # Certifications relevance
    cert_names = [normalize_text(c.get("name", "")) for c in certs]
    features["has_relevant_certs"] = any(
        any(kw in cn for kw in ("aws", "gcp", "azure", "ml", "ai", "data", "python",
                                 "tensorflow", "pytorch", "deep learning"))
        for cn in cert_names
    )
    features["num_certs"] = len(certs)
    
    return features


# =========================================================================
# Helper functions
# =========================================================================

def _is_ai_ml_title(title: str) -> bool:
    """Check if a title is an AI/ML/Data/Engineering technical role."""
    title = title.lower().strip()
    
    # Direct match
    if title in AI_ML_TITLES:
        return True
    
    # Partial match — check if any AI/ML keyword is in the title
    ai_keywords = {"ai", "ml", "machine learning", "data scien", "deep learning",
                   "nlp", "research scientist", "applied scientist", "data engineer",
                   "software engineer", "backend engineer", "platform engineer",
                   "sde", "full stack", "devops"}
    return any(kw in title for kw in ai_keywords)


def _is_consulting_company(company: str) -> bool:
    """Check if a company is a known consulting/services firm."""
    company = company.lower().strip()
    return any(c in company or company in c for c in CONSULTING_COMPANIES)


def _match_skills(candidate_skills: Set[str], keywords: Set[str]) -> Set[str]:
    """Find matches between a candidate's skills and a set of keywords.
    
    Uses substring matching in both directions to catch variations
    like 'sentence-transformers' matching 'sentence transformers'.
    """
    matched = set()
    for skill in candidate_skills:
        for kw in keywords:
            if kw in skill or skill in kw:
                matched.add(skill)
                break
    return matched


def _skill_mention_rate(skills: List[Dict], career_text: str) -> float:
    """What fraction of the candidate's skills are actually mentioned
    in their career descriptions?
    
    This catches keyword stuffers who list skills they never used.
    """
    if not skills:
        return 0.0
    
    mentioned = 0
    for s in skills:
        name = s.get("name", "").lower()
        if name and (name in career_text or 
                     name.replace("-", " ") in career_text or
                     name.replace(" ", "-") in career_text):
            mentioned += 1
    
    return mentioned / len(skills)


def _count_production_signals(text: str) -> int:
    """Count production-deployment signal keywords in career descriptions.
    
    These indicate the candidate has shipped real systems, not just done
    research or tutorials.
    """
    production_keywords = [
        "production", "deployed", "shipped", "live", "real users",
        "real-time", "realtime", "real time", "at scale", "scale",
        "api", "microservice", "pipeline", "infrastructure",
        "monitoring", "observability", "sla", "uptime", "latency",
        "throughput", "load balancing", "ci/cd", "cicd",
        "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
        "millions", "thousands of users", "data pipeline",
    ]
    return sum(1 for kw in production_keywords if kw in text)


def _count_ranking_search_signals(text: str) -> int:
    """Count ranking/search/recommendation signal keywords."""
    keywords = [
        "ranking", "search", "retrieval", "recommendation",
        "embedding", "vector", "similarity", "relevance",
        "ndcg", "mrr", "precision", "recall", "f1",
        "a/b test", "ab test", "evaluation", "benchmark",
        "information retrieval", "nlp", "natural language",
        "transformer", "bert", "gpt", "llm", "language model",
        "fine-tun", "finetun", "re-rank", "rerank",
        "feature engineering", "feature store",
        "xgboost", "lightgbm", "learning to rank",
    ]
    return sum(1 for kw in keywords if kw in text)


def _best_tier(tiers: List[str]) -> int:
    """Get best education tier as integer (1=best, 4=worst, 5=unknown)."""
    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
    if not tiers:
        return 5
    return min(tier_map.get(t, 5) for t in tiers)
