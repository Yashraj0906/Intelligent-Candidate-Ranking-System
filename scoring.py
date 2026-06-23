"""
scoring.py — Multi-signal scoring engine for the InternDedo ranking system.

Five scoring components combined with weighted sum:
  A. Career Fit Score (0.35)
  B. Skills Fit Score (0.25)
  C. Behavioral Score (0.20)
  D. Logistics Score (0.10)
  E. Background Score (0.10)
"""

import numpy as np
from typing import Dict
from config import (
    W_CAREER_FIT, W_SKILLS_FIT, W_BEHAVIORAL, W_LOGISTICS, W_BACKGROUND,
    EXP_SWEET_SPOT_CENTER, EXP_SWEET_SPOT_SIGMA,
    NOTICE_IDEAL_MAX, NOTICE_OK_MAX, NOTICE_PENALTY_THRESHOLD,
    SALARY_RANGE_MIN_LPA, SALARY_RANGE_MAX_LPA,
    STUFFER_PENALTY, AUTHENTICITY_PENALTY,
)
from utils import clamp, gaussian_score


def compute_composite_score(features: Dict, semantic_score: float = 0.0) -> Dict:
    """Compute the full composite score for a candidate.
    
    Args:
        features: Extracted feature dict from features.py
        semantic_score: Dense semantic similarity score (0-1) from embeddings
    
    Returns:
        Dict with component scores and final composite.
    """
    career = score_career_fit(features, semantic_score)
    skills = score_skills_fit(features)
    behavioral = score_behavioral(features)
    logistics = score_logistics(features)
    background = score_background(features)
    
    composite = (
        W_CAREER_FIT * career +
        W_SKILLS_FIT * skills +
        W_BEHAVIORAL * behavioral +
        W_LOGISTICS * logistics +
        W_BACKGROUND * background
    )
    
    # Apply stuffer penalty (multiplicative, not additive — it's severe)
    if features.get("is_likely_stuffer", False):
        composite *= 0.15  # Reduce to 15% of original score
    
    return {
        "composite": clamp(composite, 0.0, 1.0),
        "career_fit": career,
        "skills_fit": skills,
        "behavioral": behavioral,
        "logistics": logistics,
        "background": background,
        "is_stuffer": features.get("is_likely_stuffer", False),
    }


def score_career_fit(features: Dict, semantic_score: float) -> float:
    """Score career fit (0-1).
    
    Components:
    - Title trajectory (is current/past title in AI/ML/Data?)
    - Product vs consulting company ratio
    - Production deployment signal density
    - Ranking/search/recommendation signal density
    - Semantic similarity to JD (from embeddings)
    - Tenure stability (anti title-chaser)
    """
    scores = []
    
    # 1. Title trajectory (0-1) — weight: 0.25
    title_score = 0.0
    if features.get("is_ai_ml_title"):
        title_score = 1.0
    elif features.get("has_ai_ml_title_in_history"):
        title_score = 0.6
    elif not features.get("is_non_tech_title"):
        title_score = 0.2  # At least not a non-tech title
    scores.append(("title", 0.25, title_score))
    
    # 2. Product vs consulting (0-1) — weight: 0.15
    if features.get("all_consulting"):
        prod_score = 0.05  # Entire career at consulting = very low
    else:
        prod_score = 1.0 - features.get("consulting_ratio", 0) * 0.7
    scores.append(("product_vs_consulting", 0.15, clamp(prod_score)))
    
    # 3. Production deployment signals (0-1) — weight: 0.15
    prod_signals = features.get("production_signal_count", 0)
    prod_signal_score = clamp(prod_signals / 8.0)  # Normalize: 8+ signals = 1.0
    scores.append(("production_signals", 0.15, prod_signal_score))
    
    # 4. Ranking/search/recommendation signals (0-1) — weight: 0.15
    rank_signals = features.get("ranking_search_signal_count", 0)
    rank_signal_score = clamp(rank_signals / 6.0)  # Normalize: 6+ signals = 1.0
    scores.append(("ranking_search_signals", 0.15, rank_signal_score))
    
    # 5. Semantic similarity to JD (0-1) — weight: 0.20
    scores.append(("semantic_similarity", 0.20, clamp(semantic_score)))
    
    # 6. Tenure stability (0-1) — weight: 0.10
    avg_tenure = features.get("avg_tenure_months", 0)
    if avg_tenure >= 24:  # 2+ year average tenure — stable
        tenure_score = 1.0
    elif avg_tenure >= 18:
        tenure_score = 0.8
    elif avg_tenure >= 12:
        tenure_score = 0.5
    else:
        tenure_score = 0.2  # Very short stints — title chaser signal
    scores.append(("tenure_stability", 0.10, tenure_score))
    
    # Weighted sum
    return sum(w * s for _, w, s in scores)


def score_skills_fit(features: Dict) -> float:
    """Score skills fit (0-1).
    
    Components:
    - Must-have skill bucket coverage (4 buckets)
    - Skill authenticity (mention rate)
    - Assessment scores for relevant skills
    - Nice-to-have skills bonus
    """
    scores = []
    
    # 1. Must-have skill bucket coverage (0-1) — weight: 0.45
    # Each bucket is worth 0.25; having at least 1 match per bucket is key
    bucket_scores = []
    for bucket_name in ["embeddings_retrieval", "vector_db_search", 
                         "python_coding", "ranking_evaluation"]:
        count = features.get(f"must_have_{bucket_name}_count", 0)
        # 1+ match = full credit for this bucket; 0 = no credit
        bucket_scores.append(1.0 if count >= 1 else 0.0)
    
    must_have_score = sum(bucket_scores) / 4.0
    scores.append(("must_have_coverage", 0.45, must_have_score))
    
    # 2. Skill authenticity (0-1) — weight: 0.20
    mention_rate = features.get("skill_mention_rate", 0)
    authenticity_score = clamp(mention_rate / 0.4)  # 40% mention rate = 1.0
    scores.append(("skill_authenticity", 0.20, authenticity_score))
    
    # 3. Assessment scores (0-1) — weight: 0.20
    if features.get("relevant_assessment_count", 0) > 0:
        assess_score = features.get("relevant_assessment_avg", 0) / 100.0
    elif features.get("has_assessments"):
        assess_score = features.get("avg_assessment_score", 0) / 100.0 * 0.5
    else:
        assess_score = 0.3  # Neutral — many good candidates may not have taken assessments
    scores.append(("assessments", 0.20, clamp(assess_score)))
    
    # 4. Nice-to-have skills bonus (0-1) — weight: 0.15
    nice_count = sum(
        features.get(f"nice_to_have_{b}_count", 0)
        for b in ["llm_finetuning", "mlops_infra", "open_source"]
    )
    nice_score = clamp(nice_count / 5.0)  # 5+ nice-to-have matches = 1.0
    scores.append(("nice_to_have", 0.15, nice_score))
    
    return sum(w * s for _, w, s in scores)


def score_behavioral(features: Dict) -> float:
    """Score behavioral signals (0-1).
    
    Components:
    - Availability (recency, response rate, open to work)
    - Engagement (github, interview completion, profile completeness)
    - Recruiter-side signals (saved by, search appearances)
    """
    scores = []
    
    # 1. Availability (0-1) — weight: 0.40
    # Recency: how recently were they active?
    days_ago = features.get("last_active_days_ago", 365)
    recency = clamp(1.0 - days_ago / 365.0)  # Linear decay over 1 year
    
    # Response rate
    response_rate = features.get("recruiter_response_rate", 0)
    
    # Open to work
    open_flag = 1.0 if features.get("open_to_work") else 0.3
    
    # Response time (lower is better)
    resp_time = features.get("avg_response_time_hours", 999)
    resp_time_score = clamp(1.0 - resp_time / 200.0)  # <24h = great, >200h = bad
    
    availability = 0.30 * recency + 0.30 * response_rate + 0.20 * open_flag + 0.20 * resp_time_score
    scores.append(("availability", 0.40, clamp(availability)))
    
    # 2. Engagement (0-1) — weight: 0.35
    # GitHub activity
    github = features.get("github_activity_score", -1)
    github_score = clamp(github / 100.0) if github >= 0 else 0.2  # Neutral if no GitHub
    
    # Interview completion
    interview = features.get("interview_completion_rate", 0)
    
    # Profile completeness
    profile_comp = features.get("profile_completeness", 0) / 100.0
    
    # Verification
    verified = (
        (1.0 if features.get("verified_email") else 0.0) +
        (1.0 if features.get("verified_phone") else 0.0) +
        (1.0 if features.get("linkedin_connected") else 0.0)
    ) / 3.0
    
    engagement = 0.30 * github_score + 0.25 * interview + 0.25 * profile_comp + 0.20 * verified
    scores.append(("engagement", 0.35, clamp(engagement)))
    
    # 3. Recruiter-side signals (0-1) — weight: 0.25
    saved = features.get("saved_by_recruiters_30d", 0)
    saved_score = clamp(saved / 15.0)  # 15+ saves = excellent
    
    search = features.get("search_appearance_30d", 0)
    search_score = clamp(search / 200.0)  # 200+ appearances = excellent
    
    recruiter_signal = 0.50 * saved_score + 0.50 * search_score
    scores.append(("recruiter_signals", 0.25, clamp(recruiter_signal)))
    
    return sum(w * s for _, w, s in scores)


def score_logistics(features: Dict) -> float:
    """Score logistics fit (0-1).
    
    Components:
    - Location match
    - Notice period
    - Salary range alignment
    - Work mode preference
    - Willingness to relocate
    """
    scores = []
    
    # 1. Location (0-1) — weight: 0.30
    if features.get("is_preferred_city"):
        loc_score = 1.0
    elif features.get("is_india") and features.get("is_tier1_city"):
        loc_score = 0.7
    elif features.get("is_india"):
        loc_score = 0.5
    elif features.get("willing_to_relocate"):
        loc_score = 0.3
    else:
        loc_score = 0.15  # International, not willing to relocate
    scores.append(("location", 0.30, loc_score))
    
    # 2. Notice period (0-1) — weight: 0.25
    notice = features.get("notice_period_days", 90)
    if notice <= NOTICE_IDEAL_MAX:
        notice_score = 1.0
    elif notice <= NOTICE_OK_MAX:
        notice_score = 0.7
    elif notice <= NOTICE_PENALTY_THRESHOLD:
        notice_score = 0.4
    else:
        notice_score = 0.2
    scores.append(("notice_period", 0.25, notice_score))
    
    # 3. Salary alignment (0-1) — weight: 0.20
    salary_min = features.get("salary_min_lpa", 0)
    salary_max = features.get("salary_max_lpa", 0)
    if salary_min == 0 and salary_max == 0:
        salary_score = 0.5  # No data — neutral
    elif salary_min <= SALARY_RANGE_MAX_LPA and salary_max >= SALARY_RANGE_MIN_LPA:
        salary_score = 0.9  # Overlaps with our range
    elif salary_min > SALARY_RANGE_MAX_LPA:
        salary_score = 0.2  # Wants more than we can offer
    else:
        salary_score = 0.6  # Wants less — OK but maybe overqualified concern
    scores.append(("salary", 0.20, salary_score))
    
    # 4. Work mode (0-1) — weight: 0.15
    mode = features.get("preferred_work_mode", "")
    if mode in ("hybrid", "flexible"):
        mode_score = 1.0  # Perfect match — JD says hybrid
    elif mode == "onsite":
        mode_score = 0.8  # Onsite is fine too
    elif mode == "remote":
        mode_score = 0.4  # JD is hybrid-preferred
    else:
        mode_score = 0.5
    scores.append(("work_mode", 0.15, mode_score))
    
    # 5. Willingness to relocate (0-1) — weight: 0.10
    if features.get("is_preferred_city"):
        reloc_score = 1.0  # Already there
    elif features.get("willing_to_relocate"):
        reloc_score = 0.8
    else:
        reloc_score = 0.3
    scores.append(("relocate", 0.10, reloc_score))
    
    return sum(w * s for _, w, s in scores)


def score_background(features: Dict) -> float:
    """Score background fit (0-1).
    
    Components:
    - Education tier + field relevance
    - Experience sweet spot (Gaussian around 5-9 years)
    - Relevant certifications
    - Company size/industry fit
    """
    scores = []
    
    # 1. Education (0-1) — weight: 0.25
    tier = features.get("best_edu_tier", 5)
    tier_score = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}.get(tier, 0.2)
    
    field_bonus = 0.15 if features.get("has_cs_ai_education") else 0.0
    degree_bonus = 0.10 if features.get("has_masters_phd") else 0.0
    
    edu_score = clamp(tier_score + field_bonus + degree_bonus)
    scores.append(("education", 0.25, edu_score))
    
    # 2. Experience sweet spot (0-1) — weight: 0.35
    yrs = features.get("years_of_experience", 0)
    exp_score = gaussian_score(yrs, EXP_SWEET_SPOT_CENTER, EXP_SWEET_SPOT_SIGMA)
    scores.append(("experience_sweet_spot", 0.35, exp_score))
    
    # 3. Certifications (0-1) — weight: 0.15
    if features.get("has_relevant_certs"):
        cert_score = 0.8
    elif features.get("num_certs", 0) > 0:
        cert_score = 0.4
    else:
        cert_score = 0.3  # No certs is fine — most engineers don't bother
    scores.append(("certifications", 0.15, cert_score))
    
    # 4. Company context (0-1) — weight: 0.25
    # Prefer product companies, startup experience
    industry = features.get("current_industry", "").lower()
    company_size = features.get("profile", {})  # We don't have this extracted directly
    
    # Industry relevance
    if any(kw in industry for kw in ("software", "technology", "ai", "internet",
                                      "saas", "platform", "fintech")):
        industry_score = 1.0
    elif any(kw in industry for kw in ("it services", "consulting")):
        industry_score = 0.4
    else:
        industry_score = 0.3
    scores.append(("company_context", 0.25, industry_score))
    
    return sum(w * s for _, w, s in scores)
