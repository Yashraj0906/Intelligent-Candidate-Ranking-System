"""
reasoning.py — Per-candidate reasoning generation engine for the InternDedo ranking system.

Generates specific, fact-based 1-2 sentence reasoning for each ranked candidate.
No templates, no hallucination — every claim references actual profile data.
"""

from typing import Dict


def generate_reasoning(features: Dict, score_breakdown: Dict, rank: int) -> str:
    """Generate a specific, honest reasoning string for a ranked candidate.
    
    The reasoning must:
    - Reference specific facts from the candidate's profile
    - Connect to specific JD requirements
    - Acknowledge concerns where they exist
    - Not hallucinate skills, employers, or experience
    - Vary substantively between candidates
    """
    parts = []
    
    title = features.get("current_title", "Unknown")
    company = features.get("current_company", "Unknown")
    yrs = features.get("years_of_experience", 0)
    
    # === Positive signals ===
    
    # Career fit
    if features.get("is_ai_ml_title"):
        parts.append(f"{title} at {company} with {yrs:.1f} yrs experience")
    elif features.get("has_ai_ml_title_in_history"):
        parts.append(f"Currently {title} at {company} ({yrs:.1f} yrs) with prior AI/ML roles in career history")
    else:
        parts.append(f"{title} at {company} with {yrs:.1f} yrs experience")
    
    # Must-have skill coverage
    must_have_buckets = []
    for bucket_name, label in [
        ("embeddings_retrieval", "embeddings/retrieval"),
        ("vector_db_search", "vector DB/search"),
        ("python_coding", "Python"),
        ("ranking_evaluation", "ranking/evaluation"),
    ]:
        if features.get(f"must_have_{bucket_name}_count", 0) > 0:
            matched = features.get(f"must_have_{bucket_name}_matched", set())
            if matched:
                # Pick up to 2 most specific matched skills for the reasoning
                examples = sorted(matched)[:2]
                must_have_buckets.append(f"{label} ({', '.join(examples)})")
            else:
                must_have_buckets.append(label)
    
    if must_have_buckets:
        parts.append(f"covers {len(must_have_buckets)}/4 must-have skill areas: {'; '.join(must_have_buckets[:3])}")
    else:
        parts.append("limited coverage of JD must-have skills")
    
    # Production signals
    prod_count = features.get("production_signal_count", 0)
    if prod_count >= 5:
        parts.append("strong production deployment signals in career history")
    elif prod_count >= 3:
        parts.append("some production experience indicated")
    
    # Ranking/search signals
    rank_count = features.get("ranking_search_signal_count", 0)
    if rank_count >= 4:
        parts.append("direct ranking/search/retrieval system experience")
    
    # === Concerns ===
    concerns = []
    
    # Experience range
    if yrs < 4:
        concerns.append(f"below JD's 5-9yr preferred range ({yrs:.1f} yrs)")
    elif yrs > 12:
        concerns.append(f"above JD's 5-9yr preferred range ({yrs:.1f} yrs)")
    
    # Notice period
    notice = features.get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"{notice}-day notice period")
    
    # Location
    if not features.get("is_india"):
        country = features.get("country", "unknown").title()
        if features.get("willing_to_relocate"):
            concerns.append(f"based in {country} but willing to relocate")
        else:
            concerns.append(f"based in {country}, not willing to relocate")
    
    # Consulting
    if features.get("all_consulting"):
        concerns.append("entire career at consulting/services firms")
    elif features.get("consulting_ratio", 0) > 0.5:
        concerns.append("majority of career at consulting firms")
    
    # Availability
    days_ago = features.get("last_active_days_ago", 0)
    response_rate = features.get("recruiter_response_rate", 0)
    if days_ago > 180:
        concerns.append(f"last active {days_ago} days ago")
    if response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    
    # Non-tech title concern
    if features.get("is_non_tech_title") and not features.get("has_ai_ml_title_in_history"):
        concerns.append(f"non-technical title ({title}) without AI/ML career history")
    
    # GitHub
    github = features.get("github_activity_score", -1)
    if github >= 0 and github >= 50:
        parts.append(f"active GitHub contributor (score: {github:.0f})")
    
    # === Assemble ===
    reasoning = "; ".join(parts)
    if concerns:
        reasoning += ". Concerns: " + "; ".join(concerns[:3])  # Limit to 3 concerns
    
    # Add a period if not already there
    if not reasoning.endswith("."):
        reasoning += "."
    
    return reasoning
