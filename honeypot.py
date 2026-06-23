"""
honeypot.py — Honeypot detection heuristics for the InternDedo ranking system.

Honeypots are ~80 candidates with subtly impossible profiles. They are forced
to relevance tier 0 in the ground truth. >10% honeypots in top 100 = disqualified.

Detection strategy: Multiple independent heuristics, each catching a different
type of impossibility. A candidate flagged by 2+ heuristics is very likely a honeypot.
"""

import re
from typing import Dict, List, Tuple
from config import (
    HONEYPOT_EXP_CAREER_MISMATCH_MONTHS,
    HONEYPOT_EXPERT_ZERO_DURATION,
    HONEYPOT_PERFECT_SKILL_COUNT,
    HONEYPOT_SAME_PROFICIENCY_MIN,
)


def detect_honeypot(candidate: Dict) -> Tuple[bool, List[str]]:
    """Run all honeypot heuristics on a candidate.
    
    Returns:
        (is_honeypot: bool, reasons: list of strings explaining why)
    
    A candidate is flagged if they trigger 2+ heuristics.
    """
    flags = []
    
    # H1: Experience vs career history duration mismatch
    flag = _check_experience_mismatch(candidate)
    if flag:
        flags.append(flag)
    
    # H2: Expert/advanced skills with zero duration
    flag = _check_expert_zero_duration(candidate)
    if flag:
        flags.append(flag)
    
    # H3: Suspiciously perfect skill breadth (many advanced/expert, no assessments)
    flag = _check_perfect_skill_breadth(candidate)
    if flag:
        flags.append(flag)
    
    # H4: All skills same proficiency level (synthetic pattern)
    flag = _check_uniform_proficiency(candidate)
    if flag:
        flags.append(flag)
    
    # H5: Title vs career description content mismatch
    flag = _check_title_description_mismatch(candidate)
    if flag:
        flags.append(flag)
    
    # H6: Impossible company tenure (duration > what's reasonable)
    flag = _check_impossible_tenure(candidate)
    if flag:
        flags.append(flag)
    
    # H7: Skill endorsements inconsistency
    flag = _check_endorsement_anomaly(candidate)
    if flag:
        flags.append(flag)
    
    # Flag as honeypot if 2+ heuristics triggered
    is_honeypot = len(flags) >= 2
    
    return is_honeypot, flags


def _check_experience_mismatch(candidate: Dict) -> str:
    """H1: Stated years_of_experience wildly mismatches sum of career durations."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    
    stated_yrs = profile.get("years_of_experience", 0)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    total_career_yrs = total_career_months / 12.0
    
    # Check if stated experience is WAY more than career history accounts for
    diff_months = abs(stated_yrs * 12 - total_career_months)
    if diff_months > HONEYPOT_EXP_CAREER_MISMATCH_MONTHS:
        return (f"H1: Experience mismatch — stated {stated_yrs:.1f} yrs but career "
                f"history sums to {total_career_yrs:.1f} yrs (diff: {diff_months/12:.1f} yrs)")
    
    return ""


def _check_expert_zero_duration(candidate: Dict) -> str:
    """H2: Skills listed as 'expert' or 'advanced' but with 0 months duration."""
    if not HONEYPOT_EXPERT_ZERO_DURATION:
        return ""
    
    skills = candidate.get("skills", [])
    suspicious = []
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", 0)
        if prof in ("expert", "advanced") and dur == 0:
            suspicious.append(f"{s['name']}({prof}, 0mo)")
    
    if len(suspicious) >= 2:
        return f"H2: Expert/advanced skills with 0 duration: {', '.join(suspicious[:5])}"
    
    return ""


def _check_perfect_skill_breadth(candidate: Dict) -> str:
    """H3: Too many advanced/expert skills + no assessment scores taken."""
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    high_prof_count = sum(1 for s in skills 
                         if s.get("proficiency", "").lower() in ("advanced", "expert"))
    
    assessments = signals.get("skill_assessment_scores", {})
    
    if high_prof_count >= HONEYPOT_PERFECT_SKILL_COUNT and len(assessments) == 0:
        return (f"H3: {high_prof_count} advanced/expert skills but zero assessment "
                f"scores — suspiciously perfect without validation")
    
    return ""


def _check_uniform_proficiency(candidate: Dict) -> str:
    """H4: All/most skills at the exact same proficiency level (synthetic pattern)."""
    skills = candidate.get("skills", [])
    if len(skills) < HONEYPOT_SAME_PROFICIENCY_MIN:
        return ""
    
    proficiencies = [s.get("proficiency", "").lower() for s in skills]
    from collections import Counter
    counter = Counter(proficiencies)
    most_common_count = counter.most_common(1)[0][1] if counter else 0
    
    if most_common_count >= HONEYPOT_SAME_PROFICIENCY_MIN and most_common_count == len(skills):
        return (f"H4: All {len(skills)} skills at '{counter.most_common(1)[0][0]}' — "
                f"synthetic uniform proficiency pattern")
    
    return ""


def _check_title_description_mismatch(candidate: Dict) -> str:
    """H5: Job title doesn't match the content of the role description.
    
    e.g., title is 'Mechanical Engineer' but description talks about 
    'content writing and SEO strategy'.
    """
    career = candidate.get("career_history", [])
    mismatches = 0
    
    # Domain keywords associated with titles
    title_domain_keywords = {
        "engineer": {"engineering", "technical", "code", "system", "build", "develop", "implement",
                     "architecture", "infrastructure", "deploy", "software", "platform"},
        "marketing": {"marketing", "brand", "campaign", "content", "seo", "social media", "growth",
                      "acquisition", "advertising", "digital marketing"},
        "accountant": {"accounting", "finance", "tax", "audit", "ledger", "financial", "compliance",
                       "gaap", "revenue", "invoice", "bookkeeping"},
        "hr": {"human resources", "recruitment", "hiring", "talent", "employee", "benefits",
               "onboarding", "performance review", "workforce"},
        "sales": {"sales", "revenue", "quota", "pipeline", "client", "crm", "deal", "prospect",
                  "business development"},
        "designer": {"design", "creative", "visual", "ui", "ux", "figma", "adobe", "illustrator",
                     "branding", "typography"},
        "support": {"support", "ticket", "customer", "helpdesk", "escalation", "resolution",
                    "service level"},
    }
    
    for role in career:
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        
        if not desc:
            continue
        
        # Find what domain the title belongs to
        title_domain = None
        for domain, keywords in title_domain_keywords.items():
            if domain in title:
                title_domain = domain
                break
        
        if not title_domain:
            continue
        
        # Check if description content matches the title's domain
        expected_keywords = title_domain_keywords.get(title_domain, set())
        desc_match_count = sum(1 for kw in expected_keywords if kw in desc)
        
        # Check if description actually matches a DIFFERENT domain better
        best_other_domain = None
        best_other_score = 0
        for other_domain, other_keywords in title_domain_keywords.items():
            if other_domain == title_domain:
                continue
            other_score = sum(1 for kw in other_keywords if kw in desc)
            if other_score > best_other_score:
                best_other_score = other_score
                best_other_domain = other_domain
        
        # Mismatch: another domain matches the description better than the stated title
        if best_other_score > desc_match_count + 2 and best_other_score >= 3:
            mismatches += 1
    
    if mismatches >= 2:
        return f"H5: {mismatches} career roles where title doesn't match description content"
    
    return ""


def _check_impossible_tenure(candidate: Dict) -> str:
    """H6: Career history has impossibly long tenures or overlapping dates."""
    career = candidate.get("career_history", [])
    
    for role in career:
        duration = role.get("duration_months", 0)
        # No role should be 20+ years (240 months) — that's suspicious
        if duration > 240:
            return f"H6: Impossible tenure of {duration} months at {role.get('company', 'unknown')}"
    
    # Check for significant overlaps
    dated_roles = []
    for role in career:
        try:
            start = role.get("start_date", "")
            end = role.get("end_date", "")
            if start:
                from datetime import datetime
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime(2026, 6, 1)
                dated_roles.append((start_dt, end_dt, role.get("company", "")))
        except (ValueError, TypeError):
            continue
    
    # Sort by start date and check for major overlaps
    dated_roles.sort(key=lambda x: x[0])
    for i in range(len(dated_roles) - 1):
        _, end1, comp1 = dated_roles[i]
        start2, _, comp2 = dated_roles[i + 1]
        overlap_days = (end1 - start2).days
        if overlap_days > 180 and comp1 != comp2:  # >6 months overlap at different companies
            return (f"H6: {overlap_days} day overlap between {comp1} and {comp2}")
    
    return ""


def _check_endorsement_anomaly(candidate: Dict) -> str:
    """H7: Very high endorsements on skills with very low duration, or vice versa."""
    skills = candidate.get("skills", [])
    
    anomalies = 0
    for s in skills:
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)
        proficiency = s.get("proficiency", "").lower()
        
        # Expert with 50+ endorsements but only 1-2 months? Suspicious.
        if proficiency == "expert" and endorsements > 40 and duration <= 2:
            anomalies += 1
        
        # Beginner with 50+ endorsements? Also suspicious.
        if proficiency == "beginner" and endorsements > 40:
            anomalies += 1
    
    if anomalies >= 2:
        return f"H7: {anomalies} skills with anomalous endorsement-to-duration ratios"
    
    return ""
