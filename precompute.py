"""
precompute.py — Offline pre-computation for the InternDedo ranking system.

OPTIMIZED STRATEGY:
Instead of embedding all 100K candidates (which takes 2+ hours on CPU),
we extract features and detect honeypots for all candidates (fast), then
pre-filter down to ~5000 viable candidates based on structured features,
and ONLY embed those ~5000. This keeps total precompute under 20 minutes.

Artifacts produced:
  - data/features.pkl        — Extracted feature dicts for all candidates
  - data/candidate_ids.pkl   — Ordered list of candidate IDs
  - data/honeypot_flags.pkl  — Dict of candidate_id -> (is_honeypot, reasons)
  - data/shortlist_ids.pkl   — Pre-filtered candidate IDs (~5000)
  - data/shortlist_embeddings.npy — BGE embeddings for shortlisted candidates only
  - data/jd_embedding.npy    — JD text embedding

Usage:
  python precompute.py --candidates ./candidates.jsonl
  python precompute.py --candidates ./candidates.jsonl --data-dir ./data
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

from config import EMBEDDING_MODEL, EMBEDDING_DIM, DATA_DIR, EXP_HARD_MIN, EXP_HARD_MAX
from utils import (
    load_candidates_jsonl, save_pickle, save_numpy,
    build_candidate_text, build_jd_text,
)
from features import extract_features
from honeypot import detect_honeypot


# =========================================================================
# Pre-filter scoring — fast, feature-based. Decides who gets embedded.
# =========================================================================
def prefilter_score(feat: dict) -> float:
    """Quick score to decide if a candidate is worth embedding.
    
    This is NOT the final ranking score — it's a fast heuristic to
    narrow 100K candidates to ~5000 worth of embedding compute.
    
    We're intentionally generous here to avoid missing hidden gems.
    """
    score = 0.0
    
    # AI/ML title is the strongest fast signal
    if feat.get("is_ai_ml_title"):
        score += 10.0
    elif feat.get("has_ai_ml_title_in_history"):
        score += 6.0
    elif not feat.get("is_non_tech_title"):
        score += 2.0  # At least not a clearly non-tech title
    
    # Must-have skill bucket coverage
    for bucket in ["embeddings_retrieval", "vector_db_search", 
                   "python_coding", "ranking_evaluation"]:
        count = feat.get(f"must_have_{bucket}_count", 0)
        if count > 0:
            score += 3.0
    
    # Nice-to-have skills
    for bucket in ["llm_finetuning", "mlops_infra", "open_source"]:
        if feat.get(f"nice_to_have_{bucket}_count", 0) > 0:
            score += 1.0
    
    # Production signals in career descriptions
    prod = feat.get("production_signal_count", 0)
    score += min(prod, 8) * 0.5
    
    # Ranking/search signals
    rank_sig = feat.get("ranking_search_signal_count", 0)
    score += min(rank_sig, 6) * 0.75
    
    # Experience in range
    yrs = feat.get("years_of_experience", 0)
    if 5.0 <= yrs <= 9.0:
        score += 2.0
    elif 3.0 <= yrs <= 12.0:
        score += 1.0
    
    # India location bonus
    if feat.get("is_india"):
        score += 1.0
    if feat.get("is_preferred_city"):
        score += 1.0
    
    # Anti-stuffer penalty
    if feat.get("is_likely_stuffer"):
        score -= 8.0
    
    # All-consulting penalty
    if feat.get("all_consulting"):
        score -= 3.0
    
    # Behavioral signals (lightweight)
    if feat.get("recruiter_response_rate", 0) > 0.3:
        score += 0.5
    if feat.get("open_to_work"):
        score += 0.5
    if feat.get("github_activity_score", -1) > 30:
        score += 1.0
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Pre-compute features and embeddings")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.jsonl file")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory to save pre-computed artifacts")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for embedding computation")
    parser.add_argument("--shortlist-size", type=int, default=5000,
                        help="Number of candidates to shortlist for embedding")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Max candidates to process (for testing)")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load candidates
    # =========================================================================
    print(f"[1/6] Loading candidates from {args.candidates}...")
    t0 = time.time()
    candidates = load_candidates_jsonl(args.candidates, max_candidates=args.max_candidates)
    # Build a lookup dict
    candidates_by_id = {c.get("candidate_id", ""): c for c in candidates}
    print(f"       Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")
    
    # =========================================================================
    # Step 2: Extract features for ALL candidates
    # =========================================================================
    print(f"[2/6] Extracting features for all candidates...")
    t0 = time.time()
    all_features = {}
    candidate_ids = []
    for cand in tqdm(candidates, desc="  Features"):
        cid = cand.get("candidate_id", "")
        feat = extract_features(cand)
        all_features[cid] = feat
        candidate_ids.append(cid)
    
    save_pickle(all_features, os.path.join(args.data_dir, "features.pkl"))
    save_pickle(candidate_ids, os.path.join(args.data_dir, "candidate_ids.pkl"))
    print(f"       Extracted features for {len(all_features)} candidates in {time.time()-t0:.1f}s")
    
    # =========================================================================
    # Step 3: Detect honeypots
    # =========================================================================
    print(f"[3/6] Running honeypot detection...")
    t0 = time.time()
    honeypot_flags = {}
    honeypot_count = 0
    for cand in tqdm(candidates, desc="  Honeypots"):
        cid = cand.get("candidate_id", "")
        is_hp, reasons = detect_honeypot(cand)
        honeypot_flags[cid] = (is_hp, reasons)
        if is_hp:
            honeypot_count += 1
    
    save_pickle(honeypot_flags, os.path.join(args.data_dir, "honeypot_flags.pkl"))
    print(f"       Detected {honeypot_count} honeypots in {time.time()-t0:.1f}s")
    
    # =========================================================================
    # Step 4: Pre-filter to shortlist (based on features, no embeddings)
    # =========================================================================
    print(f"[4/6] Pre-filtering to shortlist (target: {args.shortlist_size})...")
    t0 = time.time()
    
    # Score all non-honeypot candidates
    prefilter_scores = []
    for cid in candidate_ids:
        is_hp, _ = honeypot_flags.get(cid, (False, []))
        if is_hp:
            continue
        
        feat = all_features[cid]
        yrs = feat.get("years_of_experience", 0)
        if yrs < EXP_HARD_MIN or yrs > EXP_HARD_MAX:
            continue
        
        pf_score = prefilter_score(feat)
        prefilter_scores.append((cid, pf_score))
    
    # Sort by prefilter score and take top N
    prefilter_scores.sort(key=lambda x: -x[1])
    shortlist_ids = [cid for cid, _ in prefilter_scores[:args.shortlist_size]]
    
    save_pickle(shortlist_ids, os.path.join(args.data_dir, "shortlist_ids.pkl"))
    print(f"       Shortlisted {len(shortlist_ids)} candidates from "
          f"{len(prefilter_scores)} non-honeypots in {time.time()-t0:.1f}s")
    
    # Print shortlist stats
    ai_title_count = sum(1 for cid in shortlist_ids if all_features[cid].get("is_ai_ml_title"))
    non_tech_count = sum(1 for cid in shortlist_ids if all_features[cid].get("is_non_tech_title"))
    print(f"       Shortlist: {ai_title_count} AI/ML titles, {non_tech_count} non-tech titles")
    
    # =========================================================================
    # Step 5: Compute embeddings for SHORTLISTED candidates only
    # =========================================================================
    print(f"[5/6] Computing BGE embeddings for {len(shortlist_ids)} shortlisted candidates...")
    print(f"       (This should take ~2-5 minutes on CPU)")
    t0 = time.time()
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Build texts for shortlisted candidates only
    shortlist_texts = []
    for cid in tqdm(shortlist_ids, desc="  Texts"):
        cand = candidates_by_id[cid]
        shortlist_texts.append(build_candidate_text(cand))
    
    # Encode
    shortlist_embeddings = model.encode(
        shortlist_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    save_numpy(np.array(shortlist_embeddings, dtype=np.float32),
               os.path.join(args.data_dir, "shortlist_embeddings.npy"))
    # Save candidate texts for cross-encoder reranking stage
    save_pickle(shortlist_texts, os.path.join(args.data_dir, "shortlist_texts.pkl"))
    print(f"       Computed embeddings shape {shortlist_embeddings.shape} in {time.time()-t0:.1f}s")
    print(f"       Saved candidate texts for cross-encoder reranking")
    
    # =========================================================================
    # Step 6: Compute JD embedding
    # =========================================================================
    print(f"[6/6] Computing JD embedding...")
    jd_text = build_jd_text()
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)
    save_numpy(np.array(jd_embedding, dtype=np.float32),
               os.path.join(args.data_dir, "jd_embedding.npy"))
    
    total_time = time.time()
    print(f"\n{'='*60}")
    print(f"Pre-computation complete!")
    print(f"  Candidates:       {len(candidates)}")
    print(f"  Honeypots:        {honeypot_count}")
    print(f"  Shortlisted:      {len(shortlist_ids)} (for embedding)")
    print(f"  Embedding shape:  {shortlist_embeddings.shape}")
    print(f"  Artifacts dir:    {args.data_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
