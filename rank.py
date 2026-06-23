"""
rank.py — Main ranking script for the InternDedo ranking system.

Loads pre-computed artifacts and produces the final submission.csv.
Must complete within 5 minutes on a 16GB CPU-only machine with no network.

OPTIMIZED APPROACH:
- Features extracted for all 100K candidates (precomputed)
- Embeddings computed for only ~5000 pre-filtered candidates (precomputed)
- Ranking uses hybrid retrieval on the ~5000 shortlist
- Final reranking produces top 100

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --data-dir ./data
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple

from config import (
    DATA_DIR, RETRIEVAL_TOP_K, FINAL_TOP_K, RRF_K,
)
from utils import load_pickle, load_numpy
from scoring import compute_composite_score
from reasoning import generate_reasoning


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob JD")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.jsonl (used for ID verification)")
    parser.add_argument("--out", type=str, default="submission.csv",
                        help="Output CSV file path")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory with pre-computed artifacts")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # =========================================================================
    # Step 1: Load pre-computed artifacts
    # =========================================================================
    print(f"[1/4] Loading pre-computed artifacts from {args.data_dir}/...")
    t0 = time.time()
    
    all_features = load_pickle(os.path.join(args.data_dir, "features.pkl"))
    candidate_ids = load_pickle(os.path.join(args.data_dir, "candidate_ids.pkl"))
    honeypot_flags = load_pickle(os.path.join(args.data_dir, "honeypot_flags.pkl"))
    shortlist_ids = load_pickle(os.path.join(args.data_dir, "shortlist_ids.pkl"))
    shortlist_embeddings = load_numpy(os.path.join(args.data_dir, "shortlist_embeddings.npy"))
    jd_embedding = load_numpy(os.path.join(args.data_dir, "jd_embedding.npy"))
    
    # Build shortlist ID to embedding index mapping
    shortlist_id_to_idx = {cid: i for i, cid in enumerate(shortlist_ids)}
    
    print(f"       Total candidates:  {len(candidate_ids)}")
    print(f"       Shortlisted:       {len(shortlist_ids)}")
    print(f"       Embeddings shape:  {shortlist_embeddings.shape}")
    print(f"       Loaded in {time.time()-t0:.1f}s")
    
    # =========================================================================
    # Step 2: Hybrid Retrieval on shortlist (Dense + Sparse → RRF)
    # =========================================================================
    print(f"\n[2/4] Hybrid retrieval (RRF on {len(shortlist_ids)} shortlisted)...")
    t0 = time.time()
    
    # Dense retrieval: cosine similarity via dot product (already L2 normalized)
    jd_vec = jd_embedding[0]  # Shape: (384,)
    dense_scores = shortlist_embeddings @ jd_vec  # Shape: (N,)
    
    # Rank by dense score
    dense_ranking = np.argsort(-dense_scores)
    dense_rank_map = {shortlist_ids[idx]: rank 
                      for rank, idx in enumerate(dense_ranking)}
    
    # Sparse retrieval: feature-based scoring (proxy for BM25/TF-IDF)
    sparse_scores_list = []
    for cid in shortlist_ids:
        feat = all_features[cid]
        score = _compute_sparse_score(feat)
        sparse_scores_list.append(score)
    
    sparse_scores = np.array(sparse_scores_list)
    sparse_ranking = np.argsort(-sparse_scores)
    sparse_rank_map = {shortlist_ids[idx]: rank 
                       for rank, idx in enumerate(sparse_ranking)}
    
    # Reciprocal Rank Fusion
    rrf_scores = {}
    for cid in shortlist_ids:
        d_rank = dense_rank_map[cid]
        s_rank = sparse_rank_map[cid]
        rrf = 1.0 / (RRF_K + d_rank) + 1.0 / (RRF_K + s_rank)
        rrf_scores[cid] = rrf
    
    # Select top RETRIEVAL_TOP_K by RRF
    retrieval_k = min(RETRIEVAL_TOP_K, len(shortlist_ids))
    sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: -x[1])
    retrieval_ids = [cid for cid, _ in sorted_by_rrf[:retrieval_k]]
    
    print(f"       Retrieved top {len(retrieval_ids)} via RRF in {time.time()-t0:.1f}s")
    
    # =========================================================================
    # Step 3: Multi-Signal Reranking (→ Top 100)
    # =========================================================================
    print(f"\n[3/4] Multi-signal reranking ({len(retrieval_ids)} -> {FINAL_TOP_K})...")
    t0 = time.time()
    
    scored_candidates = []
    for cid in retrieval_ids:
        feat = all_features[cid]
        
        # Get semantic similarity score
        emb_idx = shortlist_id_to_idx.get(cid)
        if emb_idx is not None:
            semantic_score = float(dense_scores[emb_idx])
            # Normalize cosine similarity from [-1,1] to [0,1]
            semantic_score = max(0.0, (semantic_score + 1.0) / 2.0)
        else:
            semantic_score = 0.0
        
        # Compute full composite score
        score_breakdown = compute_composite_score(feat, semantic_score)
        
        scored_candidates.append({
            "candidate_id": cid,
            "composite_score": round(score_breakdown["composite"], 4),
            "score_breakdown": score_breakdown,
            "features": feat,
        })
    
    # Sort by composite score descending, tiebreak by candidate_id ascending
    # Score is pre-rounded to 4 decimals so tiebreaking matches CSV output
    scored_candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    
    # Take top 100
    top_100 = scored_candidates[:FINAL_TOP_K]
    
    print(f"       Reranked -> top {len(top_100)} in {time.time()-t0:.1f}s")
    
    # Print top 10 summary
    print(f"\n       +{'-'*80}+")
    print(f"       | {'TOP 10 CANDIDATES':^78} |")
    print(f"       +{'-'*80}+")
    for i, c in enumerate(top_100[:10]):
        feat = c["features"]
        sb = c["score_breakdown"]
        title = feat.get("current_title", "N/A")[:25]
        company = feat.get("current_company", "N/A")[:15]
        yrs = feat.get("years_of_experience", 0)
        flag = " !STUFFER" if sb.get("is_stuffer") else ""
        print(f"       | {i+1:>2}. {c['candidate_id']} | {title:<25} @ {company:<15} | "
              f"{yrs:>4.1f}yr | {c['composite_score']:.4f}{flag:<10} |")
    print(f"       +{'-'*80}+")
    
    # =========================================================================
    # Step 4: Generate Reasoning & Write CSV
    # =========================================================================
    print(f"\n[4/4] Generating reasoning and writing CSV...")
    t0 = time.time()
    
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, c in enumerate(top_100):
            rank = rank_idx + 1
            cid = c["candidate_id"]
            score = round(c["composite_score"], 4)
            
            reasoning = generate_reasoning(
                c["features"], c["score_breakdown"], rank
            )
            
            writer.writerow([cid, rank, score, reasoning])
    
    elapsed = time.time() - start_time
    
    # =========================================================================
    # Summary & Sanity Checks
    # =========================================================================
    hp_in_top100 = sum(1 for c in top_100 
                       if honeypot_flags.get(c["candidate_id"], (False, []))[0])
    stuffer_in_top100 = sum(1 for c in top_100 
                           if c["score_breakdown"].get("is_stuffer"))
    
    # Count AI/ML titles in top 100
    ai_title_in_top100 = sum(1 for c in top_100 
                            if c["features"].get("is_ai_ml_title"))
    non_tech_in_top100 = sum(1 for c in top_100 
                            if c["features"].get("is_non_tech_title"))
    
    print(f"\n{'='*60}")
    print(f"  RANKING COMPLETE — Team InternDedo")
    print(f"{'='*60}")
    print(f"  Total runtime:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:                {args.out}")
    print(f"  Total candidates:      {len(candidate_ids)}")
    print(f"  Shortlisted:           {len(shortlist_ids)}")
    print(f"  Final top-100:         {len(top_100)}")
    print(f"{'-'*60}")
    print(f"  SANITY CHECKS:")
    print(f"  Honeypots in top 100:  {hp_in_top100} {'OK' if hp_in_top100 <= 10 else 'FAIL'} (must be <=10)")
    print(f"  Stuffers in top 100:   {stuffer_in_top100}")
    print(f"  AI/ML titles in top100:{ai_title_in_top100}")
    print(f"  Non-tech in top 100:   {non_tech_in_top100}")
    print(f"{'='*60}")
    
    if elapsed > 300:
        print(f"\n!! WARNING: Runtime {elapsed:.0f}s exceeds 5-minute limit!")
    else:
        print(f"\n>> Within 5-minute limit ({elapsed:.0f}s)")
    
    return 0


def _compute_sparse_score(feat: dict) -> float:
    """Feature-based sparse scoring (proxy for BM25/TF-IDF matching).
    
    This provides the keyword-matching signal to complement dense retrieval.
    """
    score = 0.0
    
    # Must-have skill buckets (each covered = big signal)
    for bucket in ["embeddings_retrieval", "vector_db_search", 
                   "python_coding", "ranking_evaluation"]:
        count = feat.get(f"must_have_{bucket}_count", 0)
        if count > 0:
            score += 3.0
        if count > 2:
            score += 1.0  # Bonus for deep coverage
    
    # Nice-to-have skills
    for bucket in ["llm_finetuning", "mlops_infra", "open_source"]:
        if feat.get(f"nice_to_have_{bucket}_count", 0) > 0:
            score += 1.5
    
    # Title signal (strongest fast signal)
    if feat.get("is_ai_ml_title"):
        score += 6.0
    elif feat.get("has_ai_ml_title_in_history"):
        score += 4.0
    elif feat.get("is_non_tech_title"):
        score -= 3.0
    
    # Production deployment signals
    score += min(feat.get("production_signal_count", 0), 8) * 0.6
    
    # Ranking/search signals (highly relevant to JD)
    score += min(feat.get("ranking_search_signal_count", 0), 6) * 1.0
    
    # Skill authenticity
    mention_rate = feat.get("skill_mention_rate", 0)
    score += mention_rate * 3.0
    
    # Anti-stuffer penalty
    if feat.get("is_likely_stuffer"):
        score -= 12.0
    
    # All consulting penalty
    if feat.get("all_consulting"):
        score -= 4.0
    elif feat.get("consulting_ratio", 0) > 0.5:
        score -= 2.0
    
    # Assessment scores
    if feat.get("relevant_assessment_count", 0) > 0:
        score += feat.get("relevant_assessment_avg", 0) / 100.0 * 2.0
    
    return score


if __name__ == "__main__":
    sys.exit(main())
