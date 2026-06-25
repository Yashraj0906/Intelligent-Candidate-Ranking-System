"""
app.py — HuggingFace Spaces Gradio app for the InternDedo ranking system.

Provides a sandbox/demo interface where organizers can:
1. Upload a small candidate sample (JSON array)
2. Run the full ranking pipeline end-to-end
3. See ranked results with reasoning and score breakdowns

For HuggingFace Spaces deployment:
  - SDK: Gradio
  - Hardware: CPU Basic (free)
  - Files needed: app.py, config.py, utils.py, features.py, honeypot.py,
                   scoring.py, reasoning.py, requirements.txt

Usage (local):
  python app.py
"""

import json
import os
import sys
import time
import csv
import io
import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FINAL_TOP_K, RRF_K, EMBEDDING_MODEL
from features import extract_features
from honeypot import detect_honeypot
from scoring import compute_composite_score
from reasoning import generate_reasoning
from utils import build_candidate_text, build_jd_text

# Lazy-load models to avoid loading on import
_bi_encoder = None
_cross_encoder = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_WEIGHT = 0.30


def get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        _bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _bi_encoder


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    return _cross_encoder

def rank_candidates(candidates_json: str, top_k: int = 100, use_cross_encoder: bool = True) -> str:
    """Main ranking function for the Gradio interface."""
    top_k = int(top_k)
    
    try:
        return _rank_candidates_impl(candidates_json, top_k, use_cross_encoder)
    except Exception as e:
        import traceback
        return f"ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}"


def _rank_candidates_impl(candidates_json: str, top_k: int, use_cross_encoder: bool) -> str:
    start_time = time.time()
    
    # Parse candidates
    try:
        candidates = json.loads(candidates_json)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    
    if len(candidates) == 0:
        return "No candidates provided."
    
    if len(candidates) > 500:
        return f"Too many candidates ({len(candidates)}). Please provide <=500 for the demo."
    
    top_k = min(top_k, len(candidates), FINAL_TOP_K)
    
    # Step 1: Feature extraction + honeypot detection
    all_features = {}
    honeypot_flags = {}
    candidate_texts = []
    candidate_ids = []
    honeypot_count = 0
    
    for cand in candidates:
        cid = cand.get("candidate_id", "UNKNOWN")
        feat = extract_features(cand)
        is_hp, reasons = detect_honeypot(cand)
        all_features[cid] = feat
        honeypot_flags[cid] = (is_hp, reasons)
        candidate_texts.append(build_candidate_text(cand))
        candidate_ids.append(cid)
        if is_hp:
            honeypot_count += 1
    
    # Step 2: Compute embeddings
    model = get_bi_encoder()
    jd_text = build_jd_text()
    
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)[0]
    cand_embeddings = model.encode(candidate_texts, normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False)
    
    # Step 3: Dense similarity
    dense_scores = cand_embeddings @ jd_embedding
    
    # Step 4: Filter honeypots and compute multi-signal scores
    scored = []
    valid_texts = {}
    for i, cid in enumerate(candidate_ids):
        is_hp, _ = honeypot_flags[cid]
        if is_hp:
            continue
        
        feat = all_features[cid]
        semantic = float(max(0, (dense_scores[i] + 1) / 2))
        breakdown = compute_composite_score(feat, semantic)
        
        scored.append({
            "candidate_id": cid,
            "composite_score": breakdown["composite"],
            "score_breakdown": breakdown,
            "features": feat,
        })
        valid_texts[cid] = candidate_texts[i]
    
    # Sort by composite score
    scored.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    
    # Step 5: Cross-encoder reranking on top 200
    if use_cross_encoder and len(scored) > 10:
        try:
            ce_model = get_cross_encoder()
            top_for_ce = scored[:min(200, len(scored))]
            
            pairs = [[jd_text[:1500], valid_texts.get(c["candidate_id"], "")[:1500]] 
                     for c in top_for_ce]
            
            ce_scores = ce_model.predict(pairs, batch_size=32, show_progress_bar=False)
            ce_min, ce_max = float(np.min(ce_scores)), float(np.max(ce_scores))
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
            
            for j, c in enumerate(top_for_ce):
                ce_norm = (float(ce_scores[j]) - ce_min) / ce_range
                c["final_score"] = round(
                    (1 - CROSS_ENCODER_WEIGHT) * c["composite_score"] + 
                    CROSS_ENCODER_WEIGHT * ce_norm, 4
                )
            
            top_for_ce.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
            scored = top_for_ce
            ce_used = True
        except Exception as e:
            # Fallback if cross-encoder fails
            for c in scored:
                c["final_score"] = round(c["composite_score"], 4)
            ce_used = False
    else:
        for c in scored:
            c["final_score"] = round(c["composite_score"], 4)
        ce_used = False
    
    top = scored[:top_k]
    
    # Generate CSV output
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    
    for rank_idx, c in enumerate(top):
        rank = rank_idx + 1
        score = c["final_score"]
        reason = generate_reasoning(c["features"], c["score_breakdown"], rank)
        writer.writerow([c["candidate_id"], rank, score, reason])
    
    elapsed = time.time() - start_time
    
    result = output.getvalue()
    
    # Add summary stats
    ai_titles = sum(1 for c in top if c["features"].get("is_ai_ml_title"))
    summary = (
        f"\n\n--- RANKING SUMMARY ---\n"
        f"Candidates processed: {len(candidates)}\n"
        f"Honeypots detected:   {honeypot_count}\n"
        f"Cross-encoder used:   {'Yes' if ce_used else 'No'}\n"
        f"AI/ML titles in top:  {ai_titles}/{len(top)}\n"
        f"Runtime:              {elapsed:.1f}s\n"
    )
    
    return result + summary


def create_app():
    """Create the Gradio interface."""
    import gradio as gr
    
    with gr.Blocks(
        title="InternDedo - Intelligent Candidate Ranker",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    ) as demo:
        gr.Markdown(
            """
            # 🎯 InternDedo — Intelligent Candidate Ranking System
            ### Redrob Hackathon: India Runs Data & AI Challenge 2026
            
            **How it works:**
            1. Paste candidate profiles as a JSON array below
            2. The system runs a 5-stage pipeline: Feature Extraction -> Honeypot Detection -> 
               Hybrid Retrieval (BGE + RRF) -> Multi-Signal Scoring -> Cross-Encoder Reranking
            3. Get AI-powered ranking with per-candidate reasoning
            
            **Tech Stack:** BGE-small-en-v1.5 (bi-encoder) + ms-marco-MiniLM-L-6-v2 (cross-encoder) + 
            Reciprocal Rank Fusion + 5-component multi-signal scoring
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                input_json = gr.Textbox(
                    label="Candidate Data (JSON array)",
                    placeholder='[{"candidate_id": "CAND_001", "profile": {...}, "skills": [...], ...}]',
                    lines=15,
                )
            with gr.Column(scale=1):
                top_k = gr.Slider(
                    minimum=10, maximum=100, value=100, step=10,
                    label="Top K candidates to return",
                )
                use_ce = gr.Checkbox(
                    value=True,
                    label="Use Cross-Encoder Reranking (slower but better NDCG@10)",
                )
                run_btn = gr.Button("Rank Candidates", variant="primary", size="lg")
        
        output = gr.Textbox(label="Ranked Results (CSV + Summary)", lines=25)
        
        run_btn.click(
            fn=rank_candidates,
            inputs=[input_json, top_k, use_ce],
            outputs=output,
        )
        
        gr.Markdown(
            """
            ---
            **Team InternDedo** | Built for Redrob Hackathon 2026
            
            Architecture: Pre-filter (100K -> 5K) -> BGE-small embeddings -> 
            Hybrid Retrieval (Dense + Sparse + RRF) -> Multi-Signal Reranker (5 components) -> 
            Cross-Encoder Reranker -> Fact-based Reasoning Generation
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False)
