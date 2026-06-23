"""
app.py — HuggingFace Spaces Gradio app for the InternDedo ranking system.

Provides a sandbox/demo interface where organizers can:
1. Upload a small candidate sample (≤100 candidates as JSON)
2. Run the ranking pipeline end-to-end
3. See ranked results with reasoning

Usage (local):
  python app.py

Deployment:
  Upload to HuggingFace Spaces with Gradio SDK.
"""

import json
import os
import sys
import tempfile
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

# Lazy-load model
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def rank_candidates(candidates_json: str, top_k: int = 100) -> str:
    """Main ranking function for the Gradio interface.
    
    Args:
        candidates_json: JSON string with array of candidate objects
        top_k: Number of top candidates to return (max 100)
    
    Returns:
        CSV string with ranked candidates
    """
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
        return f"Too many candidates ({len(candidates)}). Please provide ≤500 for the demo."
    
    top_k = min(top_k, len(candidates), FINAL_TOP_K)
    
    # Step 1: Feature extraction + honeypot detection
    all_features = {}
    honeypot_flags = {}
    candidate_texts = []
    candidate_ids = []
    
    for cand in candidates:
        cid = cand.get("candidate_id", "UNKNOWN")
        feat = extract_features(cand)
        is_hp, reasons = detect_honeypot(cand)
        all_features[cid] = feat
        honeypot_flags[cid] = (is_hp, reasons)
        candidate_texts.append(build_candidate_text(cand))
        candidate_ids.append(cid)
    
    # Step 2: Compute embeddings
    model = get_model()
    jd_text = build_jd_text()
    
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)[0]
    cand_embeddings = model.encode(candidate_texts, normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False)
    
    # Step 3: Dense similarity
    dense_scores = cand_embeddings @ jd_embedding
    
    # Step 4: Filter honeypots and score
    scored = []
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
    
    # Sort and take top K
    scored.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    top = scored[:top_k]
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    
    for rank_idx, c in enumerate(top):
        rank = rank_idx + 1
        score = round(c["composite_score"], 4)
        reason = generate_reasoning(c["features"], c["score_breakdown"], rank)
        writer.writerow([c["candidate_id"], rank, score, reason])
    
    elapsed = time.time() - start_time
    
    result = output.getvalue()
    summary = (f"\n\n---\nProcessed {len(candidates)} candidates in {elapsed:.1f}s. "
               f"Filtered {sum(1 for v in honeypot_flags.values() if v[0])} honeypots. "
               f"Returned top {len(top)}.")
    
    return result + summary


def create_app():
    """Create the Gradio interface."""
    import gradio as gr
    
    demo = gr.Interface(
        fn=rank_candidates,
        inputs=[
            gr.Textbox(
                label="Candidate Data (JSON array)",
                placeholder="Paste JSON array of candidates here...",
                lines=15,
            ),
            gr.Slider(
                minimum=10, maximum=100, value=100, step=10,
                label="Top K candidates to return",
            ),
        ],
        outputs=gr.Textbox(label="Ranked Results (CSV)", lines=20),
        title="🎯 InternDedo — Intelligent Candidate Ranker",
        description=(
            "**Team InternDedo** — Redrob Hackathon\n\n"
            "Upload candidate profiles as a JSON array and get AI-powered ranking "
            "for the Senior AI Engineer role. Uses BGE-small embeddings, hybrid retrieval, "
            "multi-signal scoring, and honeypot detection."
        ),
        examples=None,
        allow_flagging="never",
    )
    
    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False)
