#!/usr/bin/env python3
"""
Semantic Tutor Demo — Shows the 5-layer matching cascade in action.

Q&A system that demonstrates:
- Exact matching (FORTRAN 1960 style)
- Bitvector matching (TUTOR 1965 style)  
- Semantic matching (Model2Vec 2024 style)
- Domain-aware matching (PLATO 2026 style)
- Deadband thresholding (knowledge relevance gating)
"""

import sys
import os

# Add parent to path so we can import plato_training
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

knowledge_base = {
    # PLATO Architecture
    "What is PLATO?": "PLATO is a persistent knowledge system using tiles, rooms, and Lamport clocks for agent coordination.",
    "How many tests does plato-training have?": "plato-training has 665+ tests across micro models, GPU training, collective inference, and semantic matching modules.",
    "What is a TrainingTile?": "A TrainingTile is a content-addressed knowledge artifact with lifecycle states (Active/Superseded/Retracted) and Lamport clock ordering.",
    "What is SplineLinear?": "SplineLinear uses Eisenstein lattice weight parameterization for 5-20x compression of neural network weights.",
    # Fleet Operations
    "How many agents in the fleet?": "The Cocapn fleet has 9 agents coordinated via PLATO rooms and Matrix bridges.",
    "What is collective inference?": "Collective inference: predict → observe → compare → gap → learn → share across fleet agents.",
    # Constraint Theory
    "What is constraint theory?": "Constraint theory proves that Eisenstein integer arithmetic eliminates floating-point drift in geometric constraints.",
    "What is spectral conservation?": "Spectral conservation is the property that norm-1 Eisenstein integers preserve spectral energy under rotation.",
    # Tools
    "What is eisenstein-embed?": "eisenstein-embed is a standalone package that enhances Model2Vec with a 5-layer matching cascade: exact → bitvector → semantic → domain → deadband.",
    "What is deadband-rs?": "deadband-rs is a Rust library implementing Bayesian Model Averaging, Fibonacci spline thresholds, and HPDF variance deadband algorithms.",
    # Hardware
    "What GPU do we have?": "NVIDIA RTX 4050 Laptop GPU with 6GB VRAM, SM 8.9, CUDA 12.6.",
    "What is the NPU?": "AMD XDNA 2 NPU provides 50 INT8 TOPS on the Ryzen AI 9 HX 370 processor.",
}

# Map questions to short keys for indexing
q_to_key = {q: f"q{i}" for i, q in enumerate(knowledge_base)}
key_to_q = {v: k for k, v in q_to_key.items()}


def run_demo():
    from plato_training.tutor_judge import TutorJudge, word_similarity, _levenshtein
    from plato_training.semantic_matcher import SemanticMatcher

    judge = TutorJudge()
    matcher = SemanticMatcher(threshold=0.4)

    # Index knowledge into semantic matcher
    for question, answer in knowledge_base.items():
        key = q_to_key[question]
        matcher.add(key, f"{question} {answer}")

    semantic_available = matcher.is_semantic

    print("=" * 72)
    print("  SEMANTIC TUTOR DEMO — PLATO Knowledge Q&A")
    print("  5-Layer Cascade: exact → bitvector → semantic → domain → deadband")
    print("=" * 72)
    print(f"\nIndexed {len(knowledge_base)} knowledge entries")
    print(f"Semantic model (Model2Vec+FAISS): {'LOADED ✓' if semantic_available else 'NOT AVAILABLE (keyword fallback)'}")
    print(f"\nRunning demo queries...\n")

    # Demo queries designed to hit different layers
    demo_queries = [
        ("What is PLATO?", "exact match"),
        ("What is Plato?", "case difference"),
        ("what is platow", "typo"),
        ("explain the tile system", "paraphrase"),
        ("how many tests", "partial"),
        ("tell me about spline compression", "paraphrase"),
        ("NPU specs", "abbreviations"),
        ("what is deadband rs", "partial match"),
        ("fleet status", "subset"),
        ("how does collective learning work", "paraphrase"),
        ("GPU hardware info", "abbreviated"),
        ("explain spectral conservation", "paraphrase"),
    ]

    stats = {"exact": 0, "bitvector": 0, "semantic": 0, "keyword": 0, "none": 0}

    for query, description in demo_queries:
        print(f"  Q: \"{query}\"  [{description}]")

        # Layer 1: Exact match
        if query in knowledge_base:
            print(f"  ✅ A (exact): {knowledge_base[query]}")
            stats["exact"] += 1
            print()
            continue

        # Layer 2: Bitvector via TutorJudge (fuzzy pattern match)
        patterns = list(knowledge_base.keys())
        judge_result = judge.judge(query, correct_patterns=patterns)

        if judge_result["correct"]:
            matched_pattern = judge_result.get("matched_pattern", "")
            method = judge_result["method"]
            sim = judge_result["similarity"]
            answer = knowledge_base.get(matched_pattern, "No answer found")
            print(f"  ✅ A ({method}, sim={sim:.2f}): {answer}")
            stats[method] = stats.get(method, 0) + 1
            print()
            continue

        # Layer 3: Semantic match via Model2Vec+FAISS
        sem_result = matcher.match(query)
        if sem_result is not None:
            key, score, text = sem_result
            matched_q = key_to_q.get(key, key)
            answer = knowledge_base.get(matched_q, text)
            print(f"  ✅ A (semantic, score={score:.2f}): {answer}")
            stats["semantic"] += 1
            print()
            continue

        # Layer 4: Keyword fallback
        kw_result = matcher.keyword_fallback(query, {q_to_key[q]: f"{q} {a}" for q, a in knowledge_base.items()})
        if kw_result is not None:
            key, score, text = kw_result
            matched_q = key_to_q.get(key, key)
            answer = knowledge_base.get(matched_q, text)
            print(f"  ⚠️  A (keyword fallback, score={score:.2f}): {answer}")
            stats["keyword"] += 1
            print()
            continue

        # Layer 5: Deadband — no match above threshold
        print(f"  ❌ No match found (below all thresholds)")
        stats["none"] += 1
        print()

    # Summary
    print("=" * 72)
    print("  CASCADE SUMMARY")
    print("=" * 72)
    total = len(demo_queries)
    matched = total - stats["none"]
    print(f"  Total queries:    {total}")
    print(f"  Matched:          {matched}/{total} ({matched/total*100:.0f}%)")
    print(f"  ---")
    for layer in ["exact", "bitvector", "concept", "semantic", "keyword", "none"]:
        count = stats.get(layer, 0)
        if count > 0:
            print(f"  {layer:15s}: {count}")
    print()
    print("  The cascade catches:")
    print("  - Exact matches directly (Layer 1)")
    print("  - Typos via bitvector fingerprinting (Layer 2, TUTOR 1965)")
    print("  - Paraphrases via semantic embeddings (Layer 3, Model2Vec 2024)")
    print("  - Partial matches via keyword overlap (Layer 4, fallback)")
    print("  - Deadband rejects anything below all thresholds (Layer 5)")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
