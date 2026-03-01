#!/usr/bin/env python3
"""
Test the outer loop with real data from harald-findings.

Uses pre-computed ground truth (TPC-H interleaved latency) and
pre-computed cost scores to demonstrate the full pipeline without
needing a live database or LLM API key.

Usage:
    cd index_openevolve
    python outer_loop/test_outer_loop.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_OPENEVOLVE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, INDEX_OPENEVOLVE_DIR)

from outer_loop.discrepancy_analyzer import DiscrepancyAnalyzer
from outer_loop.strategy import EvaluationStrategy, StrategyHistory, NoiseResult

# ============================================================
# Real data from harald-findings (TPC-H, interleaved evaluation)
# Source: interleaved/tpc-h/EVOLVED_VS_BASELINE_ANALYSIS.md
#         interleaved/tpc-ds/ALL_VERSIONS_METRICS_COMPARISON.md
# ============================================================

# Ground truth: actual TPC-H latency (seconds) from interleaved protocol
GROUND_TRUTH_TPCH = {
    "initial_program_autoadmin": 4113.80,
    "initial_program_extend": 3941.44,
    "initial_program_db2advis": 4246.88,
    "initial_program_anytime": 4160.25,
}

# What the cost-based evaluator would return (evaluator.py scores)
# Score = 0.80 * cost_reduction + 0.20
COST_SCORES_TPCH = {
    "initial_program_autoadmin": 0.80 * 0.468 + 0.20,  # = 0.5744
    "initial_program_extend": 0.80 * 0.426 + 0.20,     # = 0.5408
    "initial_program_db2advis": 0.80 * 0.497 + 0.20,   # = 0.5976
    "initial_program_anytime": 0.80 * 0.503 + 0.20,    # = 0.6024
}

# What a raw latency evaluator would return (combined_score = -latency_seconds)
LATENCY_SCORES_TPCH = {
    k: -v for k, v in GROUND_TRUTH_TPCH.items()
}

# Ground truth: TPC-DS latency (seconds)
GROUND_TRUTH_TPCDS = {
    "initial_program_autoadmin": 40183.27,
    "initial_program_extend": 39790.90,
    "initial_program_db2advis": 45822.92,
    "initial_program_anytime": 40239.64,
}

# Cost scores for TPC-DS
COST_SCORES_TPCDS = {
    "initial_program_autoadmin": 0.80 * 0.488 + 0.20,  # = 0.5904
    "initial_program_extend": 0.80 * 0.499 + 0.20,     # = 0.5992
    "initial_program_db2advis": 0.80 * 0.350 + 0.20,   # = 0.4800
    "initial_program_anytime": 0.80 * 0.403 + 0.20,    # = 0.5224
}


def test_discrepancy_analysis():
    """Run the discrepancy analyzer with real data."""
    analyzer = DiscrepancyAnalyzer()

    print("=" * 70)
    print("TEST 1: Cost-based proxy vs Ground Truth (TPC-H)")
    print("=" * 70)
    print()

    report = analyzer.analyze(
        proxy_scores=COST_SCORES_TPCH,
        ground_truth_latencies=GROUND_TRUTH_TPCH,
        noise_std_pct=0.0,  # Cost is deterministic
    )
    print(report.text)

    print()
    print(f"\n{'=' * 70}")
    print(f"TEST 2: Cost-based proxy vs Ground Truth (TPC-DS)")
    print(f"{'=' * 70}")
    print()

    report_ds = analyzer.analyze(
        proxy_scores=COST_SCORES_TPCDS,
        ground_truth_latencies=GROUND_TRUTH_TPCDS,
        noise_std_pct=0.0,
    )
    print(report_ds.text)

    print()
    print(f"\n{'=' * 70}")
    print(f"TEST 3: Latency-based proxy vs Ground Truth (TPC-H)")
    print(f"{'=' * 70}")
    print()

    # Simulate latency evaluator with some noise
    report_lat = analyzer.analyze(
        proxy_scores=LATENCY_SCORES_TPCH,
        ground_truth_latencies=GROUND_TRUTH_TPCH,
        noise_std_pct=0.003,  # 0.3% noise with full protocol
    )
    print(report_lat.text)

    return report, report_ds, report_lat


def test_full_history_simulation():
    """Simulate the full outer loop journey with real data."""
    analyzer = DiscrepancyAnalyzer()
    history = StrategyHistory()
    history.ground_truth_scores = GROUND_TRUTH_TPCH

    print(f"\n{'=' * 70}")
    print("SIMULATED OUTER LOOP JOURNEY (TPC-H)")
    print(f"{'=' * 70}")

    # --- Iteration 0: Cost-only evaluator ---
    print("\n--- Iteration 0: Cost-only evaluator (seed) ---")
    s0 = EvaluationStrategy(
        version=0,
        evaluator_code="<evaluator.py>",
        rationale="Cost-based evaluation using PostgreSQL optimizer estimates",
    )
    s0.proxy_scores = COST_SCORES_TPCH
    s0.noise_level = 0.0

    report0 = analyzer.analyze(COST_SCORES_TPCH, GROUND_TRUTH_TPCH, 0.0)
    s0.ranking_agreement = report0.spearman
    s0.discrepancy_report = report0.text
    history.add(s0)
    print(f"  Spearman: {report0.spearman:.4f}")
    print(f"  Pairwise agreement: {report0.pairwise_agreement:.1%}")
    print(f"  Noise: 0.0% (deterministic)")
    print(f"  VERDICT: {'PASS' if report0.spearman > 0.9 else 'FAIL — poor correlation'}")

    # --- Iteration 1: Raw latency (noisy) ---
    print("\n--- Iteration 1: Raw latency evaluator (LLM proposal) ---")
    # Simulate noisy latency (17% noise as documented in findings)
    import random
    random.seed(42)
    noisy_latency_scores = {}
    for name, lat in GROUND_TRUTH_TPCH.items():
        # 17% noise = ranking might flip
        noise_factor = 1.0 + random.gauss(0, 0.17)
        noisy_latency_scores[name] = -(lat * noise_factor)

    s1 = EvaluationStrategy(
        version=1,
        evaluator_code="<latency_evaluator.py>",
        rationale="Switch to actual query latency (EXPLAIN ANALYZE), single run per query",
    )
    s1.proxy_scores = noisy_latency_scores

    # Simulate noise measurement (3 repeated runs)
    noise_runs = [-(GROUND_TRUTH_TPCH["initial_program_autoadmin"] * (1 + random.gauss(0, 0.17)))
                  for _ in range(3)]
    noise1 = NoiseResult(scores=noise_runs)
    s1.noise_level = noise1.std_pct

    report1 = analyzer.analyze(noisy_latency_scores, GROUND_TRUTH_TPCH, noise1.std_pct)
    s1.ranking_agreement = report1.spearman
    s1.discrepancy_report = report1.text
    history.add(s1)
    print(f"  Spearman: {report1.spearman:.4f}")
    print(f"  Pairwise agreement: {report1.pairwise_agreement:.1%}")
    print(f"  Noise: {noise1.std_pct:.1%}")
    print(f"  VERDICT: {'PASS' if report1.spearman > 0.9 and noise1.std_pct < 0.05 else 'FAIL — high noise'}")

    # --- Iteration 2: Interleaved warmup latency ---
    print("\n--- Iteration 2: Interleaved warmup + 1 run (LLM proposal) ---")
    # ~5% noise with interleaved warmup
    noisy_interleaved = {}
    for name, lat in GROUND_TRUTH_TPCH.items():
        noise_factor = 1.0 + random.gauss(0, 0.05)
        noisy_interleaved[name] = -(lat * noise_factor)

    s2 = EvaluationStrategy(
        version=2,
        evaluator_code="<latency_interleaved.py>",
        rationale="Interleaved warmup per query to prevent cache eviction between warmup and measurement",
    )
    s2.proxy_scores = noisy_interleaved

    noise_runs2 = [-(GROUND_TRUTH_TPCH["initial_program_autoadmin"] * (1 + random.gauss(0, 0.05)))
                   for _ in range(3)]
    noise2 = NoiseResult(scores=noise_runs2)
    s2.noise_level = noise2.std_pct

    report2 = analyzer.analyze(noisy_interleaved, GROUND_TRUTH_TPCH, noise2.std_pct)
    s2.ranking_agreement = report2.spearman
    s2.discrepancy_report = report2.text
    history.add(s2)
    print(f"  Spearman: {report2.spearman:.4f}")
    print(f"  Pairwise agreement: {report2.pairwise_agreement:.1%}")
    print(f"  Noise: {noise2.std_pct:.1%}")
    print(f"  VERDICT: {'PASS' if report2.spearman > 0.9 and noise2.std_pct < 0.05 else 'FAIL — noise still too high'}")

    # --- Iteration 3: Full protocol (pg_prewarm + bg suppression + 3 runs) ---
    print("\n--- Iteration 3: Full protocol (pg_prewarm + bg suppress + 3 runs) ---")
    # 0.3% noise with full protocol
    precise_scores = {}
    for name, lat in GROUND_TRUTH_TPCH.items():
        noise_factor = 1.0 + random.gauss(0, 0.003)
        precise_scores[name] = -(lat * noise_factor)

    s3 = EvaluationStrategy(
        version=3,
        evaluator_code="<latency_interleaved_full.py>",
        rationale="Add pg_prewarm to pin data in shared_buffers, CHECKPOINT + disable autovacuum, median of 3 runs",
    )
    s3.proxy_scores = precise_scores

    noise_runs3 = [-(GROUND_TRUTH_TPCH["initial_program_autoadmin"] * (1 + random.gauss(0, 0.003)))
                   for _ in range(3)]
    noise3 = NoiseResult(scores=noise_runs3)
    s3.noise_level = noise3.std_pct

    report3 = analyzer.analyze(precise_scores, GROUND_TRUTH_TPCH, noise3.std_pct)
    s3.ranking_agreement = report3.spearman
    s3.discrepancy_report = report3.text
    history.add(s3)
    print(f"  Spearman: {report3.spearman:.4f}")
    print(f"  Pairwise agreement: {report3.pairwise_agreement:.1%}")
    print(f"  Noise: {noise3.std_pct:.1%}")
    converged = report3.spearman >= 0.9 and noise3.std_pct <= 0.05
    print(f"  VERDICT: {'CONVERGED!' if converged else 'FAIL'}")

    # Summary
    print(f"\n{'=' * 70}")
    print("STRATEGY HISTORY SUMMARY")
    print(f"{'=' * 70}")
    print(history.history_summary())

    best = history.best_strategy()
    print(f"\nBest strategy: v{best.version} — {best.rationale}")
    print(f"  Spearman: {best.ranking_agreement:.4f}, Noise: {best.noise_level:.1%}")

    # Save history
    output_dir = os.path.join(INDEX_OPENEVOLVE_DIR, "outer_loop_outputs_test")
    os.makedirs(output_dir, exist_ok=True)
    history.save(os.path.join(output_dir, "history.json"))
    print(f"\nHistory saved to {output_dir}/history.json")

    return history


def create_ground_truth_cache():
    """Create a ground truth cache file from findings data."""
    output_dir = os.path.join(INDEX_OPENEVOLVE_DIR, "outer_loop_outputs")
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "ground_truth_cache.json")

    with open(cache_path, "w") as f:
        json.dump(GROUND_TRUTH_TPCH, f, indent=2)
    print(f"Ground truth cache written to {cache_path}")
    return cache_path


if __name__ == "__main__":
    # Test 1-3: Discrepancy analysis with real data
    cost_report, ds_report, lat_report = test_discrepancy_analysis()

    # Test 4: Simulate full outer loop journey
    history = test_full_history_simulation()

    # Create ground truth cache for future use
    create_ground_truth_cache()
