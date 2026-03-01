#!/usr/bin/env python3
"""
Run the evaluator outer loop.

Automatically discovers the best evaluation metric for index selection
evolution by iteratively proposing evaluators and checking their correlation
with actual query latency (ground truth).

Usage:
    # Default settings (TPC-H, 10 iterations)
    python -m outer_loop.run_outer_loop

    # With custom config
    python -m outer_loop.run_outer_loop --config outer_loop/config_outer_loop.yaml

    # Override settings
    python -m outer_loop.run_outer_loop --benchmark tpcds --iterations 20

    # Skip ground truth measurement (use cached)
    python -m outer_loop.run_outer_loop --ground-truth-cache path/to/cache.json

    # Force re-measurement of ground truth (ignore cache)
    python -m outer_loop.run_outer_loop --force-ground-truth
"""

import argparse
import os
import sys

# Ensure we can import from the outer_loop package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_OPENEVOLVE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, INDEX_OPENEVOLVE_DIR)

from outer_loop.config import OuterLoopConfig
from outer_loop.outer_loop import OuterLoop


def main():
    parser = argparse.ArgumentParser(
        description="Run the evaluator outer loop to discover optimal evaluation metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        help="Config YAML file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["tpch", "tpcds", "job"],
        help="Override benchmark (default: tpch)",
    )
    parser.add_argument(
        "--iterations", type=int,
        help="Override max iterations (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory",
    )
    parser.add_argument(
        "--force-ground-truth",
        action="store_true",
        help="Force re-measurement of ground truth (ignore cache)",
    )
    parser.add_argument(
        "--ground-truth-cache",
        help="Path to pre-computed ground truth cache JSON",
    )
    parser.add_argument(
        "--model",
        help="Override LLM model (default: gemini-2.5-pro)",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = OuterLoopConfig.from_yaml(args.config)
    else:
        config = OuterLoopConfig()

    # Apply overrides
    if args.benchmark:
        config.benchmark = args.benchmark
    if args.iterations:
        config.max_iterations = args.iterations
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.ground_truth_cache:
        config.ground_truth.cache_file = args.ground_truth_cache
    if args.model:
        config.llm.model = args.model

    # Resolve paths relative to index_openevolve/
    config.resolve_paths(INDEX_OPENEVOLVE_DIR)

    # Run outer loop
    loop = OuterLoop(config)
    if args.force_ground_truth:
        loop.ground_truth.measure(loop.corpus, force=True)
    best = loop.run()

    if best and not best.error:
        print(f"\nDone. Best evaluator saved to: {config.output_dir}/best_evaluator.py")
        print(f"\nTo use with evolution:")
        print(f"  python openevolve-run.py initial_program.py "
              f"{config.output_dir}/best_evaluator.py --config config.yaml")
        return 0
    else:
        print("\nOuter loop did not find a satisfactory evaluator.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
