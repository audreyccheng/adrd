"""
Evaluator: Fitness function for index selection algorithms
============================================================
Evaluates evolved index selection strategies on benchmark workloads.

Supported benchmarks (via BENCHMARK env var):
- tpch (default): TPC-H workload, 18 queries, benchbase database (SF=1)
- tpcds: TPC-DS workload, 79 queries, benchbase_tpcds database (SF=1)
- job: JOB (IMDB) workload, 33 query templates, benchbase_job database
- dsb: DSB workload, 53 queries, benchbase_tpcds database (uses TPC-DS data)
- tpch_sf10: TPC-H SF=10, 18 queries, benchbase_tpch_sf10 database (13GB, 5GB budget)
- tpcds_sf10: TPC-DS SF=10, 79 queries, benchbase_tpcds_sf10 database (22GB, 5GB budget)
- all: Run tpch, tpcds, job benchmarks in parallel, aggregate scores

Scoring Philosophy:
- PRIMARY: Query cost reduction (frequency-weighted by workload)
- SECONDARY: HARD constraint enforcement (zero score if violated)
- NOT SCORED: Selection time (runs once, indexes used millions of times)

Metrics:
- query_cost_reduction: % reduction in query execution cost (0.0 to 1.0)
- constraint_score: 1.0 if valid, 0.0 if ANY constraint violated (HARD)
- storage_used_mb: Storage consumed by selected indexes
- num_indexes: Number of indexes selected
- selection_time: Time to run selection algorithm (tracked but not scored)
- combined_score: 0.0 if constraints violated, else 0.80 × cost_reduction + 0.20
"""

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


# Benchmark-specific constraints (must match initial_program.py)
# Storage = half dataset size (paper default)
BENCHMARK_CONSTRAINTS = {
    # SF=1 benchmarks (default)
    "tpch": {"budget_mb": 500.0, "max_indexes": 15},
    "tpcds": {"budget_mb": 500.0, "max_indexes": 15},
    "job": {"budget_mb": 2000.0, "max_indexes": 15},
    "dsb": {"budget_mb": 500.0, "max_indexes": 15},  # Uses TPC-DS data, same constraints
    "tpch_skew": {"budget_mb": 500.0, "max_indexes": 15},  # TPC-H with Zipf skew
    # SF=10 benchmarks (larger scale)
    "tpch_sf10": {"budget_mb": 5000.0, "max_indexes": 15},   # 13GB DB, 5GB budget
    "tpcds_sf10": {"budget_mb": 5000.0, "max_indexes": 15},  # 22GB DB, 5GB budget
}

ALL_BENCHMARKS = ["tpch", "tpcds", "job"]


class TimeoutError(Exception):
    pass


def run_with_timeout(program_path, timeout_seconds=600, benchmark="tpch"):
    """
    Run the index selection program in a separate process with timeout.
    
    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds
        benchmark: Benchmark to evaluate on ("tpch" or "tpcds")
        
    Returns:
        Tuple of (selected_indexes, selection_time, baseline_cost, optimized_cost)
    """
    # Get the directory containing this evaluator
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(eval_dir, "deps", "Index_EAB")
    
    # Resolve program_path to absolute path
    if not os.path.isabs(program_path):
        program_path = os.path.abspath(program_path)
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import os
import pickle
import traceback

# Set benchmark environment variable
os.environ['BENCHMARK'] = '{benchmark}'

# Change to project root directory so relative paths work
os.chdir('{project_root}')

# Add paths
sys.path.insert(0, '{project_root}')
sys.path.insert(0, os.path.dirname('{program_path}'))

print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")
print(f"Project root: {project_root}")
print(f"Working directory: {{os.getcwd()}}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the index selection
    print("Calling run_index_selection()...")
    selected_indexes, selection_time, baseline_cost, optimized_cost = program.run_index_selection()
    print(f"Selection completed: {{len(selected_indexes)}} indexes, time={{selection_time:.2f}}s")
    
    # Convert indexes to serializable format
    indexes_data = []
    for idx in selected_indexes:
        indexes_data.append({{
            'columns': [col.name for col in idx.columns],
            'table': str(idx.table()),
            'estimated_size': idx.estimated_size
        }})
    
    # Save results
    results = {{
        'indexes': indexes_data,
        'selection_time': selection_time,
        'baseline_cost': baseline_cost,
        'optimized_cost': optimized_cost,
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
            
            # Print output for debugging
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")
            
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")
            
            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                
                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")
                
                return (
                    results["indexes"],
                    results["selection_time"],
                    results["baseline_cost"],
                    results["optimized_cost"]
                )
            else:
                raise RuntimeError("Results file not found")
        
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path, benchmark="tpch"):
    """
    Evaluate an index selection program.
    
    The program must have a run_index_selection() function that returns:
        (selected_indexes, selection_time, baseline_cost, optimized_cost)
    
    Args:
        program_path: Path to the program file to evaluate
        benchmark: Benchmark to evaluate on ("tpch" or "tpcds")
        
    Returns:
        Dictionary with evaluation metrics:
        - query_cost_reduction: % reduction in query cost (0.0 to 1.0)
        - constraint_score: Soft penalty for violations (0.0 to 1.0)
        - storage_used_mb: Storage used by selected indexes
        - num_indexes: Number of indexes selected
        - selection_time: Time taken for selection (tracked but not scored)
        - combined_score: 0.80 × cost_reduction + 0.20 × constraint_score
    """
    try:
        start_time = time.time()
        
        # Run the program with timeout
        indexes_data, selection_time, baseline_cost, optimized_cost = run_with_timeout(
            program_path, timeout_seconds=600, benchmark=benchmark  # 10 minute timeout
        )
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        num_indexes = len(indexes_data)
        
        # Calculate storage used
        storage_used_mb = sum(idx['estimated_size'] for idx in indexes_data) / (1024 * 1024)
        
        # === PRIMARY METRIC: Cost Reduction (already frequency-weighted) ===
        if baseline_cost > 0:
            cost_reduction = (baseline_cost - optimized_cost) / baseline_cost
            cost_reduction = max(0.0, min(1.0, cost_reduction))  # Clamp to [0, 1]
        else:
            cost_reduction = 0.0
        
        # === CONSTRAINT SATISFACTION: HARD enforcement ===
        # Use benchmark-specific constraints
        constraints = BENCHMARK_CONSTRAINTS.get(benchmark, {"budget_mb": 500.0, "max_indexes": 15})
        BUDGET_MB = constraints["budget_mb"]
        MAX_INDEXES = constraints["max_indexes"]
        
        # Check for constraint violations (HARD constraints)
        storage_violated = storage_used_mb > BUDGET_MB
        count_violated = num_indexes > MAX_INDEXES
        
        # Log violations
        if storage_violated:
            print(f"❌ HARD CONSTRAINT VIOLATED: Storage {storage_used_mb:.2f} MB > {BUDGET_MB} MB")
        if count_violated:
            print(f"❌ HARD CONSTRAINT VIOLATED: {num_indexes} indexes > {MAX_INDEXES}")
        
        # Constraint score: 1.0 if valid, 0.0 if ANY constraint violated
        constraint_score = 0.0 if (storage_violated or count_violated) else 1.0
        
        # === COMBINED SCORE: Zero score for constraint violations ===
        # Constraint violations result in ZERO score (hard enforcement)
        # This prevents evolution from "cheating" by violating constraints
        if constraint_score == 0.0:
            combined_score = 0.0
            print(f"⚠️  COMBINED SCORE = 0.0 due to constraint violation")
        else:
            # 80% weight on query performance (the actual goal)
            # 20% weight reserved for future soft constraints
            combined_score = 0.80 * cost_reduction + 0.20 * constraint_score
        
        print(f"✅ Evaluation: constraint={constraint_score:.3f}, cost_reduction={cost_reduction:.3f}, "
              f"time={selection_time:.2f}s, score={combined_score:.3f}")
        
        return {
            "query_cost_reduction": float(cost_reduction),
            "constraint_score": float(constraint_score),
            "storage_used_mb": float(storage_used_mb),
            "num_indexes": float(num_indexes),
            "selection_time": float(selection_time),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }
    
    except TimeoutError as e:
        print(f"❌ Evaluation timed out: {e}")
        return {
            "query_cost_reduction": 0.0,
            "constraint_score": 0.0,
            "storage_used_mb": 0.0,
            "num_indexes": 0.0,
            "selection_time": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }
    
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "query_cost_reduction": 0.0,
            "constraint_score": 0.0,
            "storage_used_mb": 0.0,
            "num_indexes": 0.0,
            "selection_time": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


# Cascade evaluation functions for faster filtering
def evaluate_stage1(program_path, benchmark="tpch"):
    """
    Stage 1: Quick validation check (30 second timeout).
    Filters out programs that fail fast or violate basic constraints.
    Uses simplified scoring to quickly identify promising programs.
    """
    try:
        # Shorter timeout for quick filter
        indexes_data, selection_time, baseline_cost, optimized_cost = run_with_timeout(
            program_path, timeout_seconds=30, benchmark=benchmark
        )
        
        # Quick metrics
        num_indexes = len(indexes_data)
        storage_used_mb = sum(idx['estimated_size'] for idx in indexes_data) / (1024 * 1024)
        
        # Cost reduction
        if baseline_cost > 0:
            cost_reduction = max(0.0, (baseline_cost - optimized_cost) / baseline_cost)
        else:
            cost_reduction = 0.0
        
        # HARD constraint score (same logic as full evaluation)
        constraints = BENCHMARK_CONSTRAINTS.get(benchmark, {"budget_mb": 500.0, "max_indexes": 15})
        storage_violated = storage_used_mb > constraints["budget_mb"]
        count_violated = num_indexes > constraints["max_indexes"]
        constraint_score = 0.0 if (storage_violated or count_violated) else 1.0
        
        # Combined score with hard constraint enforcement
        combined_score = 0.0 if constraint_score == 0.0 else (0.80 * cost_reduction + 0.20 * constraint_score)
        
        return {
            "constraint_score": float(constraint_score),
            "combined_score": float(combined_score),
            "query_cost_reduction": float(cost_reduction),
        }
    
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return {
            "constraint_score": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage2(program_path, benchmark="tpch"):
    """
    Stage 2: Full evaluation.
    Only called on programs that pass stage 1.
    """
    return evaluate(program_path, benchmark=benchmark)


def _evaluate_single_benchmark(args):
    """Helper for parallel execution - evaluates a single benchmark."""
    program_path, benchmark = args
    try:
        return benchmark, evaluate(program_path, benchmark=benchmark)
    except Exception as e:
        return benchmark, {
            "query_cost_reduction": 0.0,
            "constraint_score": 0.0,
            "storage_used_mb": 0.0,
            "num_indexes": 0.0,
            "selection_time": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_all_benchmarks(program_path, parallel=True):
    """
    Evaluate program on all benchmarks (tpch, tpcds, job).
    
    Args:
        program_path: Path to the program file to evaluate
        parallel: If True, run benchmarks in parallel (default: True)
        
    Returns:
        Dictionary with aggregated metrics and per-benchmark breakdown
    """
    start_time = time.time()
    results_by_benchmark = {}
    
    if parallel:
        # Run all benchmarks in parallel using ProcessPoolExecutor
        print(f"🚀 Running {len(ALL_BENCHMARKS)} benchmarks in parallel...")
        with ProcessPoolExecutor(max_workers=len(ALL_BENCHMARKS)) as executor:
            futures = {
                executor.submit(_evaluate_single_benchmark, (program_path, bm)): bm 
                for bm in ALL_BENCHMARKS
            }
            for future in as_completed(futures):
                benchmark, result = future.result()
                results_by_benchmark[benchmark] = result
                print(f"  ✓ {benchmark.upper()}: score={result.get('combined_score', 0):.3f}")
    else:
        # Run sequentially
        print(f"🔄 Running {len(ALL_BENCHMARKS)} benchmarks sequentially...")
        for benchmark in ALL_BENCHMARKS:
            print(f"  Running {benchmark.upper()}...")
            results_by_benchmark[benchmark] = evaluate(program_path, benchmark=benchmark)
            print(f"  ✓ {benchmark.upper()}: score={results_by_benchmark[benchmark].get('combined_score', 0):.3f}")
    
    total_time = time.time() - start_time
    
    # Aggregate scores (simple average)
    valid_scores = [r["combined_score"] for r in results_by_benchmark.values() if "error" not in r]
    valid_cost_reductions = [r["query_cost_reduction"] for r in results_by_benchmark.values() if "error" not in r]
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        avg_cost_reduction = sum(valid_cost_reductions) / len(valid_cost_reductions)
    else:
        avg_score = 0.0
        avg_cost_reduction = 0.0
    
    # Check if any benchmark had constraint violations
    all_constraints_satisfied = all(
        r.get("constraint_score", 0) == 1.0 
        for r in results_by_benchmark.values() 
        if "error" not in r
    )
    
    print(f"\n📊 Aggregate: avg_score={avg_score:.3f}, avg_cost_reduction={avg_cost_reduction:.3f}, time={total_time:.1f}s")
    
    return {
        # Aggregated metrics
        "combined_score": float(avg_score),
        "query_cost_reduction": float(avg_cost_reduction),
        "constraint_score": 1.0 if all_constraints_satisfied else 0.0,
        "eval_time": float(total_time),
        # Per-benchmark breakdown
        "tpch": results_by_benchmark.get("tpch", {}),
        "tpcds": results_by_benchmark.get("tpcds", {}),
        "job": results_by_benchmark.get("job", {}),
    }


def main():
    """Command-line interface for testing evaluator."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate an index selection program.")
    parser.add_argument("program_path", help="Path to the program file to evaluate.")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Evaluation stage (for cascade)")
    parser.add_argument("--benchmark", choices=["tpch", "tpcds", "job", "dsb", "tpch_skew", "tpch_sf10", "tpcds_sf10", "all"], default="tpch",
                       help="Benchmark workload to evaluate on (default: tpch, 'all' runs tpch/tpcds/job)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run benchmarks sequentially instead of parallel (only with --benchmark all)")
    args = parser.parse_args()
    
    if args.benchmark == "all":
        print(f"Evaluating on ALL benchmarks ({'sequential' if args.sequential else 'parallel'})...")
        results = evaluate_all_benchmarks(args.program_path, parallel=not args.sequential)
    elif args.stage == 1:
        print(f"Evaluating on benchmark: {args.benchmark.upper()} (Stage 1)")
        results = evaluate_stage1(args.program_path, benchmark=args.benchmark)
    elif args.stage == 2:
        print(f"Evaluating on benchmark: {args.benchmark.upper()} (Stage 2)")
        results = evaluate_stage2(args.program_path, benchmark=args.benchmark)
    else:
        print(f"Evaluating on benchmark: {args.benchmark.upper()}")
        results = evaluate(args.program_path, benchmark=args.benchmark)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

