#!/usr/bin/env python3
"""
Latency Evaluator: Measure actual query execution time for index selection algorithms.

This script evaluates algorithms using REAL query execution latency AND collects
per-query cost estimates for training a cost→latency prediction model.

Based on paper.md experimental setup:
- SF=1 databases (1GB → 500MB budget)
- Warm cache (1 run per query, sequential execution)
- Frequency-weighted latency
- Index creation time recorded separately

Output: Per-query data with (cost_estimate, actual_latency) pairs for model training.

Usage:
    python latency_evaluator.py                        # Run all baselines on all benchmarks
    python latency_evaluator.py --algorithm extend     # Run specific algorithm
    python latency_evaluator.py --benchmark tpch       # Run specific benchmark
    python latency_evaluator.py --evolved explore_extend_1215  # Run evolved program
"""

import argparse
import configparser
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

# Project root for imports
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", "Index_EAB")
sys.path.insert(0, PROJECT_ROOT)

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index
# Note: CostEvaluation removed - requires torch which is heavy and not needed for latency eval

# Baseline algorithms to evaluate
BASELINES = {
    "initial_program": "initial_programs/initial_program.py",
    "autoadmin": "initial_programs/initial_program_autoadmin.py",
    "anytime": "initial_programs/initial_program_anytime.py",
    "extend": "initial_programs/initial_program_extend.py",
    "db2advis": "initial_programs/initial_program_db2advis.py",
}

# Top 5 evolved programs (selected for diversity + performance)
EVOLVED_PROGRAMS = {
    "best_explore_extend_1215": "initial_programs/best_explore_extend_1215.py",
    "best_tpch_v3_extend_evolved": "initial_programs/best_tpch_v3_extend_evolved.py",
}

# Benchmark configurations (SF=1, per paper)
BENCHMARK_CONFIG = {
    "tpch": {
        "db_name": "benchbase",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpch.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
        "num_queries": 18,
        "budget_mb": 500,
    },
    "tpcds": {
        "db_name": "benchbase_tpcds",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
        "num_queries": 79,
        "budget_mb": 500,
    },
    "job": {
        "db_name": "benchbase_job",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_job.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/job_work_temp_multi_freq.json",
        "num_queries": 33,
        "budget_mb": 2000,
    },
}


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def load_workload(benchmark: str) -> Tuple[Workload, PostgresDatabaseConnector, list]:
    """Load workload and create database connector."""
    config = BENCHMARK_CONFIG[benchmark]
    
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    _, columns = heu_com.get_columns_from_schema(config["schema_file"])
    
    with open(config["workload_file"], "r") as rf:
        work_list = json.load(rf)
    
    all_queries = work_list[0]
    
    if benchmark == "job" and config["num_queries"] < len(all_queries):
        sorted_queries = sorted(all_queries, key=lambda x: x[2], reverse=True)
        query_data = sorted_queries[:config["num_queries"]]
    else:
        query_data = all_queries[:config["num_queries"]]
    
    workload = Workload(heu_com.read_row_query(
        query_data, {}, columns,
        type="", varying_frequencies=True, seed=666
    ))
    
    return workload, connector, columns


def get_program_path(algorithm_name: str) -> str:
    """Get the path to a program (baseline or evolved)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if algorithm_name in BASELINES:
        return os.path.join(script_dir, BASELINES[algorithm_name])
    elif algorithm_name in EVOLVED_PROGRAMS:
        return os.path.join(script_dir, EVOLVED_PROGRAMS[algorithm_name])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_selection_algorithm(algorithm_name: str, benchmark: str) -> Tuple[Set[Index], float, float, float]:
    """Run a selection algorithm and return selected indexes."""
    program_path = get_program_path(algorithm_name)
    
    os.environ["BENCHMARK"] = benchmark
    os.environ["FULL_WORKLOAD"] = "true"
    
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    selected_indexes, selection_time, baseline_cost, optimized_cost = program.run_index_selection()
    
    return selected_indexes, selection_time, baseline_cost, optimized_cost


def create_real_indexes(indexes: Set[Index], connector: PostgresDatabaseConnector) -> Tuple[float, List[str]]:
    """Create real B-tree indexes in PostgreSQL."""
    index_names = []
    total_time = 0.0
    
    for idx in indexes:
        table_name = idx.table().name if hasattr(idx.table(), 'name') else str(idx.table())
        col_names = "_".join([c.name for c in idx.columns])
        index_name = f"lat_eval_{table_name}_{col_names}"[:63]
        
        columns_str = ", ".join([c.name for c in idx.columns])
        create_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_name} ({columns_str})'
        
        try:
            start = time.time()
            connector.exec_only(create_sql)
            connector.commit()
            elapsed = time.time() - start
            total_time += elapsed
            index_names.append(index_name)
            log(f"    Created {index_name} in {elapsed:.2f}s")
        except Exception as e:
            log(f"    Warning: Failed to create {index_name}: {e}")
    
    log("    Running ANALYZE...")
    start = time.time()
    connector.exec_only("ANALYZE")
    connector.commit()
    log(f"    ANALYZE completed in {time.time() - start:.2f}s")
    
    return total_time, index_names


def drop_indexes(index_names: List[str], connector: PostgresDatabaseConnector):
    """Drop all created indexes."""
    for name in index_names:
        try:
            connector.exec_only(f'DROP INDEX IF EXISTS "{name}"')
            connector.commit()
        except Exception as e:
            log(f"    Warning: Failed to drop {name}: {e}")


def cleanup_all_test_indexes(connector: PostgresDatabaseConnector):
    """Clean up ALL lat_eval_* indexes (from previous failed runs)."""
    try:
        result = connector.exec_fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'lat_eval_%'",
            one=False
        )
        if result:
            log(f"    ⚠️  Cleaning up {len(result)} leftover indexes from previous runs...")
            for row in result:
                connector.exec_only(f'DROP INDEX IF EXISTS "{row[0]}"')
                connector.commit()
            log(f"    ✅ Cleanup complete")
    except Exception as e:
        log(f"    Warning: Cleanup check failed: {e}")


def measure_query_cost(query, connector: PostgresDatabaseConnector) -> float:
    """Get optimizer cost estimate for a query using EXPLAIN."""
    query_text = query.text
    
    # Handle views in query
    if "create view" in query_text.lower():
        statements = query_text.split(";")
        cost = 0.0
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            if "create view" in stmt.lower():
                connector.exec_only(stmt)
            elif "drop view" in stmt.lower():
                pass
            else:
                explain_sql = f"EXPLAIN (FORMAT JSON) {stmt}"
                try:
                    result = connector.exec_fetch(explain_sql, one=True)
                    cost = result[0][0]["Plan"]["Total Cost"]
                except Exception as e:
                    log(f"      Cost query error: {e}")
                    cost = 0.0
        
        # Cleanup views
        for stmt in statements:
            if "drop view" in stmt.lower():
                try:
                    connector.exec_only(stmt.strip())
                except:
                    pass
        
        return cost
    else:
        explain_sql = f"EXPLAIN (FORMAT JSON) {query_text}"
        try:
            result = connector.exec_fetch(explain_sql, one=True)
            return result[0][0]["Plan"]["Total Cost"]
        except Exception as e:
            log(f"      Cost query error: {e}")
            return 0.0


def measure_query_latency(query, connector: PostgresDatabaseConnector) -> float:
    """Execute a query and return actual execution time in milliseconds."""
    query_text = query.text
    
    if "create view" in query_text.lower():
        statements = query_text.split(";")
        latency_ms = 0.0
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            if "create view" in stmt.lower():
                connector.exec_only(stmt)
            elif "drop view" in stmt.lower():
                pass
            else:
                explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {stmt}"
                try:
                    result = connector.exec_fetch(explain_sql, one=True)
                    latency_ms = result[0][0]["Execution Time"]
                except Exception as e:
                    log(f"      Query error: {e}")
                    latency_ms = 0.0
        
        for stmt in statements:
            if "drop view" in stmt.lower():
                try:
                    connector.exec_only(stmt.strip())
                except:
                    pass
        
        return latency_ms
    else:
        explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {query_text}"
        try:
            result = connector.exec_fetch(explain_sql, one=True)
            return result[0][0]["Execution Time"]
        except Exception as e:
            log(f"      Query error: {e}")
            return 0.0


def warmup_cache(workload: Workload, connector: PostgresDatabaseConnector):
    """
    Warmup cache by running all queries once (discard results).
    
    This ensures consistent warm cache state for fair comparison between
    baseline (no indexes) and optimized (with indexes) measurements.
    """
    log(f"     Warming up cache ({len(workload.queries)} queries)...")
    for i, query in enumerate(workload.queries):
        try:
            measure_query_latency(query, connector)  # Discard results
        except Exception as e:
            pass  # Ignore warmup errors
        if (i + 1) % 10 == 0:
            log(f"       Warmup progress: {i + 1}/{len(workload.queries)}")
    log(f"     Warmup complete")


def measure_workload_metrics(
    workload: Workload, 
    connector: PostgresDatabaseConnector,
    has_indexes: bool,
    algorithm: str,
    benchmark: str,
    num_indexes: int = 0,
    storage_mb: float = 0.0
) -> Tuple[float, float, List[Dict]]:
    """
    Measure cost AND latency for each query in the workload.
    
    Returns:
        total_weighted_cost: Sum of frequency-weighted costs
        total_weighted_latency: Sum of frequency-weighted latencies
        per_query_data: List of per-query measurements for model training
    """
    total_weighted_cost = 0.0
    total_weighted_latency = 0.0
    per_query_data = []
    total_freq = sum(q.frequency for q in workload.queries)
    
    for i, query in enumerate(workload.queries):
        # Measure cost estimate FIRST (doesn't modify state)
        cost = measure_query_cost(query, connector)
        
        # Measure actual latency
        latency_ms = measure_query_latency(query, connector)
        
        weighted_cost = cost * query.frequency
        weighted_latency = latency_ms * query.frequency
        total_weighted_cost += weighted_cost
        total_weighted_latency += weighted_latency
        
        # Save per-query data for model training
        per_query_data.append({
            "benchmark": benchmark,
            "query_id": i,
            "frequency": query.frequency,
            "weight_pct": (query.frequency / total_freq) * 100,
            "cost_estimate": cost,
            "actual_latency_ms": latency_ms,
            "has_indexes": has_indexes,
            "num_indexes": num_indexes if has_indexes else 0,
            "storage_mb": storage_mb if has_indexes else 0.0,
            "algorithm": algorithm,
        })
        
        if (i + 1) % 10 == 0 or i == len(workload.queries) - 1:
            log(f"      Executed {i + 1}/{len(workload.queries)} queries...")
    
    return total_weighted_cost, total_weighted_latency, per_query_data


def evaluate_algorithm_on_benchmark(algorithm_name: str, benchmark: str) -> Dict:
    """Full latency evaluation of one algorithm on one benchmark."""
    log(f"\n{'='*60}")
    log(f"  {algorithm_name.upper()} on {benchmark.upper()}")
    log(f"{'='*60}")
    
    result = {
        "algorithm": algorithm_name,
        "benchmark": benchmark,
        "timestamp": datetime.now().isoformat(),
        "per_query_baseline": [],
        "per_query_optimized": [],
    }
    
    connector = None  # Initialize for finally block
    
    try:
        # Step 1: Run selection algorithm
        log(f"\n  1. Running selection algorithm...")
        selected_indexes, selection_time, baseline_cost, optimized_cost = run_selection_algorithm(
            algorithm_name, benchmark
        )
        
        result["selection_time"] = selection_time
        result["num_indexes"] = len(selected_indexes)
        result["baseline_cost"] = baseline_cost
        result["optimized_cost"] = optimized_cost
        
        if baseline_cost > 0:
            result["cost_reduction"] = (baseline_cost - optimized_cost) / baseline_cost * 100
        else:
            result["cost_reduction"] = 0.0
        
        storage_bytes = sum(idx.estimated_size or 0 for idx in selected_indexes)
        result["storage_mb"] = storage_bytes / (1024 * 1024)
        
        log(f"     Selected {len(selected_indexes)} indexes in {selection_time:.2f}s")
        log(f"     Cost reduction: {result['cost_reduction']:.1f}%")
        log(f"     Storage: {result['storage_mb']:.1f} MB")
        
        # Step 2: Load workload and connect
        log(f"\n  2. Loading workload...")
        workload, connector, columns = load_workload(benchmark)
        log(f"     Loaded {len(workload.queries)} queries")
        
        # Step 2b: Clean up any leftover indexes from previous failed runs
        cleanup_all_test_indexes(connector)
        
        # Step 2c: Run ANALYZE to ensure fresh statistics for fair comparison
        log(f"\n  2c. Running ANALYZE for fresh statistics...")
        analyze_start = time.time()
        connector.exec_only("ANALYZE")
        connector.commit()
        log(f"      ANALYZE completed in {time.time() - analyze_start:.2f}s")
        
        # Step 3: Warmup cache and measure baseline (no indexes)
        log(f"\n  3. Measuring baseline (no indexes) - cost and latency...")
        warmup_cache(workload, connector)  # Warmup for fair comparison
        baseline_cost_total, baseline_latency_total, baseline_per_query = measure_workload_metrics(
            workload, connector,
            has_indexes=False,
            algorithm=algorithm_name,
            benchmark=benchmark
        )
        result["baseline_latency_ms"] = baseline_latency_total
        result["per_query_baseline"] = baseline_per_query
        log(f"     Baseline latency total: {baseline_latency_total/1000:.2f}s")
        log(f"     Baseline cost total: {baseline_cost_total:.0f}")
        
        # Step 4: Create real indexes
        log(f"\n  4. Creating {len(selected_indexes)} real indexes...")
        creation_time, index_names = create_real_indexes(selected_indexes, connector)
        result["index_creation_time"] = creation_time
        log(f"     Total creation time: {creation_time:.2f}s")
        
        # Step 5: Warmup cache and measure optimized (with indexes)
        log(f"\n  5. Measuring optimized (with indexes) - cost and latency...")
        warmup_cache(workload, connector)  # Warmup for fair comparison
        optimized_cost_total, optimized_latency_total, optimized_per_query = measure_workload_metrics(
            workload, connector,
            has_indexes=True,
            algorithm=algorithm_name,
            benchmark=benchmark,
            num_indexes=len(selected_indexes),
            storage_mb=result["storage_mb"]
        )
        result["optimized_latency_ms"] = optimized_latency_total
        result["per_query_optimized"] = optimized_per_query
        log(f"     Optimized latency total: {optimized_latency_total/1000:.2f}s")
        log(f"     Optimized cost total: {optimized_cost_total:.0f}")
        
        # Calculate latency reduction
        if baseline_latency_total > 0:
            result["latency_reduction"] = (baseline_latency_total - optimized_latency_total) / baseline_latency_total * 100
        else:
            result["latency_reduction"] = 0.0
        
        log(f"\n  RESULTS:")
        log(f"     Cost reduction:    {result['cost_reduction']:.1f}%")
        log(f"     Latency reduction: {result['latency_reduction']:.1f}%")
        delta = result['latency_reduction'] - result['cost_reduction']
        log(f"     Correlation delta: {'+' if delta >= 0 else ''}{delta:.1f}%")
        
        result["status"] = "success"
        
    except Exception as e:
        import traceback
        log(f"\n  ERROR: {e}")
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = str(e)
    
    finally:
        # ALWAYS cleanup indexes, even on error
        try:
            if connector is not None:
                log(f"\n  6. Cleaning up indexes...")
                cleanup_all_test_indexes(connector)
                connector.close()
        except Exception as cleanup_error:
            log(f"    Warning: Cleanup failed: {cleanup_error}")
    
    return result


def run_all_evaluations(algorithms: List[str] = None, benchmarks: List[str] = None) -> List[Dict]:
    """Run evaluations for all specified algorithms and benchmarks."""
    if algorithms is None:
        algorithms = list(BASELINES.keys())
    if benchmarks is None:
        benchmarks = list(BENCHMARK_CONFIG.keys())
    
    all_results = []
    all_per_query = []  # Collect all per-query data for model training
    total_start = time.time()
    
    log("\n" + "=" * 70)
    log("  LATENCY EVALUATION - WITH PER-QUERY DATA COLLECTION")
    log("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 70)
    log(f"\n  Algorithms: {', '.join(algorithms)}")
    log(f"  Benchmarks: {', '.join(benchmarks)}")
    log(f"  Total evaluations: {len(algorithms) * len(benchmarks)}")
    
    for algorithm in algorithms:
        for benchmark in benchmarks:
            result = evaluate_algorithm_on_benchmark(algorithm, benchmark)
            all_results.append(result)
            
            # Collect per-query data
            if result["status"] == "success":
                all_per_query.extend(result.get("per_query_baseline", []))
                all_per_query.extend(result.get("per_query_optimized", []))
    
    total_time = time.time() - total_start
    
    # Print summary
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    
    for benchmark in benchmarks:
        log(f"\n  {benchmark.upper()}:")
        log(f"  {'Algorithm':<20} {'Indexes':<8} {'Cost':<10} {'Latency':<10} {'Delta':<10}")
        log(f"  {'-'*58}")
        
        for result in all_results:
            if result["benchmark"] == benchmark and result["status"] == "success":
                delta = result.get("latency_reduction", 0) - result.get("cost_reduction", 0)
                log(f"  {result['algorithm']:<20} "
                    f"{result.get('num_indexes', 0):<8} "
                    f"{result.get('cost_reduction', 0):>6.1f}%   "
                    f"{result.get('latency_reduction', 0):>6.1f}%   "
                    f"{delta:>+5.1f}%")
    
    log(f"\n  Total time: {total_time/60:.1f} minutes")
    log(f"  Per-query samples collected: {len(all_per_query)}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save full results (with per-query embedded)
    output_file = f"latency_results_{timestamp}.json"
    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\n  Full results saved to: {output_file}")
    
    # Save per-query data separately for model training
    per_query_file = f"per_query_data_{timestamp}.json"
    per_query_path = os.path.join(script_dir, per_query_file)
    with open(per_query_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "algorithms": algorithms,
                "benchmarks": benchmarks,
                "total_samples": len(all_per_query),
            },
            "data": all_per_query,
        }, f, indent=2)
    log(f"  Per-query data saved to: {per_query_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate index selection algorithms with actual query latency and per-query data collection."
    )
    parser.add_argument(
        "--algorithm", "-a",
        help="Run specific baseline algorithm"
    )
    parser.add_argument(
        "--evolved", "-e",
        help="Run specific evolved program"
    )
    parser.add_argument(
        "--benchmark", "-b",
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Run specific benchmark (default: all)"
    )
    parser.add_argument(
        "--all-evolved",
        action="store_true",
        help="Run all evolved programs (in addition to baselines)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available algorithms and benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.list:
        log("\nBaseline algorithms:")
        for name, path in BASELINES.items():
            log(f"  - {name}: {path}")
        log("\nEvolved programs:")
        for name, path in EVOLVED_PROGRAMS.items():
            log(f"  - {name}: {path}")
        log("\nBenchmarks:")
        for name, config in BENCHMARK_CONFIG.items():
            log(f"  - {name}: {config['num_queries']} queries, {config['budget_mb']}MB budget")
        return
    
    # Determine which algorithms to run
    algorithms = []
    
    if args.algorithm:
        if args.algorithm not in BASELINES:
            log(f"Error: Unknown baseline '{args.algorithm}'")
            log(f"Available: {', '.join(BASELINES.keys())}")
            return
        algorithms = [args.algorithm]
    elif args.evolved:
        if args.evolved not in EVOLVED_PROGRAMS:
            log(f"Error: Unknown evolved program '{args.evolved}'")
            log(f"Available: {', '.join(EVOLVED_PROGRAMS.keys())}")
            return
        algorithms = [args.evolved]
    elif args.all_evolved:
        algorithms = list(BASELINES.keys()) + list(EVOLVED_PROGRAMS.keys())
    else:
        algorithms = list(BASELINES.keys())
    
    benchmarks = [args.benchmark] if args.benchmark else None
    
    run_all_evaluations(algorithms, benchmarks)


if __name__ == "__main__":
    main()
