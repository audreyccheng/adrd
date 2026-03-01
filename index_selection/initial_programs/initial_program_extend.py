"""
Initial Program: Extend Algorithm for OpenEvolve
=================================================
Based on Extend heuristic (Schlosser, Kossmann, Boissier, ICDE 2019):
- Start with single-column indexes
- Iteratively extend with best prefix expansion
- Uses benefit-to-size ratio for selection

Supports multiple benchmarks via BENCHMARK environment variable:
- BENCHMARK=tpch (default): TPC-H workload, 18 queries, benchbase database
- BENCHMARK=tpcds: TPC-DS workload, 79 queries, benchbase_tpcds database
- BENCHMARK=job: JOB (IMDB) workload, 33 query templates, benchbase_job database

OpenEvolve will mutate this to discover better selection strategies.
"""

import sys
import os
import json
import configparser
from typing import Set, List, Dict, Optional
import copy

sys.path.insert(0, os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB")))

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index


# EVOLVE-BLOCK-START
def select_best_indexes(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int = 5,
    budget_MB: float = 200.0,
    min_improvement: float = 1.001,  # Minimum 0.1% improvement to continue
) -> Set[Index]:
    """
    Extend-style selection: prefix-based expansion with greedy marginal benefit.
    
    Algorithm:
    1. Start with single-column candidates
    2. Greedily add best candidate (by marginal cost reduction)
    3. Try extending existing indexes with additional columns
    4. Continue until budget exhausted or no improvement
    
    Key fix: Use marginal benefit selection to utilize budget effectively.
    
    This function will be evolved by OpenEvolve.
    """
    budget_bytes = budget_MB * 1024 * 1024
    
    if max_indexes == 0 or budget_MB == 0:
        return set()
    
    if not candidate_indexes:
        return set()
    
    # Separate single-column and multi-column candidates
    single_column = [idx for idx in sorted(list(candidate_indexes)) if len(idx.columns) == 1]
    multi_column = [idx for idx in sorted(list(candidate_indexes)) if len(idx.columns) > 1]
    
    # Get extension columns (for extending existing indexes)
    extension_columns = set()
    for idx in candidate_indexes:
        for col in idx.columns:
            extension_columns.add(col)
    
    # Initialize
    current_combination = []
    current_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    initial_cost = current_cost
    
    def get_current_size():
        return sum(idx.estimated_size or 0 for idx in current_combination)
    
    # Phase 1: Greedy selection from all candidates (single + multi column)
    # This ensures we utilize budget even when benefit/size ratios are low
    all_candidates = single_column + multi_column
    used_candidates = set()
    
    improved = True
    while improved and len(current_combination) < max_indexes:
        improved = False
        best_candidate = None
        best_cost = current_cost
        
        for candidate in all_candidates:
            if candidate in used_candidates:
                continue
            if candidate in current_combination:
                continue
            
            # Check budget
            candidate_size = candidate.estimated_size or 0
            if get_current_size() + candidate_size > budget_bytes:
                continue
            
            # Evaluate marginal benefit
            test_combination = current_combination + [candidate]
            cost = cost_evaluation.calculate_cost(workload, set(test_combination), store_size=True)
            
            # Check minimum improvement threshold
            if current_cost > 0 and cost < best_cost:
                improvement_ratio = current_cost / cost
                if improvement_ratio >= min_improvement:
                    best_cost = cost
                    best_candidate = candidate
        
        if best_candidate is not None:
            current_combination.append(best_candidate)
            used_candidates.add(best_candidate)
            current_cost = best_cost
            improved = True
    
    # Phase 2: Try extending existing indexes with additional columns
    # This can find better multi-column indexes than pre-generated ones
    improved = True
    max_extend_iterations = 10  # Prevent infinite loops
    extend_iterations = 0
    
    while improved and extend_iterations < max_extend_iterations:
        improved = False
        extend_iterations += 1
        
        for i, existing_idx in enumerate(current_combination):
            if len(existing_idx.columns) >= 3:  # Max width limit
                continue
            
            for col in sorted(list(extension_columns)):
                if col in existing_idx.columns:
                    continue
                if col.table != existing_idx.table():
                    continue
                
                # Create extended index
                extended_idx = Index(existing_idx.columns + (col,))
                
                # Calculate actual size
                cost_evaluation.estimate_size(extended_idx)
                
                old_size = existing_idx.estimated_size or 0
                new_size = extended_idx.estimated_size or old_size
                size_diff = new_size - old_size
                
                if get_current_size() + size_diff > budget_bytes:
                    continue
                
                # Evaluate
                test_combination = current_combination.copy()
                test_combination[i] = extended_idx
                cost = cost_evaluation.calculate_cost(workload, set(test_combination), store_size=True)
                
                # Check if this is an improvement
                if cost < current_cost:
                    improvement_ratio = current_cost / cost
                    if improvement_ratio >= min_improvement:
                        current_combination[i] = extended_idx
                        current_cost = cost
                        improved = True
                        break  # Restart extension search
            
            if improved:
                break
    
    return set(current_combination)
# EVOLVE-BLOCK-END


def generate_candidates(workload, max_index_width: int = 2):
    """Generate candidate indexes from workload."""
    candidates = set()
    potential_indexes = workload.potential_indexes()
    
    for current_width in range(1, max_index_width + 1):
        for query in workload.queries:
            query_indexes = set()
            for index in potential_indexes:
                if index.columns[0] in query.columns:
                    query_indexes.add(index)
            candidates |= query_indexes
        
        if current_width < max_index_width:
            multicolumn_candidates = set()
            for index in candidates:
                for column in (set(index.table().columns) & set(workload.indexable_columns())) - set(index.columns):
                    multicolumn_candidates.add(Index(index.columns + (column,)))
            potential_indexes = candidates | multicolumn_candidates
    
    return candidates


def run_index_selection():
    """
    Main entry point: Run index selection on configured benchmark workload.
    
    Benchmark is selected via BENCHMARK environment variable:
    - tpch (default): TPC-H, 18 queries, benchbase database
    - tpcds: TPC-DS, 30 queries (sampled), benchbase_tpcds database
    - job: JOB (IMDB), 20 queries (top by frequency), benchbase_job database
    """
    import time
    
    # Project root for absolute paths
    project_root = os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB"))
    
    # Select benchmark from environment variable (default: tpch)
    benchmark = os.environ.get("BENCHMARK", "tpch").lower()
    
    # Benchmark-specific configuration
    BENCHMARK_CONFIG = {
        "tpch": {
            "db_name": "benchbase",
            "schema_file": f"{project_root}/configuration_loader/database/schema_tpch.json",
            "workload_file": f"{project_root}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
            "num_queries": 18,
            "budget_mb": 500,
        },
        "tpcds": {
            "db_name": "benchbase_tpcds",
            "schema_file": f"{project_root}/configuration_loader/database/schema_tpcds.json",
            "workload_file": f"{project_root}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
            "num_queries": 30,
            "budget_mb": 500,
        },
        "job": {
            "db_name": "benchbase_job",
            "schema_file": f"{project_root}/configuration_loader/database/schema_job.json",
            "workload_file": f"{project_root}/workload_generator/template_based/job_work_temp_multi_freq.json",
            "num_queries": 10,  # Top 10 by frequency (~50% of workload weight)
            "budget_mb": 2000,
        },
    }
    
    if benchmark not in BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark: {benchmark}. Supported: {list(BENCHMARK_CONFIG.keys())}")
    
    config = BENCHMARK_CONFIG[benchmark]
    # FULL_WORKLOAD override: use all queries when set by evaluator_full.py
    if os.environ.get("FULL_WORKLOAD", "").lower() == "true":
        config = config.copy()
        config["num_queries"] = 999  # Use all available queries
    
    # TPCDS_WORKLOAD_FILE override: use custom workload file for tpcds
    if benchmark == "tpcds" and os.environ.get("TPCDS_WORKLOAD_FILE"):
        config = config.copy()
        config["workload_file"] = os.environ.get("TPCDS_WORKLOAD_FILE")
        print(f"  Using custom workload: {config['workload_file']}")
    
    print(f"Using benchmark: {benchmark.upper()}")
    
    # Common configuration
    MAX_INDEX_WIDTH = 2
    MAX_INDEXES = 15  # Matches evaluator constraint
    
    start_time = time.time()
    
    # 1. Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{project_root}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    # 2. Load schema
    _, columns = heu_com.get_columns_from_schema(config["schema_file"])
    
    # 3. Load workload
    with open(config["workload_file"], "r") as rf:
        work_list = json.load(rf)
    
    # Get queries based on benchmark config
    all_queries = work_list[0]
    if benchmark == "job" and config["num_queries"] < len(all_queries):
        sorted_queries = sorted(all_queries, key=lambda x: x[2], reverse=True)
        first_workload = sorted_queries[:config["num_queries"]]
        print(f"  Selected top {config['num_queries']} queries by frequency")
    else:
        first_workload = all_queries[:config["num_queries"]]
    
    # 4. Create workload object
    exp_conf = {}
    workload = Workload(heu_com.read_row_query(
        first_workload, exp_conf, columns,
        type="", varying_frequencies=True, seed=666
    ))
    
    # 5. Generate candidate indexes
    candidates = generate_candidates(workload, max_index_width=MAX_INDEX_WIDTH)
    
    # 6. Create cost evaluation object
    cost_evaluation = CostEvaluation(connector)
    
    # 7. Calculate baseline cost (no indexes)
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    
    # 8. Run index selection
    selected_indexes = select_best_indexes(
        workload=workload,
        candidate_indexes=candidates,
        cost_evaluation=cost_evaluation,
        max_indexes=MAX_INDEXES,
        budget_MB=config["budget_mb"]
    )
    
    # 9. Calculate optimized cost
    optimized_cost = cost_evaluation.calculate_cost(workload, selected_indexes, store_size=True)
    
    selection_time = time.time() - start_time
    
    # Clean up
    connector.close()
    
    return selected_indexes, selection_time, baseline_cost, optimized_cost


if __name__ == "__main__":
    print("=" * 80)
    print("  Extend Index Selection - Initial Program")
    print("=" * 80)
    
    indexes, time_taken, baseline, optimized = run_index_selection()
    
    print(f"\n✅ Selection completed in {time_taken:.2f}s")
    print(f"\nSelected {len(indexes)} indexes:")
    for i, idx in enumerate(indexes, 1):
        size_mb = (idx.estimated_size or 0) / (1024 * 1024)
        print(f"  {i}. {idx} ({size_mb:.2f} MB)")
    
    cost_reduction = (baseline - optimized) / baseline if baseline > 0 else 0
    print(f"\nBaseline cost: {baseline:.2f}")
    print(f"Optimized cost: {optimized:.2f}")
    print(f"Cost reduction: {cost_reduction*100:.1f}%")









