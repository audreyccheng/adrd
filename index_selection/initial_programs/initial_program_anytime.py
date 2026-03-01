"""
Initial Program: Anytime (DTA) Algorithm for OpenEvolve
========================================================
Based on DTA Anytime algorithm (SQL Server Database Tuning Advisor):
- Generate candidates from workload
- Add merged indexes (combine two indexes from same table)
- Use multiple seeds (each single index as a seed)
- Greedy expansion from each seed
- Time-bounded search with best configuration tracking

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
from typing import Set, List, Dict, Optional, Tuple
import time as time_module
import itertools

sys.path.insert(0, os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB")))

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index, index_merge


# EVOLVE-BLOCK-START
def add_merged_indexes(
    indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_index_width: int = 2
) -> Set[Index]:
    """
    Add merged indexes: combine two indexes from same table.
    This matches paper's _add_merged_indexes.
    """
    result = set(indexes)
    
    # Group indexes by table
    by_table = {}
    for idx in indexes:
        table = idx.table()
        if table not in by_table:
            by_table[table] = []
        by_table[table].append(idx)
    
    # For each pair of indexes from same table, try merging
    for table, table_indexes in by_table.items():
        for idx1, idx2 in itertools.permutations(table_indexes, 2):
            # Merge columns
            merged = index_merge(idx1, idx2)
            
            # Truncate if too wide
            if len(merged.columns) > max_index_width:
                merged = Index(merged.columns[:max_index_width])
            
            if merged not in result:
                # Estimate size for the new merged index
                cost_evaluation.estimate_size(merged)
                result.add(merged)
    
    return result


def select_best_indexes(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int = 5,
    budget_MB: float = 200.0,
    max_runtime_seconds: int = 60,  # Time limit for search
) -> Set[Index]:
    """
    Anytime-style selection: multi-seed greedy with time bounds.
    
    Algorithm:
    1. Add merged indexes (combine pairs from same table)
    2. Filter candidates that fit in budget
    3. Create seeds: each single candidate + empty set
    4. For each seed, greedily expand to find best configuration
    5. Track best configuration across all seeds
    6. Stop when time limit reached or all seeds evaluated
    
    This function will be evolved by OpenEvolve.
    """
    budget_bytes = budget_MB * 1024 * 1024
    
    if max_indexes == 0 or budget_MB == 0:
        return set()
    
    if not candidate_indexes:
        return set()
    
    # Step 1: Add merged indexes (no expensive utilized check)
    candidates_with_merged = add_merged_indexes(candidate_indexes, cost_evaluation, max_index_width=2)
    
    # Step 2: Filter candidates that fit in budget
    valid_candidates = set()
    for idx in candidates_with_merged:
        if (idx.estimated_size or 0) <= budget_bytes:
            valid_candidates.add(idx)
    
    if not valid_candidates:
        return set()
    
    # Step 3: Create seeds: empty set first, then each single index
    seeds = [set()]
    for idx in sorted(list(valid_candidates)):
        seeds.append({idx})
    
    # Track best configuration across all seeds
    best_config = (set(), float('inf'))  # (indexes, cost)
    
    start_time = time_module.time()
    
    for seed_idx, seed in enumerate(seeds):
        # Check time limit
        elapsed = time_module.time() - start_time
        if elapsed > max_runtime_seconds:
            break
        
        # Greedy expansion from this seed
        current_indexes = set(seed)
        remaining = valid_candidates - current_indexes
        
        # Calculate initial cost
        current_cost = cost_evaluation.calculate_cost(workload, current_indexes, store_size=True)
        
        while len(current_indexes) < max_indexes and remaining:
            # Check time limit
            if time_module.time() - start_time > max_runtime_seconds:
                break
            
            best_addition = None
            best_new_cost = current_cost
            
            # Try adding each remaining candidate
            for candidate in sorted(list(remaining)):
                test_indexes = current_indexes | {candidate}
                
                # Check budget
                total_size = sum(idx.estimated_size or 0 for idx in test_indexes)
                if total_size > budget_bytes:
                    continue
                
                # Evaluate
                cost = cost_evaluation.calculate_cost(workload, test_indexes, store_size=True)
                
                if cost < best_new_cost:
                    best_addition = candidate
                    best_new_cost = cost
            
            # Add best candidate if improvement found
            if best_addition and best_new_cost < current_cost:
                current_indexes.add(best_addition)
                remaining.remove(best_addition)
                current_cost = best_new_cost
            else:
                # No improvement, stop expanding this seed
                break
        
        # Update best configuration if this is better
        if current_cost < best_config[1]:
            best_config = (current_indexes.copy(), current_cost)
    
    return best_config[0]
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
    """Main entry point: Run index selection on configured benchmark workload."""
    import time
    
    project_root = os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB"))
    benchmark = os.environ.get("BENCHMARK", "tpch").lower()
    
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
            "num_queries": 10,
            "budget_mb": 2000,
        },
    }
    
    if benchmark not in BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark: {benchmark}. Supported: {list(BENCHMARK_CONFIG.keys())}")
    
    config = BENCHMARK_CONFIG[benchmark]
    if os.environ.get("FULL_WORKLOAD", "").lower() == "true":
        config = config.copy()
        config["num_queries"] = 999
    print(f"Using benchmark: {benchmark.upper()}")
    
    MAX_INDEX_WIDTH = 2
    MAX_INDEXES = 15
    
    start_time = time.time()
    
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{project_root}/configuration_loader/database/db_con.conf")
    
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
        first_workload = sorted_queries[:config["num_queries"]]
    else:
        first_workload = all_queries[:config["num_queries"]]
    
    exp_conf = {}
    workload = Workload(heu_com.read_row_query(
        first_workload, exp_conf, columns,
        type="", varying_frequencies=True, seed=666
    ))
    
    candidates = generate_candidates(workload, max_index_width=MAX_INDEX_WIDTH)
    cost_evaluation = CostEvaluation(connector)
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    
    selected_indexes = select_best_indexes(
        workload=workload,
        candidate_indexes=candidates,
        cost_evaluation=cost_evaluation,
        max_indexes=MAX_INDEXES,
        budget_MB=config["budget_mb"],
        max_runtime_seconds=60
    )
    
    optimized_cost = cost_evaluation.calculate_cost(workload, selected_indexes, store_size=True)
    selection_time = time.time() - start_time
    
    connector.close()
    return selected_indexes, selection_time, baseline_cost, optimized_cost


if __name__ == "__main__":
    print("=" * 80)
    print("  Anytime (DTA) Index Selection - Simplified (Merged Indexes)")
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
