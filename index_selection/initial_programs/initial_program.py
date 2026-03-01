"""
Initial Program: Index Selection Algorithm for OpenEvolve
===========================================================
This is the starting point algorithm that OpenEvolve will evolve.

Based on AutoAdmin algorithm (Chaudhuri & Narasayya 1997):
- Enumerate naive: Brute force search for seed configuration (size m <= k)
- Enumerate greedy: Greedy expansion to full configuration (size k)

Supports multiple benchmarks via BENCHMARK environment variable:
- BENCHMARK=tpch (default): TPC-H workload, 18 queries, benchbase database
- BENCHMARK=tpcds: TPC-DS workload, 79 queries, benchbase_tpcds database
- BENCHMARK=job: JOB (IMDB) workload, 33 query templates, benchbase_job database

OpenEvolve will mutate the selection strategy to discover better approaches.
"""

import sys
import os
import json
import configparser
from typing import Set, Tuple
import itertools

# Add project root to path
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
    max_indexes_naive: int = 1
) -> Set[Index]:
    """
    Select the best indexes from candidates to minimize workload cost.
    
    This function will be evolved by OpenEvolve to discover better selection strategies.
    
    Args:
        workload: Workload object containing queries
        candidate_indexes: Set of candidate indexes to choose from
        cost_evaluation: Cost evaluation object for simulating index benefit
        max_indexes: Maximum number of indexes to select
        budget_MB: Storage budget in megabytes
        max_indexes_naive: Number of indexes for exhaustive enumeration seed
        
    Returns:
        Set of selected indexes
    """
    budget_bytes = budget_MB * 1024 * 1024
    
    if max_indexes == 0 or budget_MB == 0:
        return set()
    
    if not candidate_indexes:
        return set()
    
    # Sort candidates for reproducibility
    candidate_indexes = set(sorted(list(candidate_indexes)))
    
    # Phase 1: Enumerate naive (brute force seed)
    number_indexes_naive = min(max_indexes_naive, len(candidate_indexes))
    lowest_cost_indexes = set()
    lowest_cost = None
    
    # Try all combinations up to size number_indexes_naive
    for number_of_indexes in range(1, number_indexes_naive + 1):
        for index_combination in sorted(list(itertools.combinations(
                candidate_indexes, number_of_indexes))):
            
            # Check storage constraint (skip if any index has None size)
            total_size = sum(index.estimated_size or 0 for index in index_combination)
            if total_size > budget_bytes:
                continue
            
            # Evaluate cost
            cost = cost_evaluation.calculate_cost(workload, index_combination, store_size=True)
            cost = round(cost, 2)
            
            if not lowest_cost or cost < lowest_cost:
                lowest_cost_indexes = set(index_combination)
                lowest_cost = cost
    
    # If no valid configuration found, use empty set
    if not lowest_cost:
        lowest_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
        lowest_cost_indexes = set()
    
    # Phase 2: Enumerate greedy (expand seed greedily)
    current_indexes = lowest_cost_indexes
    current_cost = lowest_cost
    remaining_indexes = candidate_indexes - current_indexes
    
    while len(current_indexes) < max_indexes and remaining_indexes:
        best_index = None
        best_cost = None
        
        # Try adding each remaining index
        for index in sorted(list(remaining_indexes)):
            test_indexes = current_indexes | {index}
            
            # Check storage constraint (skip if any index has None size)
            total_size = sum(idx.estimated_size or 0 for idx in test_indexes)
            if total_size > budget_bytes:
                continue
            
            # Evaluate cost
            cost = cost_evaluation.calculate_cost(workload, test_indexes, store_size=True)
            cost = round(cost, 2)
            
            if not best_cost or cost < best_cost:
                best_index = index
                best_cost = cost
        
        # Add best index if it improves cost
        if best_index and best_cost < current_cost:
            current_indexes.add(best_index)
            remaining_indexes.remove(best_index)
            current_cost = best_cost
        else:
            # No improvement, stop
            break
    
    return set(sorted(list(current_indexes)))
# EVOLVE-BLOCK-END


def generate_candidates(workload, max_index_width: int = 2):
    """
    Generate candidate indexes from workload.
    
    Args:
        workload: Workload object
        max_index_width: Maximum number of columns per index
        
    Returns:
        Set of candidate indexes
    """
    candidates = set()
    
    # Start with single-column indexes from workload
    potential_indexes = workload.potential_indexes()
    
    for current_width in range(1, max_index_width + 1):
        # Select candidates for this width
        for query in workload.queries:
            # Get indexes relevant to this query
            query_indexes = set()
            for index in potential_indexes:
                # Leading column must be in query
                if index.columns[0] in query.columns:
                    query_indexes.add(index)
            
            candidates |= query_indexes
        
        # Create multi-column indexes for next iteration
        if current_width < max_index_width:
            multicolumn_candidates = set()
            for index in candidates:
                # Extend with other columns from same table
                for column in (set(index.table().columns) & set(workload.indexable_columns())) - set(index.columns):
                    multicolumn_candidates.add(Index(index.columns + (column,)))
            potential_indexes = candidates | multicolumn_candidates
    
    return set(sorted(list(candidates)))


def run_index_selection():
    """
    Main entry point: Run index selection on configured benchmark workload.
    
    Benchmark is selected via BENCHMARK environment variable:
    - tpch (default): TPC-H, 18 queries, benchbase database
    - tpcds: TPC-DS, 30 queries (sampled), benchbase_tpcds database
    - job: JOB (IMDB), 20 queries (top by frequency), benchbase_job database
    
    Returns:
        Tuple of (selected_indexes, selection_time, baseline_cost, optimized_cost)
    """
    import time
    
    # Project root for absolute paths
    project_root = os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB"))
    
    # Select benchmark from environment variable (default: tpch)
    benchmark = os.environ.get("BENCHMARK", "tpch").lower()
    
    # Benchmark-specific configuration (matching paper.md settings)
    # Paper: "storage budget, set at half the size of the dataset" (line 896-897)
    # Paper: "default maximum index width is 2" (line 897-898)
    BENCHMARK_CONFIG = {
        "tpch": {
            "db_name": "benchbase",
            "schema_file": f"{project_root}/configuration_loader/database/schema_tpch.json",
            "workload_file": f"{project_root}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
            "num_queries": 18,  # Paper: 22 templates, workload has 18 usable
            "budget_mb": 500,   # Half of 1GB SF1 dataset
        },
        "tpcds": {
            "db_name": "benchbase_tpcds",
            "schema_file": f"{project_root}/configuration_loader/database/schema_tpcds.json",
            "workload_file": f"{project_root}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
            "num_queries": 30,  # First 30 queries for faster evolution (full: 79)
            "budget_mb": 500,   # Half of ~1GB SF1 dataset
        },
        "job": {
            "db_name": "benchbase_job",
            "schema_file": f"{project_root}/configuration_loader/database/schema_job.json",
            "workload_file": f"{project_root}/workload_generator/template_based/job_work_temp_multi_freq.json",
            "num_queries": 10,  # Top 10 by frequency (~50% of workload weight)
            "budget_mb": 2000,  # Half of ~4GB IMDB dataset
        },
    }
    
    if benchmark not in BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark: {benchmark}. Supported: {list(BENCHMARK_CONFIG.keys())}")
    
    config = BENCHMARK_CONFIG[benchmark]
    # FULL_WORKLOAD override: use all queries when set by evaluator_full.py
    if os.environ.get("FULL_WORKLOAD", "").lower() == "true":
        config = config.copy()
        config["num_queries"] = 999  # Use all available queries
    print(f"Using benchmark: {benchmark.upper()}")
    
    # Common configuration
    MAX_INDEX_WIDTH = 2  # Paper default for all benchmarks
    MAX_INDEXES = 15     # Matches evaluator constraint
    MAX_INDEXES_NAIVE = 1  # Use 1 for speed (2 is O(n²) DB calls)
    
    start_time = time.time()
    
    # 1. Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{project_root}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user="audreycc", password=""
    )
    
    # 2. Load schema
    _, columns = heu_com.get_columns_from_schema(config["schema_file"])
    
    # 3. Load workload
    with open(config["workload_file"], "r") as rf:
        work_list = json.load(rf)
    
    # Get queries based on benchmark config
    # For JOB: select top N by frequency (covers 81%+ of workload weight)
    # For others: take first N queries
    all_queries = work_list[0]
    if benchmark == "job" and config["num_queries"] < len(all_queries):
        # Sort by frequency (index 2) descending, take top N
        sorted_queries = sorted(all_queries, key=lambda x: x[2], reverse=True)
        first_workload = sorted_queries[:config["num_queries"]]
        print(f"  Selected top {config['num_queries']} queries by frequency")
    else:
        first_workload = all_queries[:config["num_queries"]]
    
    # 4. Create workload object
    exp_conf = {}
    workload = Workload(heu_com.read_row_query(
        first_workload, exp_conf, columns,
        type="",
        varying_frequencies=True,
        seed=666
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
        budget_MB=config["budget_mb"],
        max_indexes_naive=MAX_INDEXES_NAIVE
    )
    
    # 9. Calculate optimized cost
    optimized_cost = cost_evaluation.calculate_cost(workload, selected_indexes, store_size=True)
    
    selection_time = time.time() - start_time
    
    # Clean up
    connector.close()
    
    return selected_indexes, selection_time, baseline_cost, optimized_cost


if __name__ == "__main__":
    print("=" * 80)
    print("  Index Selection - Initial Program")
    print("=" * 80)
    
    indexes, time_taken, baseline, optimized = run_index_selection()
    
    print(f"\n✅ Selection completed in {time_taken:.2f}s")
    print(f"\nSelected {len(indexes)} indexes:")
    for i, idx in enumerate(indexes, 1):
        size_mb = idx.estimated_size / (1024 * 1024)
        print(f"  {i}. {idx} ({size_mb:.2f} MB)")
    
    cost_reduction = (baseline - optimized) / baseline if baseline > 0 else 0
    print(f"\nBaseline cost: {baseline:.2f}")
    print(f"Optimized cost: {optimized:.2f}")
    print(f"Cost reduction: {cost_reduction*100:.1f}%")

