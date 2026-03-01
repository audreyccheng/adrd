"""
Faithful AutoAdmin Implementation (Chaudhuri & Narasayya 1997)
Enumerate naive (brute-force seed) + enumerate greedy (expansion)
Multi-level iteration: per-query candidate generation, multicolumn from selected indexes

Supports multiple benchmarks via BENCHMARK environment variable:
- BENCHMARK=tpch (default): TPC-H workload, 18 queries, benchbase database
- BENCHMARK=tpcds: TPC-DS workload, 79 queries, benchbase_tpcds database
- BENCHMARK=job: JOB (IMDB) workload, 33 query templates, benchbase_job database
"""

import sys
import os
import json
import configparser
from typing import Set, Tuple
import itertools

sys.path.insert(0, os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB")))

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index


# EVOLVE-BLOCK-START
def enumerate_naive(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    number_indexes_naive: int,
    budget_bytes: float
) -> Tuple[Set[Index], float]:
    lowest_cost_indexes = set()
    lowest_cost = None
    
    for number_of_indexes in range(1, number_indexes_naive + 1):
        for index_combination in sorted(list(itertools.combinations(
                candidate_indexes, number_of_indexes))):
            
            total_size = sum(index.estimated_size or 0 for index in index_combination)
            if total_size > budget_bytes:
                continue
            
            cost = cost_evaluation.calculate_cost(workload, index_combination, store_size=True)
            cost = round(cost, 2)
            
            if not lowest_cost or cost < lowest_cost:
                lowest_cost_indexes = set(index_combination)
                lowest_cost = cost
    
    if not lowest_cost:
        lowest_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
        lowest_cost_indexes = set()
    
    return lowest_cost_indexes, lowest_cost


def enumerate_greedy(
    workload,
    current_indexes: Set[Index],
    current_cost: float,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int,
    budget_bytes: float
) -> Tuple[Set[Index], float]:
    if len(current_indexes) >= max_indexes:
        return current_indexes, current_cost
    
    total_size = sum(index.estimated_size or 0 for index in current_indexes)
    if total_size > budget_bytes:
        return current_indexes, current_cost
    
    if not candidate_indexes:
        return current_indexes, current_cost
    
    best_index = None
    best_cost = None
    
    for index in sorted(list(candidate_indexes)):
        test_indexes = current_indexes | {index}
        
        total_size = sum(idx.estimated_size or 0 for idx in test_indexes)
        if total_size > budget_bytes:
            continue
        
        cost = cost_evaluation.calculate_cost(workload, test_indexes, store_size=True)
        cost = round(cost, 2)
        
        if not best_cost or cost < best_cost:
            best_index = index
            best_cost = cost
    
    if best_index and best_cost < current_cost:
        current_indexes = current_indexes | {best_index}
        candidate_indexes = candidate_indexes - {best_index}
        
        return enumerate_greedy(
            workload,
            current_indexes,
            best_cost,
            candidate_indexes,
            cost_evaluation,
            max_indexes,
            budget_bytes
        )
    
    return set(sorted(list(current_indexes))), current_cost


def enumerate_combinations(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int,
    max_indexes_naive: int,
    budget_bytes: float
) -> Set[Index]:
    if not candidate_indexes:
        return set()
    
    number_indexes_naive = min(max_indexes_naive, len(candidate_indexes))
    current_indexes, current_cost = enumerate_naive(
        workload,
        candidate_indexes,
        cost_evaluation,
        number_indexes_naive,
        budget_bytes
    )
    
    remaining_candidates = candidate_indexes - current_indexes
    number_indexes = min(max_indexes, len(candidate_indexes))
    
    final_indexes, final_cost = enumerate_greedy(
        workload,
        current_indexes,
        current_cost,
        remaining_candidates,
        cost_evaluation,
        number_indexes,
        budget_bytes
    )
    
    return final_indexes


def potential_indexes_for_query(query, potential_indexes: Set[Index]) -> Set[Index]:
    indexes = set()
    for index in sorted(list(potential_indexes)):
        if index.columns[0] in query.columns:
            indexes.add(index)
    return indexes


def select_index_candidates(
    workload,
    potential_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int,
    max_indexes_naive: int,
    budget_bytes: float
) -> Set[Index]:
    candidates = set()
    
    for query in workload.queries:
        query_workload = Workload([query])
        query_indexes = potential_indexes_for_query(query, potential_indexes)
        
        selected_for_query = enumerate_combinations(
            query_workload,
            query_indexes,
            cost_evaluation,
            max_indexes=max_indexes,
            max_indexes_naive=max_indexes_naive,
            budget_bytes=budget_bytes
        )
        
        candidates |= selected_for_query
    
    return set(sorted(list(candidates)))


def create_multicolumn_indexes(
    workload,
    indexes: Set[Index]
) -> Set[Index]:
    multicolumn_candidates = set()
    
    for index in indexes:
        indexable_cols = set(workload.indexable_columns())
        table_cols = set(index.table().columns)
        current_cols = set(index.columns)
        
        available_cols = (table_cols & indexable_cols) - current_cols
        
        for column in available_cols:
            multicolumn_candidates.add(Index(index.columns + (column,)))
    
    return multicolumn_candidates


def select_best_indexes(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int = 5,
    budget_MB: float = 200.0,
    max_indexes_naive: int = 2
) -> Set[Index]:
    """
    AutoAdmin algorithm implementation with correct signature.
    NOTE: This ignores candidate_indexes and generates its own (legacy behavior).
    New evolutions should use candidate_indexes parameter!
    """
    budget_bytes = budget_MB * 1024 * 1024
    max_index_width = 2  # Fixed for AutoAdmin
    
    if max_indexes == 0 or budget_MB == 0:
        return set()
    
    # Legacy: Generate own candidates (should use candidate_indexes in evolved versions)
    potential_indexes = workload.potential_indexes()
    
    for current_width in range(1, max_index_width + 1):
        candidates = select_index_candidates(
            workload,
            potential_indexes,
            cost_evaluation,
            max_indexes,
            max_indexes_naive,
            budget_bytes
        )
        
        if not candidates:
            return set()
        
        selected_indexes = enumerate_combinations(
            workload,
            candidates,
            cost_evaluation,
            max_indexes,
            max_indexes_naive,
            budget_bytes
        )
        
        if current_width < max_index_width:
            multicolumn_indexes = create_multicolumn_indexes(workload, selected_indexes)
            potential_indexes = selected_indexes | multicolumn_indexes
        else:
            return set(sorted(list(selected_indexes)))
    
    return set(sorted(list(selected_indexes)))
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
    
    return set(sorted(list(candidates)))


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
    print(f"Using benchmark: {benchmark.upper()}")
    
    # Common configuration
    MAX_INDEX_WIDTH = 2
    MAX_INDEXES = 15  # Matches evaluator constraint
    MAX_INDEXES_NAIVE = 1
    
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
        type="",
        varying_frequencies=True,
        seed=666
    ))
    
    # 5. Create cost evaluation object
    cost_evaluation = CostEvaluation(connector)
    
    # 6. Calculate baseline cost (no indexes)
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    
    # 7. Generate candidate indexes
    candidates = generate_candidates(workload, max_index_width=MAX_INDEX_WIDTH)
    
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
    print("  Index Selection - Faithful AutoAdmin Implementation")
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


