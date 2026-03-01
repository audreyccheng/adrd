"""
Initial Program: DB2Advis Algorithm for OpenEvolve
====================================================
Based on DB2 Advisor (Valentin et al., ICDE 2000):
- Calculate index benefits per query
- Combine subsumed indexes
- Select by benefit-to-size ratio until budget exhausted

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
from typing import Set, List, Dict
import itertools

sys.path.insert(0, os.environ.get("INDEX_PROJECT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "Index_EAB")))

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index


class IndexBenefit:
    """Tracks benefit of an index across queries."""
    def __init__(self, index: Index):
        self.index = index
        self.benefit = 0  # Total cost savings
        self.query_benefits = {}  # Per-query benefits
    
    def size(self):
        return self.index.estimated_size or 0
    
    def benefit_size_ratio(self):
        if self.size() == 0:
            return float('inf') if self.benefit > 0 else 0
        return self.benefit / self.size()


# EVOLVE-BLOCK-START
def select_best_indexes(
    workload,
    candidate_indexes: Set[Index],
    cost_evaluation: CostEvaluation,
    max_indexes: int = 5,
    budget_MB: float = 200.0,
) -> Set[Index]:
    """
    DB2Advis-style selection: benefit-to-size ratio ranking + greedy marginal selection.
    
    Algorithm:
    1. Calculate single-index benefit/size ratio for initial ranking
    2. Use greedy marginal benefit selection (like AutoAdmin) to pick indexes
    3. This ensures we utilize budget effectively on complex workloads
    
    This function will be evolved by OpenEvolve.
    """
    budget_bytes = budget_MB * 1024 * 1024
    
    if max_indexes == 0 or budget_MB == 0:
        return set()
    
    if not candidate_indexes:
        return set()
    
    # Step 1: Calculate baseline cost (no indexes)
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    
    # Step 2: Calculate single-index benefit for ranking (include all, not just positive)
    index_benefits = []
    for index in sorted(list(candidate_indexes)):
        index_size = index.estimated_size or 0
        # Skip if single index exceeds budget
        if index_size > budget_bytes:
            continue
        
        # Calculate cost with this single index
        cost_with_index = cost_evaluation.calculate_cost(
            workload, {index}, store_size=True
        )
        
        benefit = baseline_cost - cost_with_index
        ib = IndexBenefit(index)
        ib.benefit = benefit
        index_benefits.append(ib)
    
    if not index_benefits:
        return set()
    
    # Step 3: Sort by benefit-to-size ratio (descending) - indexes with benefit <= 0 go last
    index_benefits.sort(key=lambda x: x.benefit_size_ratio(), reverse=True)
    
    # Step 4: Greedy selection using MARGINAL benefit (key fix!)
    # This ensures we utilize budget effectively even when single-index benefits are low
    selected = set()
    current_cost = baseline_cost
    
    # First pass: add indexes in benefit/size order while they improve cost
    for ib in index_benefits:
        if len(selected) >= max_indexes:
            break
        
        # Check storage constraint
        current_size = sum(idx.estimated_size or 0 for idx in selected)
        if current_size + ib.size() > budget_bytes:
            continue
        
        # Check MARGINAL benefit (cost reduction when added to current set)
        test_set = selected | {ib.index}
        new_cost = cost_evaluation.calculate_cost(workload, test_set, store_size=True)
        
        if new_cost < current_cost:
            selected.add(ib.index)
            current_cost = new_cost
    
    # Second pass: try remaining candidates that might help in combination
    # (indexes with negative single-benefit may help when combined with others)
    remaining = [ib for ib in index_benefits if ib.index not in selected]
    
    for ib in remaining:
        if len(selected) >= max_indexes:
            break
        
        current_size = sum(idx.estimated_size or 0 for idx in selected)
        if current_size + ib.size() > budget_bytes:
            continue
        
        test_set = selected | {ib.index}
        new_cost = cost_evaluation.calculate_cost(workload, test_set, store_size=True)
        
        if new_cost < current_cost:
            selected.add(ib.index)
            current_cost = new_cost
    
    return selected
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
    
    print(f"📊 Loaded workload: {len(workload.queries)} queries")
    
    # 5. Generate candidate indexes
    candidates = generate_candidates(workload, max_index_width=MAX_INDEX_WIDTH)
    print(f"🔍 Generated {len(candidates)} candidate indexes")
    
    # 6. Create cost evaluation object
    cost_evaluation = CostEvaluation(connector)
    
    # 7. Calculate baseline cost (no indexes)
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
    print(f"📈 Baseline cost: {baseline_cost:.2f}")
    
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
    print("  DB2Advis Index Selection - Initial Program")
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



