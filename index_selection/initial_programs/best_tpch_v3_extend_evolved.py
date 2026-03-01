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
    max_indexes: int = 15,
    budget_MB: float = 500.0,
    min_target_indexes: int = 12,  # Aim for 12-15 indexes to avoid under-utilization
) -> Set[Index]:
    """
    TPCH-optimized greedy selection with table-balanced quotas and join-key prioritization.
    
    Key improvements:
    - Select 12-15 indexes (min_target_indexes enforced) to utilize budget.
    - Balance index distribution across tables (partsupp/orders/lineitem emphasized).
    - Prioritize join keys over pure filter predicates.
    - Precompute single-index benefits and sizes with store_size=True.
    - Greedy selection with re-evaluation on top-K candidates for speed (O(n^2)).
    """
    budget_bytes = budget_MB * 1024 * 1024
    if max_indexes <= 0 or budget_bytes <= 0 or not candidate_indexes:
        return set()

    # Baseline cost (no indexes) and ensure size estimation is populated via store_size=True
    baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)

    # Helper: extract table/column names robustly
    def _table_name(idx: Index) -> str:
        t = idx.table()
        return getattr(t, "name", str(t))

    def _col_names(idx: Index) -> List[str]:
        names = []
        for c in idx.columns:
            names.append(getattr(c, "name", str(c)))
        return names

    # TPCH join keys to prioritize (table, column)
    join_keys = {
        ("lineitem", "l_orderkey"),
        ("lineitem", "l_partkey"),
        ("lineitem", "l_suppkey"),
        ("orders", "o_orderkey"),
        ("orders", "o_custkey"),
        ("customer", "c_custkey"),
        ("supplier", "s_suppkey"),
        ("supplier", "s_nationkey"),
        ("partsupp", "ps_partkey"),
        ("partsupp", "ps_suppkey"),
        ("part", "p_partkey"),
        ("nation", "n_nationkey"),
        ("region", "r_regionkey"),
    }

    # Balanced distribution targets: enforce min and max per table (critical for latency)
    table_min = {
        "partsupp": 3,
        "orders": 2,
        "lineitem": 2,
        "supplier": 1,
        "customer": 1,
        "nation": 1,
        "part": 1,
        "region": 1,
    }
    table_max = {
        "partsupp": 4,
        "orders": 3,
        "lineitem": 3,
        "supplier": 2,
        "customer": 2,
        "nation": 1,
        "part": 2,
        "region": 1,
    }
    table_counts: Dict[str, int] = {t: 0 for t in set(list(table_min.keys()) + list(table_max.keys()))}

    # Precompute single-index benefits and ensure sizes via store_size=True
    # This also satisfies the size estimation requirement for selected indexes later.
    candidate_info = []  # List of dicts per candidate
    # Sort candidates to stabilize selection order
    sorted_candidates = sorted(list(candidate_indexes))
    for idx in sorted_candidates:
        # Evaluate single-index cost; store_size=True populates idx.estimated_size
        single_cost = cost_evaluation.calculate_cost(workload, {idx}, store_size=True)
        size = idx.estimated_size or 0
        benefit = baseline_cost - single_cost  # Higher is better
        tname = _table_name(idx)
        cnames = _col_names(idx)
        # Reliability weighting: prioritize join keys, de-emphasize date filters
        join_weight = 1.0
        if any((tname, c) in join_keys for c in cnames):
            join_weight *= 1.6
        elif any(c.lower().endswith("key") for c in cnames):
            join_weight *= 1.2
        # Extra boost for partsupp join hub
        ps_hits = sum(("partkey" in c.lower() or "suppkey" in c.lower()) for c in cnames)
        if _table_name(idx).lower() == "partsupp":
            if ps_hits >= 2:
                join_weight *= 1.2
            elif ps_hits == 1:
                join_weight *= 1.1
        # Date penalty to avoid misleading cost improvements on large fact tables
        date_cols = ("orderdate", "shipdate", "receiptdate", "commitdate", "shipmode")
        date_penalty = 0.7 if any(any(dc in c.lower() for dc in date_cols) for c in cnames) else 1.0
        candidate_info.append({
            "idx": idx,
            "size": size,
            "benefit": benefit,
            "table": tname,
            "join_weight": join_weight,
            "date_penalty": date_penalty,
            "columns": cnames,
        })

    # Filter out clearly non-beneficial candidates
    candidate_info = [c for c in candidate_info if c["benefit"] > 0 and c["size"] > 0]

    # Score function combining benefit/size, join priority, and table quota factor
    def base_score(c) -> float:
        cnt = table_counts.get(c["table"], 0)
        if cnt < table_min.get(c["table"], 0):
            quota_factor = 1.5
        elif cnt < table_max.get(c["table"], max_indexes):
            quota_factor = 1.0
        else:
            quota_factor = 0.6
        # Mildly discourage over-weighting large fact tables once minimum met
        if c["table"] in ("lineitem", "orders") and cnt >= table_min.get(c["table"], 0):
            quota_factor *= 0.9
        return (c["benefit"] / max(c["size"], 1)) * c["join_weight"] * c.get("date_penalty", 1.0) * quota_factor

    # Greedy selection
    selected: List[Index] = []
    selected_set: Set[Index] = set()
    current_cost = baseline_cost
    current_size = 0

    # Pre-sort by base score
    candidate_info.sort(key=base_score, reverse=True)

    # At each iteration, only re-evaluate top-K promising candidates for speed
    TOP_K = 12

    while len(selected) < max_indexes and current_size < budget_bytes and candidate_info:
        # Take top-K feasible candidates by base score
        feasible = []
        for c in candidate_info:
            # Enforce max-per-table to maintain balanced distribution
            if table_counts.get(c["table"], 0) >= table_max.get(c["table"], max_indexes):
                continue
            if current_size + c["size"] <= budget_bytes:
                feasible.append(c)
            if len(feasible) >= TOP_K:
                break

        if not feasible:
            break

        # Recompute marginal benefit for feasible subset with current selection
        best_c = None
        best_score = 0.0
        best_new_cost = current_cost

        for c in feasible:
            # Evaluate marginal cost with this candidate added
            new_cost = cost_evaluation.calculate_cost(workload, selected_set | {c["idx"]}, store_size=True)
            marginal_benefit = current_cost - new_cost
            if marginal_benefit <= 0:
                continue
            cnt = table_counts.get(c["table"], 0)
            quota_factor = 1.5 if cnt < table_min.get(c["table"], 0) else (1.0 if cnt < table_max.get(c["table"], max_indexes) else 0.6)
            if c["table"] in ("lineitem", "orders") and cnt >= table_min.get(c["table"], 0):
                quota_factor *= 0.9
            score = (marginal_benefit / max(c["size"], 1)) * c["join_weight"] * c.get("date_penalty", 1.0) * quota_factor
            if score > best_score:
                best_score = score
                best_c = c
                best_new_cost = new_cost

        # If no positive marginal candidate found, but we under-target (< min_target_indexes),
        # fall back to best base-scored candidate (still positive single benefit)
        if best_c is None and len(selected) < min_target_indexes:
            for c in candidate_info:
                # Respect table max to avoid over-weighting fact tables
                if table_counts.get(c["table"], 0) >= table_max.get(c["table"], max_indexes):
                    continue
                if current_size + c["size"] <= budget_bytes:
                    best_c = c
                    best_new_cost = current_cost - c["benefit"]  # approximate
                    break

        if best_c is None:
            break

        # Commit selection
        selected.append(best_c["idx"])
        selected_set.add(best_c["idx"])
        table_counts[best_c["table"]] = table_counts.get(best_c["table"], 0) + 1
        current_size += best_c["size"]
        current_cost = best_new_cost

        # Remove chosen candidate from pool
        candidate_info = [c for c in candidate_info if c["idx"] != best_c["idx"]]

        # Prefer balanced distribution: lightly penalize over-quota tables by re-sorting
        candidate_info.sort(key=base_score, reverse=True)

    # Final fill step: ensure we reach at least min_target_indexes if possible,
    # preferring small, join-key beneficial indexes within budget.
    if len(selected) < min_target_indexes and candidate_info:
        # Filter remaining candidates with positive single benefit and within budget
        remaining = [c for c in candidate_info if c["benefit"] > 0]
        # Sort by reliability (join_weight with date_penalty), then benefit/size ratio
        remaining.sort(key=lambda c: (c["join_weight"] * c.get("date_penalty", 1.0), c["benefit"] / max(c["size"], 1)), reverse=True)
        for c in remaining:
            if len(selected) >= max_indexes:
                break
            # Enforce max-per-table cap
            if table_counts.get(c["table"], 0) >= table_max.get(c["table"], max_indexes):
                continue
            if current_size + c["size"] > budget_bytes:
                continue
            selected.append(c["idx"])
            selected_set.add(c["idx"])
            table_counts[c["table"]] = table_counts.get(c["table"], 0) + 1
            current_size += c["size"]
            if len(selected) >= min_target_indexes:
                break

    # Ensure sizes are populated for all selected indexes (store_size=True requirement)
    if selected_set:
        cost_evaluation.calculate_cost(workload, selected_set, store_size=True)

    return selected_set
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









