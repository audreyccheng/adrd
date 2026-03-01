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
    Hybrid Extend + Exploration:
    - Greedy add under budget
    - Prefix extension up to width 3 (Extend-style)
    - Light exploration: swap-based local search + short simulated annealing
    - Benefit-per-byte ranking and cost caching
    """
    import random, math

    budget_bytes = int(budget_MB * 1024 * 1024)
    if max_indexes <= 0 or budget_MB <= 0 or not candidate_indexes:
        return set()
    # Slightly relax improvement threshold to allow small beneficial moves
    impr_thr = min(1.0005, max(1.0, float(min_improvement)))

    # Ensure size estimates exist; fallback to a small default if missing
    def size_bytes(idx: Index) -> int:
        sz = idx.estimated_size or 0
        if sz <= 0:
            try:
                cost_evaluation.estimate_size(idx)
                sz = idx.estimated_size or 0
            except Exception:
                sz = 0
        if sz <= 0:
            sz = 16 * 1024  # 16KB fallback
            try:
                idx.estimated_size = sz
            except Exception:
                pass
        return int(max(1, sz))

    for idx in candidate_indexes:
        _ = size_bytes(idx)

    # Cost cache for efficiency
    cost_cache: Dict[frozenset, float] = {}
    def eval_cost(index_set: Set[Index]) -> float:
        key = frozenset(index_set)
        if key in cost_cache:
            return cost_cache[key]
        c = cost_evaluation.calculate_cost(workload, set(index_set), store_size=False)
        cost_cache[key] = c
        return c

    baseline_cost = eval_cost(set())

    # Separate single and multi-column candidates
    single_column = [idx for idx in candidate_indexes if len(idx.columns) == 1 and size_bytes(idx) <= budget_bytes]
    multi_column = [idx for idx in candidate_indexes if len(idx.columns) > 1 and size_bytes(idx) <= budget_bytes]

    # Get extension columns (for extending existing indexes)
    extension_columns = set()
    for idx in candidate_indexes:
        for col in idx.columns:
            extension_columns.add(col)

    # Initialize with greedy selection under budget
    current_combination: List[Index] = []
    current_cost = baseline_cost

    def get_current_size() -> int:
        return sum(size_bytes(idx) for idx in current_combination)

    all_candidates = single_column + multi_column
    used_candidates = set()

    improved = True
    while improved and len(current_combination) < max_indexes:
        improved = False
        best_candidate = None
        best_cost = current_cost
        cur_size = get_current_size()

        for candidate in all_candidates:
            if candidate in used_candidates or candidate in current_combination:
                continue
            csz = size_bytes(candidate)
            if cur_size + csz > budget_bytes:
                continue
            test = set(current_combination)
            test.add(candidate)
            c = eval_cost(test)
            if current_cost > 0 and c < best_cost:
                improvement_ratio = current_cost / max(1e-12, c)
                if improvement_ratio >= impr_thr:
                    best_cost = c
                    best_candidate = candidate

        if best_candidate is not None:
            current_combination.append(best_candidate)
            used_candidates.add(best_candidate)
            current_cost = best_cost
            improved = True

    # Phase 2: Extend existing indexes (up to width 3)
    improved = True
    max_extend_iterations = 6
    extend_iterations = 0
    while improved and extend_iterations < max_extend_iterations:
        improved = False
        extend_iterations += 1
        for i, existing_idx in enumerate(list(current_combination)):
            if len(existing_idx.columns) >= 3:
                continue
            for col in list(extension_columns):
                if col in existing_idx.columns:
                    continue
                if hasattr(col, "table") and col.table != existing_idx.table():
                    continue
                extended_idx = Index(existing_idx.columns + (col,))
                # Estimate size of extended index
                cost_evaluation.estimate_size(extended_idx)
                old_size = size_bytes(existing_idx)
                new_size = size_bytes(extended_idx)
                size_diff = new_size - old_size
                if get_current_size() + size_diff > budget_bytes:
                    continue
                test_list = list(current_combination)
                test_list[i] = extended_idx
                c = eval_cost(set(test_list))
                if c < current_cost:
                    improvement_ratio = current_cost / max(1e-12, c)
                    if improvement_ratio >= impr_thr:
                        current_combination[i] = extended_idx
                        current_cost = c
                        improved = True
                        break
            if improved:
                break

    # Build a small high-quality pool (benefit-per-byte) for exploration
    outside = [idx for idx in candidate_indexes if idx not in current_combination and size_bytes(idx) <= budget_bytes]
    scored = []
    for idx in outside:
        single_c = eval_cost({idx})
        benefit = max(0.0, baseline_cost - single_c)
        denom = float(size_bytes(idx))
        score = benefit / denom if denom > 0 else 0.0
        scored.append((score, benefit, idx))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top_k = min(len(scored), max(24, 5 * max_indexes))
    pool = [t[2] for t in scored[:top_k]]

    # Local swap-based refinement (very light)
    def swap_local_search(sol_list: List[Index], cur_cost_val: float, max_rounds: int = 1):
        sol_set = set(sol_list)
        best_cost_val = cur_cost_val
        improved_local = True
        rounds = 0
        while improved_local and rounds < max_rounds:
            improved_local = False
            rounds += 1
            cur_sz = sum(size_bytes(i) for i in sol_set)
            for out in list(sol_set):
                out_sz = size_bytes(out)
                for inn in pool:
                    if inn in sol_set:
                        continue
                    inn_sz = size_bytes(inn)
                    new_sz = cur_sz - out_sz + inn_sz
                    if new_sz > budget_bytes:
                        continue
                    test = set(sol_set)
                    test.remove(out)
                    test.add(inn)
                    c = eval_cost(test)
                    if c + 1e-9 < best_cost_val:
                        sol_set = test
                        best_cost_val = c
                        improved_local = True
                        cur_sz = new_sz
                        break
                if improved_local:
                    break
        return list(sol_set), best_cost_val

    current_combination, current_cost = swap_local_search(current_combination, current_cost, max_rounds=1)

    # Short simulated annealing to escape local optima
    def sol_size(sol: Set[Index]) -> int:
        return sum(size_bytes(i) for i in sol)

    def random_neighbor(sol: Set[Index]) -> Set[Index]:
        s = set(sol)
        move = random.random()
        if move < 0.33 and s:
            # remove
            out = random.choice(tuple(s))
            s.remove(out)
            return s
        elif move < 0.66:
            # add
            choices = [i for i in pool if i not in s]
            if not choices:
                return s
            cand = random.choice(choices)
            if len(s) + 1 <= max_indexes and sol_size(s) + size_bytes(cand) <= budget_bytes:
                s.add(cand)
            return s
        else:
            # swap
            if not s:
                return s
            choices = [i for i in pool if i not in s]
            if not choices:
                return s
            out = random.choice(tuple(s))
            inn = random.choice(choices)
            new_sz = sol_size(s) - size_bytes(out) + size_bytes(inn)
            if new_sz <= budget_bytes:
                s.remove(out)
                s.add(inn)
            return s

    def anneal(sol_list: List[Index], cur_cost_val: float, steps: int = 10):
        T = 0.05
        cur_sol = set(sol_list)
        cur_c = cur_cost_val
        best_sol = set(cur_sol)
        best_c = cur_c
        for _ in range(steps):
            neigh = random_neighbor(cur_sol)
            if len(neigh) > max_indexes or sol_size(neigh) > budget_bytes:
                T *= 0.95
                continue
            nc = eval_cost(neigh)
            delta = nc - cur_c
            if delta < 0 or (cur_c > 0 and math.exp(-delta / max(1e-9, cur_c * T)) > random.random()):
                cur_sol, cur_c = neigh, nc
                if nc < best_c:
                    best_sol, best_c = set(cur_sol), cur_c
            T *= 0.95
        return list(best_sol), best_c

    seeds = []
    if current_combination:
        seeds.append(list(current_combination))
    seeds.append([])  # empty seed
    if pool:
        seeds.append([pool[0]])  # best singleton by ratio

    best_sol_list = list(current_combination)
    best_cost = current_cost
    for seed in seeds[:3]:
        # Start from seed -> greedy add from pool if possible
        sol = set(seed)
        cur_sz = sol_size(sol)
        cur_c = eval_cost(sol)
        improved_add = True
        while improved_add and len(sol) < max_indexes:
            improved_add = False
            best_idx = None
            best_cand_cost = cur_c
            for idx in pool:
                if idx in sol:
                    continue
                if cur_sz + size_bytes(idx) > budget_bytes:
                    continue
                test = set(sol)
                test.add(idx)
                c = eval_cost(test)
                if c < best_cand_cost:
                    best_cand_cost = c
                    best_idx = idx
            if best_idx is not None:
                sol.add(best_idx)
                cur_sz += size_bytes(best_idx)
                cur_c = best_cand_cost
                improved_add = True
        # Anneal and light swap
        sol_list, sol_cost = anneal(list(sol), cur_c, steps=10)
        sol_list, sol_cost = swap_local_search(sol_list, sol_cost, max_rounds=1)
        if sol_cost < best_cost and len(sol_list) <= max_indexes and sum(size_bytes(i) for i in sol_list) <= budget_bytes:
            best_sol_list = sol_list
            best_cost = sol_cost

    # Kick-and-refill if stuck to escape local optimum (single kick)
    try:
        cur_best_set = set(best_sol_list)
        if cur_best_set:
            base_c = eval_cost(cur_best_set)
            # Contribution: how much cost increases if we remove idx
            contribs = []
            for idx in cur_best_set:
                c_wo = eval_cost(cur_best_set - {idx})
                contribs.append((base_c - c_wo, idx))
            contribs.sort(key=lambda t: t[0])
            worst_idx = contribs[0][1] if contribs else None
            if worst_idx is not None:
                kicked = set(cur_best_set)
                kicked.remove(worst_idx)
                kicked_sz = sum(size_bytes(i) for i in kicked)
                kicked_cost = eval_cost(kicked)
                # Greedy refill from pool
                improved_add = True
                while improved_add and len(kicked) < max_indexes:
                    improved_add = False
                    best_idx = None
                    best_cand_cost = kicked_cost
                    for cand in pool:
                        if cand in kicked:
                            continue
                        if kicked_sz + size_bytes(cand) > budget_bytes:
                            continue
                        test = set(kicked)
                        test.add(cand)
                        c = eval_cost(test)
                        if c < best_cand_cost:
                            best_cand_cost = c
                            best_idx = cand
                    if best_idx is not None:
                        kicked.add(best_idx)
                        kicked_sz += size_bytes(best_idx)
                        kicked_cost = best_cand_cost
                        improved_add = True
                if kicked_cost + 1e-9 < best_cost and len(kicked) <= max_indexes and kicked_sz <= budget_bytes:
                    best_sol_list = list(kicked)
                    best_cost = kicked_cost
    except Exception:
        pass

    # Final strict enforcement of constraints
    final_set = set(best_sol_list)
    def marginal_delta(idx: Index, sol: Set[Index], base_cost: float) -> float:
        t = set(sol)
        t.remove(idx)
        c = eval_cost(t)
        return c - base_cost

    while len(final_set) > max_indexes or sol_size(final_set) > budget_bytes:
        base = eval_cost(final_set)
        worst = None
        worst_d = float("inf")
        for idx in list(final_set):
            d = marginal_delta(idx, final_set, base)
            if d < worst_d:
                worst_d = d
                worst = idx
        if worst is None:
            break
        final_set.remove(worst)

    return set(final_set)
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
        db_name=config["db_name"], user="audreycc", password=""
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









