"""
FULL Evaluator: Fitness function for index selection algorithms
================================================================
Evaluates evolved index selection strategies on ALL benchmark workloads
with FULL query sets (no sampling/subsets).

This is the FULL evaluation version - use evaluator.py for faster iteration.

IMPORTANT: This evaluator RE-EVALUATES indexes on the FULL workload.
Programs may train on subsets, but final scores are always computed
on complete query sets to prevent overfitting.

NEW: RELIABILITY-WEIGHTED SCORING (Dec 2025)
============================================
When USE_RELIABILITY_WEIGHTS=true (default), the evaluator uses pre-computed
query reliability weights derived from historical cost-latency correlation
analysis (see COST_MODEL.md). This improves algorithm ranking by down-weighting
queries where PostgreSQL cost estimates don't correlate well with actual latency.

Improvements in Spearman's ρ (algorithm ranking correlation):
- TPC-H: -0.248 → 0.382 (improvement: +0.630)
- JOB:   -0.103 → 0.176 (improvement: +0.279)  
- TPC-DS: 0.567 → 0.567 (unchanged, already good)

Set USE_RELIABILITY_WEIGHTS=false to disable and use standard scoring.

Supported benchmarks (via BENCHMARK env var):
- tpch (default): TPC-H workload, 18 queries (FULL), benchbase database (SF=1)
- tpcds: TPC-DS workload, 79 queries (FULL), benchbase_tpcds database (SF=1)
- job: JOB (IMDB) workload, 33 queries (FULL), benchbase_job database
- dsb: DSB workload, 53 queries (FULL), benchbase_tpcds database (uses TPC-DS data)
- tpch_sf10: TPC-H SF=10, 18 queries, benchbase_tpch_sf10 database (13GB, 5GB budget)
- tpcds_sf10: TPC-DS SF=10, 79 queries, benchbase_tpcds_sf10 database (22GB, 5GB budget)
- all: Run tpch, tpcds, job benchmarks in parallel, aggregate scores

Scoring Philosophy:
- PRIMARY: Query cost reduction (reliability-weighted when enabled)
- SECONDARY: HARD constraint enforcement (zero score if violated)
- NOT SCORED: Selection time (runs once, indexes used millions of times)

Metrics:
- query_cost_reduction: % reduction in query execution cost (0.0 to 1.0)
- weighted_cost_reduction: Reliability-weighted cost reduction (when enabled)
- constraint_score: 1.0 if valid, 0.0 if ANY constraint violated (HARD)
- storage_used_mb: Storage consumed by selected indexes
- num_indexes: Number of indexes selected
- selection_time: Time to run selection algorithm (tracked but not scored)
- combined_score: 0.0 if constraints violated, else (weighted) cost_reduction
"""

import configparser
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root for imports
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", "Index_EAB")
sys.path.insert(0, PROJECT_ROOT)
# Export for subprocess programs
os.environ['INDEX_PROJECT_ROOT'] = PROJECT_ROOT

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index


# =============================================================================
# RELIABILITY-WEIGHTED SCORING (from COST_MODEL.md analysis)
# =============================================================================
# Load pre-computed query reliability weights derived from historical 
# cost-latency correlation analysis. Queries where cost correlates well with
# latency get higher weights; queries with poor/inverse correlation get 
# downweighted or zeroed.
#
# This improves algorithm ranking:
#   - TPC-H: -0.248 → 0.382 (Spearman ρ improvement: +0.630)
#   - JOB:   -0.103 → 0.176 (Spearman ρ improvement: +0.279)
#   - TPC-DS: 0.567 → 0.567 (unchanged, already good)
# =============================================================================

# Feature toggle: set USE_RELIABILITY_WEIGHTS=false to disable
USE_RELIABILITY_WEIGHTS = os.environ.get("USE_RELIABILITY_WEIGHTS", "true").lower() == "true"

# =============================================================================
# PERIODIC LATENCY VALIDATION (TPC-H only)
# =============================================================================
# At certain intervals, run actual latency evaluation on candidate programs.
# This catches "degenerate" solutions that optimize cost but hurt latency.
# 
# CRITICAL FOR TPC-H: Cost model has poor correlation with latency!
# - 8/18 queries have ZERO reliability (cost changes don't predict latency)
# - Evolution MUST be guided by actual latency, not cost estimates
# 
# Configuration:
#   - LATENCY_CHECK_INTERVAL: Check every N iterations (5 for TPC-H!)
#   - LATENCY_DEGRADATION_THRESHOLD: 0.0 = must be better than no indexes
#   - LATENCY_PENALTY_FACTOR: 0.0 = ZERO score if latency degrades
#   - TPCH_LATENCY_PRIMARY: If true, use latency_reduction as primary metric
#
# Set ENABLE_LATENCY_VALIDATION=false to disable.
# =============================================================================
ENABLE_LATENCY_VALIDATION = os.environ.get("ENABLE_LATENCY_VALIDATION", "true").lower() == "true"
LATENCY_CHECK_INTERVAL = int(os.environ.get("LATENCY_CHECK_INTERVAL", "1"))  # EVERY iteration for TPC-H!
LATENCY_DEGRADATION_THRESHOLD = float(os.environ.get("LATENCY_DEGRADATION_THRESHOLD", "0.0"))  # Must be positive
LATENCY_PENALTY_FACTOR = float(os.environ.get("LATENCY_PENALTY_FACTOR", "0.0"))  # Zero score if degraded

# TPC-H baseline latency reduction (from db2advis baseline: 19.7%)
TPCH_BASELINE_LATENCY_REDUCTION = 0.197

# NEW: Use latency as primary metric for TPC-H (cost doesn't correlate well)
TPCH_LATENCY_PRIMARY = os.environ.get("TPCH_LATENCY_PRIMARY", "true").lower() == "true"

# =============================================================================
# TPC-DS LATENCY-BASED FITNESS (NEW - Jan 2026)
# =============================================================================
# For TPC-DS, cost estimates do NOT correlate with latency (only 59% direction
# agreement, 22 queries with inverse correlation). Use actual latency instead!
#
# Key insight: Queries 5 (1457s) and 56 (605s) dominate 88% of evaluation time
# but represent only 0.2% of workload frequency. Skip them for fast evaluation.
#
# Configuration:
#   - USE_TPCDS_LATENCY_FITNESS: Enable latency-based fitness for TPC-DS
#   - TPCDS_SKIP_QUERIES: Comma-separated query IDs to skip (default: "5,56")
#   - TPCDS_QUERY_TIMEOUT: Per-query timeout in seconds (default: 300)
# =============================================================================
# Auto-detect latency mode based on benchmark:
# - tpcds_77: Always use latency fitness (77-query subset)
# - Environment variable USE_TPCDS_LATENCY_FITNESS=true: Force latency for tpcds/tpcds_27
_USE_TPCDS_LATENCY_ENV = os.environ.get("USE_TPCDS_LATENCY_FITNESS", "false").lower() == "true"
TPCDS_SKIP_QUERIES_STR = os.environ.get("TPCDS_SKIP_QUERIES", "5,56")
TPCDS_SKIP_QUERIES = set(int(q.strip()) for q in TPCDS_SKIP_QUERIES_STR.split(",") if q.strip())
TPCDS_QUERY_TIMEOUT = int(os.environ.get("TPCDS_QUERY_TIMEOUT", "300"))
TPCDS_BASELINE_LATENCY_REDUCTION = 0.081  # autoadmin baseline: +8.1%

def should_use_tpcds_latency(benchmark):
    """Auto-detect if TPC-DS latency fitness should be used based on benchmark name."""
    if benchmark == "tpcds_77":
        return True  # Always use latency for tpcds_77
    if _USE_TPCDS_LATENCY_ENV and benchmark in ("tpcds", "tpcds_27"):
        return True
    return False

if _USE_TPCDS_LATENCY_ENV:
    print(f"🔬 TPC-DS LATENCY FITNESS enabled (env): skipping queries {TPCDS_SKIP_QUERIES}, timeout={TPCDS_QUERY_TIMEOUT}s")

# =============================================================================
# JOB LATENCY-BASED FITNESS (Jan 2026)
# =============================================================================
# For JOB, cost estimates significantly overestimate latency benefit (delta: -23%).
# With 16GB shared_buffers, all 33 queries complete in ~47s total, making full
# latency evaluation feasible during evolution (~2.6 hours for 100 iterations).
#
# Key insight: extend baseline achieves 7.9% latency reduction vs db2advis 3.7%.
# Evolution should optimize for actual latency, not cost proxy.
#
# Configuration:
#   - USE_JOB_LATENCY_FITNESS: Enable latency-based fitness for JOB
#   - JOB_SKIP_QUERIES: Comma-separated query IDs to skip (default: "" = none)
#   - JOB_QUERY_TIMEOUT: Per-query timeout in seconds (default: 30)
# =============================================================================
USE_JOB_LATENCY_FITNESS = os.environ.get("USE_JOB_LATENCY_FITNESS", "false").lower() == "true"
JOB_SKIP_QUERIES_STR = os.environ.get("JOB_SKIP_QUERIES", "")  # Empty = include all
JOB_SKIP_QUERIES = set(int(q.strip()) for q in JOB_SKIP_QUERIES_STR.split(",") if q.strip())
JOB_QUERY_TIMEOUT = int(os.environ.get("JOB_QUERY_TIMEOUT", "30"))
JOB_BASELINE_LATENCY_REDUCTION = 0.192  # extend baseline: +19.2% (remote, optimized PG config)

if USE_JOB_LATENCY_FITNESS:
    skip_msg = f"skipping queries {JOB_SKIP_QUERIES}" if JOB_SKIP_QUERIES else "all 33 queries"
    print(f"🔬 JOB LATENCY FITNESS enabled: {skip_msg}, timeout={JOB_QUERY_TIMEOUT}s")

def run_latency_validation(indexes_data, benchmark="tpch"):
    """
    Run actual latency validation on selected indexes.
    
    Creates REAL indexes, runs queries with EXPLAIN ANALYZE, measures actual
    execution time, then cleans up.
    
    Args:
        indexes_data: List of dicts with 'columns', 'table', 'estimated_size'
        benchmark: Only "tpch" is currently supported
        
    Returns:
        Dict with:
        - latency_reduction: float (e.g., 0.15 = 15% faster)
        - baseline_latency_ms: total weighted baseline latency
        - optimized_latency_ms: total weighted optimized latency
        - validation_time: seconds taken for validation
        - error: error message if failed
    """
    if benchmark != "tpch":
        return {"error": f"Latency validation only supports tpch, got {benchmark}"}
    
    import time as time_module
    start_time = time_module.time()
    
    config = BENCHMARK_CONFIG.get(benchmark)
    if not config:
        return {"error": f"Unknown benchmark: {benchmark}"}
    
    # Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    try:
        # Load schema and workload
        _, columns = heu_com.get_columns_from_schema(config["schema_file"])
        
        col_lookup = {}
        for col in columns:
            table_name = col.table.name if hasattr(col.table, 'name') else str(col.table)
            col_lookup[(table_name.lower(), col.name.lower())] = col
        
        with open(config["workload_file"], "r") as rf:
            work_list = json.load(rf)
        
        all_queries = work_list[0][:18]  # TPC-H has 18 queries
        
        workload = Workload(heu_com.read_row_query(
            all_queries, {}, columns,
            type="", varying_frequencies=True, seed=666
        ))
        
        # Helper: measure latency for a single query
        def measure_query_latency(query):
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
                        except Exception:
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
                except Exception:
                    return 0.0
        
        # Clean up any leftover test indexes
        try:
            result = connector.exec_fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'latval_%'",
                one=False
            )
            if result:
                for row in result:
                    connector.exec_only(f'DROP INDEX IF EXISTS "{row[0]}"')
                    connector.commit()
        except:
            pass
        
        # NOTE: PostgreSQL restart is NOT needed for fair comparison.
        # The warmup phase (running all queries once) fills the cache consistently.
        # With shared_buffers >> database size, warmup ensures everything is cached.
        # This was validated empirically: restart vs no-restart gives ~1% variance.
        
        # Run ANALYZE first to ensure fresh statistics for baseline
        print("  Running ANALYZE for baseline...")
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Query-based warmup: run all queries once (standard benchmarking practice)
        print("  Warming up cache (query-based)...")
        for query in workload.queries:
            try:
                measure_query_latency(query)
            except:
                pass
        
        # Measure baseline latency (no indexes) - warm cache
        baseline_latency = 0.0
        for query in workload.queries:
            latency_ms = measure_query_latency(query)
            baseline_latency += latency_ms * query.frequency
        
        # Reconstruct Index objects and create REAL indexes
        index_names = []
        for idx_data in indexes_data:
            table_str = idx_data['table']
            if '.' in table_str:
                table_name = table_str.split('.')[-1].lower()
            else:
                table_name = table_str.lower()
            
            col_names = "_".join(c.lower() for c in idx_data['columns'])[:30]
            index_name = f"latval_{table_name}_{col_names}"[:63]
            
            columns_str = ", ".join(idx_data['columns'])
            create_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_name} ({columns_str})'
            
            try:
                connector.exec_only(create_sql)
                connector.commit()
                index_names.append(index_name)
            except Exception as e:
                print(f"  ⚠️ Failed to create index {index_name}: {e}")
        
        # Run ANALYZE to update stats
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Query-based warmup with indexes (standard benchmarking practice)
        print("  Warming up cache (query-based, with indexes)...")
        for query in workload.queries:
            try:
                measure_query_latency(query)
            except:
                pass
        
        # Measure optimized latency (with indexes) - warm cache
        optimized_latency = 0.0
        for query in workload.queries:
            latency_ms = measure_query_latency(query)
            optimized_latency += latency_ms * query.frequency
        
        # Clean up indexes
        for name in index_names:
            try:
                connector.exec_only(f'DROP INDEX IF EXISTS "{name}"')
                connector.commit()
            except:
                pass
        
        connector.close()
        
        # Calculate reduction
        if baseline_latency > 0:
            latency_reduction = (baseline_latency - optimized_latency) / baseline_latency
        else:
            latency_reduction = 0.0
        
        validation_time = time_module.time() - start_time
        
        return {
            "latency_reduction": latency_reduction,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": optimized_latency,
            "validation_time": validation_time,
            "num_indexes_created": len(index_names),
        }
        
    except Exception as e:
        try:
            connector.close()
        except:
            pass
        return {"error": str(e), "latency_reduction": 0.0}


def run_tpcds_latency_validation(indexes_data, benchmark="tpcds", skip_queries=None, query_timeout=300):
    """
    Run actual latency validation for TPC-DS workload.
    
    Skips pathological slow queries (Q5, Q56 by default) to reduce evaluation time
    from ~40 min to ~6 min while maintaining 99.8% workload coverage.
    
    Args:
        indexes_data: List of dicts with 'columns', 'table', 'estimated_size'
        benchmark: Must be "tpcds" or "tpcds_27"
        skip_queries: Set of query IDs to skip (default: {5, 56})
        query_timeout: Per-query timeout in seconds (default: 300)
        
    Returns:
        Dict with:
        - latency_reduction: float (e.g., 0.15 = 15% faster)
        - baseline_latency_ms: total weighted baseline latency
        - optimized_latency_ms: total weighted optimized latency  
        - validation_time: seconds taken for validation
        - queries_evaluated: number of queries evaluated
        - queries_skipped: number of queries skipped
        - error: error message if failed
    """
    import time as time_module
    import signal
    
    if skip_queries is None:
        skip_queries = TPCDS_SKIP_QUERIES
    
    if benchmark not in ("tpcds", "tpcds_27", "tpcds_77"):
        return {"error": f"Latency validation only supports tpcds variants, got {benchmark}"}
    
    start_time = time_module.time()
    
    # Use tpcds config for tpcds_77 (same database/schema, just different query handling)
    config_key = "tpcds" if benchmark == "tpcds_77" else benchmark
    config = BENCHMARK_CONFIG.get(config_key)
    if not config:
        return {"error": f"Unknown benchmark: {benchmark}"}
    
    # Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    try:
        # Load schema and workload
        _, columns = heu_com.get_columns_from_schema(config["schema_file"])
        
        col_lookup = {}
        for col in columns:
            table_name = col.table.name if hasattr(col.table, 'name') else str(col.table)
            col_lookup[(table_name.lower(), col.name.lower())] = col
        
        # Always load full 79-query TPC-DS workload
        workload_file = f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json"
        with open(workload_file, "r") as rf:
            work_list = json.load(rf)
        
        all_queries = work_list[0][:79]  # TPC-DS has 79 queries
        
        workload = Workload(heu_com.read_row_query(
            all_queries, {}, columns,
            type="", varying_frequencies=True, seed=666
        ))
        
        # Filter out queries to skip
        queries_to_eval = []
        queries_skipped = []
        for i, query in enumerate(workload.queries):
            if i in skip_queries:
                queries_skipped.append(i)
            else:
                queries_to_eval.append((i, query))
        
        print(f"  📊 TPC-DS latency: evaluating {len(queries_to_eval)}/79 queries (skipping {queries_skipped})")
        
        # Helper: measure latency for a single query with timeout
        def measure_query_latency_with_timeout(query, timeout_sec=query_timeout):
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
                            # Set statement timeout
                            connector.exec_only(f"SET statement_timeout = '{timeout_sec * 1000}'")
                            result = connector.exec_fetch(explain_sql, one=True)
                            latency_ms = result[0][0]["Execution Time"]
                        except Exception as e:
                            if "canceling statement due to statement timeout" in str(e):
                                latency_ms = timeout_sec * 1000  # Use timeout as latency
                            else:
                                latency_ms = 0.0
                        finally:
                            connector.exec_only("SET statement_timeout = '0'")  # Reset timeout
                
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
                    # Set statement timeout
                    connector.exec_only(f"SET statement_timeout = '{timeout_sec * 1000}'")
                    result = connector.exec_fetch(explain_sql, one=True)
                    return result[0][0]["Execution Time"]
                except Exception as e:
                    if "canceling statement due to statement timeout" in str(e):
                        return timeout_sec * 1000  # Use timeout as latency
                    return 0.0
                finally:
                    connector.exec_only("SET statement_timeout = '0'")  # Reset timeout
        
        # Clean up any leftover test indexes
        try:
            result = connector.exec_fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'tpcds_latval_%'",
                one=False
            )
            if result:
                for row in result:
                    connector.exec_only(f'DROP INDEX IF EXISTS "{row[0]}"')
                    connector.commit()
        except:
            pass
        
        # Run ANALYZE first to ensure fresh statistics for baseline
        print("  Running ANALYZE for baseline...")
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Warmup pass: run each query once to ensure consistent warm cache
        print("  Warming up cache (baseline)...")
        warmup_count = 0
        for query_idx, query in queries_to_eval:
            measure_query_latency_with_timeout(query)
            warmup_count += 1
            if warmup_count % 20 == 0:
                print(f"    Warmup progress: {warmup_count}/{len(queries_to_eval)}")
        
        # Measure baseline latency (no indexes) - warm cache
        print("  Measuring baseline latency...")
        baseline_latency = 0.0
        for query_idx, query in queries_to_eval:
            latency_ms = measure_query_latency_with_timeout(query)
            baseline_latency += latency_ms * query.frequency
        
        # Reconstruct Index objects and create REAL indexes
        index_names = []
        for idx_data in indexes_data:
            table_str = idx_data['table']
            if '.' in table_str:
                table_name = table_str.split('.')[-1].lower()
            else:
                table_name = table_str.lower()
            
            col_names = "_".join(c.lower() for c in idx_data['columns'])[:30]
            index_name = f"tpcds_latval_{table_name}_{col_names}"[:63]
            
            columns_str = ", ".join(idx_data['columns'])
            create_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_name} ({columns_str})'
            
            try:
                connector.exec_only(create_sql)
                connector.commit()
                index_names.append(index_name)
            except Exception as e:
                print(f"  ⚠️ Failed to create index {index_name}: {e}")
        
        # Run ANALYZE to update stats
        print("  Running ANALYZE after index creation...")
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Warmup pass: run each query once to ensure consistent warm cache
        print("  Warming up cache (with indexes)...")
        warmup_count = 0
        for query_idx, query in queries_to_eval:
            measure_query_latency_with_timeout(query)
            warmup_count += 1
            if warmup_count % 20 == 0:
                print(f"    Warmup progress: {warmup_count}/{len(queries_to_eval)}")
        
        # Measure optimized latency (with indexes) - warm cache
        print("  Measuring optimized latency...")
        optimized_latency = 0.0
        for query_idx, query in queries_to_eval:
            latency_ms = measure_query_latency_with_timeout(query)
            optimized_latency += latency_ms * query.frequency
        
        # Clean up indexes
        print("  Cleaning up indexes...")
        for name in index_names:
            try:
                connector.exec_only(f'DROP INDEX IF EXISTS "{name}"')
                connector.commit()
            except:
                pass
        
        connector.close()
        
        # Calculate reduction
        if baseline_latency > 0:
            latency_reduction = (baseline_latency - optimized_latency) / baseline_latency
        else:
            latency_reduction = 0.0
        
        validation_time = time_module.time() - start_time
        
        return {
            "latency_reduction": latency_reduction,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": optimized_latency,
            "validation_time": validation_time,
            "num_indexes_created": len(index_names),
            "queries_evaluated": len(queries_to_eval),
            "queries_skipped": len(queries_skipped),
            "skip_query_ids": list(queries_skipped),
        }
        
    except Exception as e:
        try:
            connector.close()
        except:
            pass
        import traceback
        traceback.print_exc()
        return {"error": str(e), "latency_reduction": 0.0}


def run_job_latency_validation(indexes_data, benchmark="job", skip_queries=None, query_timeout=30):
    """
    Run actual latency validation for JOB (IMDB) workload.
    
    With 16GB shared_buffers, all 33 JOB queries complete in ~47s total (baseline).
    This makes full latency evaluation feasible during evolution.
    
    Args:
        indexes_data: List of dicts with 'columns', 'table', 'estimated_size'
        benchmark: Must be "job"
        skip_queries: Set of query IDs to skip (default: None = include all)
        query_timeout: Per-query timeout in seconds (default: 30)
        
    Returns:
        Dict with:
        - latency_reduction: float (e.g., 0.079 = 7.9% faster)
        - baseline_latency_ms: total weighted baseline latency
        - optimized_latency_ms: total weighted optimized latency  
        - validation_time: seconds taken for validation
        - queries_evaluated: number of queries evaluated
        - queries_skipped: number of queries skipped
        - error: error message if failed
    """
    import time as time_module
    
    if skip_queries is None:
        skip_queries = JOB_SKIP_QUERIES
    
    if benchmark != "job":
        return {"error": f"JOB latency validation only supports job, got {benchmark}"}
    
    start_time = time_module.time()
    
    config = BENCHMARK_CONFIG.get(benchmark)
    if not config:
        return {"error": f"Unknown benchmark: {benchmark}"}
    
    # Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    try:
        # Load schema and workload
        _, columns = heu_com.get_columns_from_schema(config["schema_file"])
        
        col_lookup = {}
        for col in columns:
            table_name = col.table.name if hasattr(col.table, 'name') else str(col.table)
            col_lookup[(table_name.lower(), col.name.lower())] = col
        
        # Load full 33-query JOB workload
        with open(config["workload_file"], "r") as rf:
            work_list = json.load(rf)
        
        all_queries = work_list[0][:33]  # JOB has 33 queries
        
        workload = Workload(heu_com.read_row_query(
            all_queries, {}, columns,
            type="", varying_frequencies=True, seed=666
        ))
        
        # Filter out queries to skip (if any)
        queries_to_eval = []
        queries_skipped = []
        for i, query in enumerate(workload.queries):
            if i in skip_queries:
                queries_skipped.append(i)
            else:
                queries_to_eval.append((i, query))
        
        skip_msg = f"(skipping {queries_skipped})" if queries_skipped else "(all queries)"
        print(f"  📊 JOB latency: evaluating {len(queries_to_eval)}/33 queries {skip_msg}")
        
        # Helper: measure latency for a single query with timeout
        def measure_query_latency_with_timeout(query, timeout_sec=query_timeout):
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
                            connector.exec_only(f"SET statement_timeout = '{timeout_sec * 1000}'")
                            result = connector.exec_fetch(explain_sql, one=True)
                            latency_ms = result[0][0]["Execution Time"]
                        except Exception as e:
                            if "canceling statement due to statement timeout" in str(e):
                                latency_ms = timeout_sec * 1000
                            else:
                                latency_ms = 0.0
                        finally:
                            connector.exec_only("SET statement_timeout = '0'")
                
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
                    connector.exec_only(f"SET statement_timeout = '{timeout_sec * 1000}'")
                    result = connector.exec_fetch(explain_sql, one=True)
                    return result[0][0]["Execution Time"]
                except Exception as e:
                    if "canceling statement due to statement timeout" in str(e):
                        return timeout_sec * 1000
                    return 0.0
                finally:
                    connector.exec_only("SET statement_timeout = '0'")
        
        # Clean up any leftover test indexes
        try:
            result = connector.exec_fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'job_latval_%'",
                one=False
            )
            if result:
                for row in result:
                    connector.exec_only(f'DROP INDEX IF EXISTS "{row[0]}"')
                    connector.commit()
        except:
            pass
        
        # Run ANALYZE first to ensure fresh statistics for baseline
        print("  Running ANALYZE for baseline...")
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Warmup pass: run each query once to ensure consistent warm cache
        print("  Warming up cache (baseline)...")
        for query_idx, query in queries_to_eval:
            measure_query_latency_with_timeout(query)
        
        # Measure baseline latency (no indexes) - warm cache
        print("  Measuring baseline latency...")
        baseline_latency = 0.0
        for query_idx, query in queries_to_eval:
            latency_ms = measure_query_latency_with_timeout(query)
            baseline_latency += latency_ms * query.frequency
        
        # Reconstruct Index objects and create REAL indexes
        index_names = []
        for idx_data in indexes_data:
            table_str = idx_data['table']
            if '.' in table_str:
                table_name = table_str.split('.')[-1].lower()
            else:
                table_name = table_str.lower()
            
            col_names = "_".join(c.lower() for c in idx_data['columns'])[:30]
            index_name = f"job_latval_{table_name}_{col_names}"[:63]
            
            columns_str = ", ".join(idx_data['columns'])
            create_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_name} ({columns_str})'
            
            try:
                connector.exec_only(create_sql)
                connector.commit()
                index_names.append(index_name)
            except Exception as e:
                print(f"  ⚠️ Failed to create index {index_name}: {e}")
        
        # Run ANALYZE to update stats
        print("  Running ANALYZE after index creation...")
        connector.exec_only("ANALYZE")
        connector.commit()
        
        # Warmup pass: run each query once to ensure consistent warm cache
        print("  Warming up cache (with indexes)...")
        for query_idx, query in queries_to_eval:
            measure_query_latency_with_timeout(query)
        
        # Measure optimized latency (with indexes) - warm cache
        print("  Measuring optimized latency...")
        optimized_latency = 0.0
        for query_idx, query in queries_to_eval:
            latency_ms = measure_query_latency_with_timeout(query)
            optimized_latency += latency_ms * query.frequency
        
        # Clean up indexes
        print("  Cleaning up indexes...")
        for name in index_names:
            try:
                connector.exec_only(f'DROP INDEX IF EXISTS "{name}"')
                connector.commit()
            except:
                pass
        
        connector.close()
        
        # Calculate reduction
        if baseline_latency > 0:
            latency_reduction = (baseline_latency - optimized_latency) / baseline_latency
        else:
            latency_reduction = 0.0
        
        validation_time = time_module.time() - start_time
        
        return {
            "latency_reduction": latency_reduction,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": optimized_latency,
            "validation_time": validation_time,
            "num_indexes_created": len(index_names),
            "queries_evaluated": len(queries_to_eval),
            "queries_skipped": len(queries_skipped),
            "skip_query_ids": list(queries_skipped),
        }
        
    except Exception as e:
        try:
            connector.close()
        except:
            pass
        import traceback
        traceback.print_exc()
        return {"error": str(e), "latency_reduction": 0.0}


# Load reliability weights from JSON file
RELIABILITY_WEIGHTS = {}
_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_reliability_weights.json")
if os.path.exists(_weights_path):
    try:
        with open(_weights_path, "r") as f:
            _weights_data = json.load(f)
            RELIABILITY_WEIGHTS = _weights_data.get("weights", {})
        print(f"📊 Loaded reliability weights for {list(RELIABILITY_WEIGHTS.keys())} benchmarks")
    except Exception as e:
        print(f"⚠️  Warning: Could not load reliability weights: {e}")
        RELIABILITY_WEIGHTS = {}
else:
    print(f"ℹ️  No reliability weights file found at {_weights_path}, using uniform weights")


def calculate_weighted_cost_reduction(per_query_costs, benchmark):
    """
    Calculate cost reduction weighted by query reliability.
    
    Queries where PostgreSQL cost estimates historically correlated well with
    actual latency get higher weights. This improves the fitness signal for
    evolution by down-weighting unreliable cost estimates.
    
    Args:
        per_query_costs: Dict mapping query_id -> {
            'baseline_cost': float,
            'optimized_cost': float, 
            'frequency': int
        }
        benchmark: Benchmark name ('tpch', 'tpcds', 'job')
        
    Returns:
        Tuple of (weighted_reduction, standard_reduction) where:
        - weighted_reduction: Reliability-weighted cost reduction [0, 1]
        - standard_reduction: Standard frequency-weighted cost reduction [0, 1]
    """
    weights = RELIABILITY_WEIGHTS.get(benchmark, {})
    
    # Calculate both weighted and standard reductions for comparison
    weighted_numerator = 0.0
    weighted_denominator = 0.0
    standard_numerator = 0.0  # Total cost reduction (freq-weighted)
    standard_denominator = 0.0  # Total baseline cost (freq-weighted)
    
    for query_id, costs in per_query_costs.items():
        baseline = costs.get('baseline_cost', 0)
        optimized = costs.get('optimized_cost', baseline)
        frequency = costs.get('frequency', 1)
        
        if baseline <= 0:
            continue
        
        # Per-query cost reduction
        reduction = (baseline - optimized) / baseline
        
        # Get reliability weight (default to small positive value for unknown queries)
        # Using 0.1 as default ensures new queries still contribute, but less than reliable ones
        reliability = weights.get(str(query_id), 0.1)
        
        # Weighted contribution: reliability * frequency * reduction
        weight = reliability * frequency
        weighted_numerator += reduction * weight
        weighted_denominator += weight
        
        # Standard contribution: frequency * costs
        standard_numerator += (baseline - optimized) * frequency
        standard_denominator += baseline * frequency
    
    # Calculate final reductions
    if weighted_denominator > 0:
        weighted_reduction = weighted_numerator / weighted_denominator
    else:
        weighted_reduction = 0.0
    
    if standard_denominator > 0:
        standard_reduction = standard_numerator / standard_denominator
    else:
        standard_reduction = 0.0
    
    # Clamp to [0, 1]
    weighted_reduction = max(0.0, min(1.0, weighted_reduction))
    standard_reduction = max(0.0, min(1.0, standard_reduction))
    
    return weighted_reduction, standard_reduction


# Query subset for faster evolution (uses only reliable queries)
# Set TPCDS_QUERY_SUBSET env var to comma-separated indices (0-indexed)
# Example: TPCDS_QUERY_SUBSET="1,2,3,4,5,9,10" (high+medium reliability queries)
TPCDS_QUERY_SUBSET = os.environ.get("TPCDS_QUERY_SUBSET", "").strip()
if TPCDS_QUERY_SUBSET:
    TPCDS_SUBSET_INDICES = set(int(x.strip()) for x in TPCDS_QUERY_SUBSET.split(",") if x.strip())
    print(f"📋 TPC-DS query subset enabled: {len(TPCDS_SUBSET_INDICES)} queries")
else:
    TPCDS_SUBSET_INDICES = None

# Benchmark-specific constraints (must match initial_program.py)
# Storage = half dataset size (paper default)
# FULL WORKLOAD: Uses ALL queries (not subsets)
BENCHMARK_CONSTRAINTS = {
    # SF=1 benchmarks (default)
    "tpch": {"budget_mb": 500.0, "max_indexes": 20, "num_queries": 18},   # 18 queries (full), 20 indexes
    "tpcds": {"budget_mb": 500.0, "max_indexes": 15, "num_queries": 79},  # 79 queries (full, was 30)
    "tpcds_27": {"budget_mb": 500.0, "max_indexes": 15, "num_queries": 27},  # 27 reliable queries
    "tpcds_77": {"budget_mb": 500.0, "max_indexes": 15, "num_queries": 77},  # 77 queries (skip Q5, Q56 for latency)
    "job": {"budget_mb": 2000.0, "max_indexes": 15, "num_queries": 33},   # 33 queries (full, was 10)
    "dsb": {"budget_mb": 500.0, "max_indexes": 15, "num_queries": 53},    # 53 queries (full), uses TPC-DS data
    "tpch_skew": {"budget_mb": 500.0, "max_indexes": 20, "num_queries": 18},  # TPC-H with Zipf skew
    # SF=10 benchmarks (larger scale)
    "tpch_sf10": {"budget_mb": 5000.0, "max_indexes": 20, "num_queries": 18},   # 13GB DB, 5GB budget
    "tpcds_sf10": {"budget_mb": 5000.0, "max_indexes": 15, "num_queries": 79},  # 22GB DB, 5GB budget
}

# Benchmark database and file configurations
BENCHMARK_CONFIG = {
    # SF=1 benchmarks (default)
    "tpch": {
        "db_name": "benchbase",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpch.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
    },
    "tpcds": {
        "db_name": "benchbase_tpcds",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
    },
    "tpcds_27": {  # 27 high-reliability queries with 100% table coverage
        "db_name": "benchbase_tpcds",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_27_reliable.json",
    },
    "tpcds_77": {  # 77 queries for latency evaluation (skip Q5, Q56 which take 88% of time)
        "db_name": "benchbase_tpcds",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
        # Note: queries 5 and 56 are skipped at evaluation time, not via workload file
    },
    "job": {
        "db_name": "benchbase_job",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_job.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/job_work_temp_multi_freq.json",
    },
    "dsb": {
        "db_name": "benchbase_tpcds",  # DSB uses TPC-DS data
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_dsb.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/dsb_work_temp_multi_freq.json",
    },
    "tpch_skew": {
        "db_name": "benchbase_tpch_skew",  # TPC-H with Zipf skewed data
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpch_skew.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpch_skew_work_temp_multi_freq.json",
    },
    # SF=10 benchmarks (larger scale - same schema/workload, different database)
    "tpch_sf10": {
        "db_name": "benchbase_tpch_sf10",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpch.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
    },
    "tpcds_sf10": {
        "db_name": "benchbase_tpcds_sf10",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
    },
}

# FULL evaluator runs all 3 benchmarks with complete query sets
ALL_BENCHMARKS = ["tpch", "tpcds", "job"]

# No timeout for full evaluation - let it run to completion
FULL_TIMEOUT_SECONDS = None


def reevaluate_on_full_workload(indexes_data, benchmark="tpch", return_per_query=False):
    """
    Re-evaluate selected indexes on the FULL workload.
    
    This is the KEY function that ensures evaluation is done on ALL queries,
    regardless of what subset the program used internally for selection.
    
    Args:
        indexes_data: List of dicts with 'columns', 'table', 'estimated_size'
        benchmark: Benchmark to evaluate on ("tpch", "tpcds", or "job")
        return_per_query: If True, also return per-query cost breakdown for
                         reliability-weighted scoring
        
    Returns:
        If return_per_query=False:
            Tuple of (baseline_cost, optimized_cost, storage_used_bytes)
        If return_per_query=True:
            Tuple of (baseline_cost, optimized_cost, storage_used_bytes, per_query_costs)
            where per_query_costs is a dict: query_id -> {baseline_cost, optimized_cost, frequency}
    """
    config = BENCHMARK_CONFIG.get(benchmark)
    constraints = BENCHMARK_CONSTRAINTS.get(benchmark)
    if not config or not constraints:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    num_queries = constraints["num_queries"]
    
    # Connect to database
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host="127.0.0.1", port="5432",
        db_name=config["db_name"], user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), password=""
    )
    
    try:
        # Load schema
        _, columns = heu_com.get_columns_from_schema(config["schema_file"])
        
        # Build column lookup: (table_name, col_name) -> Column object
        col_lookup = {}
        for col in columns:
            table_name = col.table.name if hasattr(col.table, 'name') else str(col.table)
            col_lookup[(table_name.lower(), col.name.lower())] = col
        
        # Load FULL workload
        with open(config["workload_file"], "r") as rf:
            work_list = json.load(rf)
        
        all_queries = work_list[0]
        original_indices = None  # Track original indices for reliability weights
        
        # For JOB, sort by frequency to get consistent ordering
        if benchmark == "job" and num_queries < len(all_queries):
            sorted_queries = sorted(all_queries, key=lambda x: x[2], reverse=True)
            first_workload = sorted_queries[:num_queries]
        elif benchmark == "tpcds" and TPCDS_SUBSET_INDICES is not None:
            # Filter to only the specified subset of queries (by position index)
            # Track original indices for reliability weight lookup
            original_indices = sorted(TPCDS_SUBSET_INDICES)
            first_workload = [all_queries[i] for i in original_indices if i < len(all_queries)]
            print(f"  📋 Using TPC-DS subset: {len(first_workload)} queries (indices: {original_indices[:5]}...)")
        else:
            first_workload = all_queries[:num_queries]
        
        workload = Workload(heu_com.read_row_query(
            first_workload, {}, columns,
            type="", varying_frequencies=True, seed=666
        ))
        
        subset_note = f" (subset)" if original_indices else ""
        print(f"  📊 Re-evaluating on {benchmark.upper()} workload{subset_note}: {len(workload.queries)} queries")
        
        # Create cost evaluation object
        cost_evaluation = CostEvaluation(connector)
        
        # Reconstruct Index objects from serialized data
        selected_indexes = set()
        total_storage = 0
        
        for idx_data in indexes_data:
            # Parse table name (format: "schema.table" or just "table")
            table_str = idx_data['table']
            if '.' in table_str:
                table_name = table_str.split('.')[-1].lower()
            else:
                table_name = table_str.lower()
            
            # Find matching columns
            idx_columns = []
            for col_name in idx_data['columns']:
                col_key = (table_name, col_name.lower())
                if col_key in col_lookup:
                    idx_columns.append(col_lookup[col_key])
                else:
                    # Try without table prefix
                    found = False
                    for (t, c), col_obj in col_lookup.items():
                        if c == col_name.lower():
                            idx_columns.append(col_obj)
                            found = True
                            break
                    if not found:
                        print(f"  ⚠️ Warning: Could not find column {col_name} in table {table_name}")
            
            if idx_columns:
                idx = Index(tuple(idx_columns))
                # Set estimated size from program's calculation
                idx.estimated_size = idx_data.get('estimated_size', 0)
                selected_indexes.add(idx)
                total_storage += idx.estimated_size or 0
        
        # =================================================================
        # Calculate costs - either aggregate or per-query
        # =================================================================
        if return_per_query:
            # Calculate per-query costs for reliability-weighted scoring
            per_query_costs = {}
            baseline_cost = 0.0
            optimized_cost = 0.0
            
            # First pass: calculate baseline costs (no indexes)
            cost_evaluation._prepare_cost_calculation(set(), store_size=True)
            for enum_idx, query in enumerate(workload.queries):
                # Use original index if subset, otherwise enumerated index
                # This ensures reliability weights are looked up correctly
                query_idx = original_indices[enum_idx] if original_indices else enum_idx
                
                # Get cost for this query with no indexes
                q_cost = connector.get_cost(query)
                per_query_costs[query_idx] = {
                    'baseline_cost': q_cost,
                    'optimized_cost': q_cost,  # Will be updated below
                    'frequency': query.frequency,
                }
                baseline_cost += q_cost * query.frequency
            
            # Second pass: calculate optimized costs (with indexes)
            if selected_indexes:
                cost_evaluation._prepare_cost_calculation(selected_indexes, store_size=True)
                for enum_idx, query in enumerate(workload.queries):
                    query_idx = original_indices[enum_idx] if original_indices else enum_idx
                    q_cost = connector.get_cost(query)
                    per_query_costs[query_idx]['optimized_cost'] = q_cost
                    optimized_cost += q_cost * query.frequency
            else:
                optimized_cost = baseline_cost
            
            # Close connection
            connector.close()
            
            return baseline_cost, optimized_cost, total_storage, per_query_costs
        else:
            # Original aggregate calculation (faster, no per-query breakdown)
            # Calculate baseline cost (no indexes)
            baseline_cost = cost_evaluation.calculate_cost(workload, set(), store_size=True)
            
            # Calculate optimized cost with selected indexes
            if selected_indexes:
                optimized_cost = cost_evaluation.calculate_cost(workload, selected_indexes, store_size=True)
            else:
                optimized_cost = baseline_cost
            
            # Close connection
            connector.close()
            
            return baseline_cost, optimized_cost, total_storage
        
    except Exception as e:
        connector.close()
        raise e


def run_program(program_path, timeout_seconds=None, benchmark="tpch"):
    """
    Run the index selection program in a separate process.
    
    Uses FULL_WORKLOAD=true to signal programs to use complete query sets.
    No timeout by default - runs until completion.
    
    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds (None = no timeout)
        benchmark: Benchmark to evaluate on ("tpch", "tpcds", or "job")
        
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
# For tpcds_27 and tpcds_77, tell the program to use tpcds (same database/schema)
actual_benchmark = 'tpcds' if '{benchmark}' in ('tpcds_27', 'tpcds_77') else '{benchmark}'
os.environ['BENCHMARK'] = actual_benchmark

# FULL EVALUATOR: Signal to use complete workload (all queries)
os.environ['FULL_WORKLOAD'] = 'true'

# Set INDEX_PROJECT_ROOT for programs that use it
os.environ['INDEX_PROJECT_ROOT'] = '{project_root}'

# Custom workload file for tpcds_27 (27 high-reliability queries)
if '{benchmark}' == 'tpcds_27':
    os.environ['TPCDS_WORKLOAD_FILE'] = '{project_root}/workload_generator/template_based/tpcds_work_27_reliable.json'

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
        
        # Wait for completion (no timeout)
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
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path, benchmark=None):
    """
    Evaluate an index selection program using FULL workloads.
    
    Sets FULL_WORKLOAD=true environment variable to signal programs
    to use complete query sets (no sampling).
    
    When USE_RELIABILITY_WEIGHTS=true (default), uses reliability-weighted
    cost reduction which down-weights queries where PostgreSQL cost estimates
    don't correlate well with actual latency.
    
    The program must have a run_index_selection() function that returns:
        (selected_indexes, selection_time, baseline_cost, optimized_cost)
    
    Args:
        program_path: Path to the program file to evaluate
        benchmark: Benchmark to evaluate on ("tpch", "tpcds", "job", or "all")
                   If None, reads from EVAL_BENCHMARK env var (default: "tpch")
        
    Returns:
        Dictionary with evaluation metrics:
        - query_cost_reduction: % reduction in query cost (0.0 to 1.0)
        - weighted_cost_reduction: Reliability-weighted cost reduction (if enabled)
        - constraint_score: Soft penalty for violations (0.0 to 1.0)
        - storage_used_mb: Storage used by selected indexes
        - num_indexes: Number of indexes selected
        - selection_time: Time taken for selection (tracked but not scored)
        - combined_score: 0.0 if constraints violated, else (weighted) cost_reduction
    """
    # Read benchmark from env var if not explicitly passed (for OpenEvolve integration)
    if benchmark is None:
        benchmark = os.environ.get("EVAL_BENCHMARK", "tpch")
    
    # Handle "all" benchmark - run all benchmarks in parallel
    if benchmark == "all":
        return evaluate_all_benchmarks(program_path, parallel=True)
    
    try:
        start_time = time.time()
        
        # Run the program to get selected indexes (ignore program's cost calculations)
        indexes_data, selection_time, _prog_baseline, _prog_optimized = run_program(
            program_path, benchmark=benchmark
        )
        
        # Calculate metrics
        num_indexes = len(indexes_data)
        
        # === KEY FIX: Re-evaluate on FULL workload ===
        # Don't trust program's cost - it may have used a subset of queries
        # Always evaluate on the complete workload for accurate metrics
        
        # Determine if we need per-query costs for weighted scoring
        use_weighted = USE_RELIABILITY_WEIGHTS and benchmark in RELIABILITY_WEIGHTS
        
        if use_weighted:
            # Get per-query costs for reliability-weighted scoring
            baseline_cost, optimized_cost, storage_bytes, per_query_costs = reevaluate_on_full_workload(
                indexes_data, benchmark=benchmark, return_per_query=True
            )
            
            # Calculate both weighted and standard cost reductions
            weighted_reduction, standard_reduction = calculate_weighted_cost_reduction(
                per_query_costs, benchmark
            )
            
            # Use weighted reduction as the primary metric
            cost_reduction = weighted_reduction
            
            # Log comparison for transparency
            if abs(weighted_reduction - standard_reduction) > 0.01:  # >1% difference
                print(f"  📊 Reliability-weighted: {weighted_reduction*100:.1f}% vs standard: {standard_reduction*100:.1f}%")
        else:
            # Standard aggregate calculation (no per-query breakdown)
            baseline_cost, optimized_cost, storage_bytes = reevaluate_on_full_workload(
                indexes_data, benchmark=benchmark, return_per_query=False
            )
            
            # Calculate standard cost reduction
            if baseline_cost > 0:
                cost_reduction = (baseline_cost - optimized_cost) / baseline_cost
                cost_reduction = max(0.0, min(1.0, cost_reduction))
            else:
                cost_reduction = 0.0
            
            standard_reduction = cost_reduction
            weighted_reduction = cost_reduction  # Same when not using weights
        
        eval_time = time.time() - start_time
        
        # Calculate storage used (from re-evaluation for accuracy)
        storage_used_mb = storage_bytes / (1024 * 1024)
        
        # Log the difference if program reported different costs
        if _prog_baseline > 0:
            prog_reduction = (_prog_baseline - _prog_optimized) / _prog_baseline
            if abs(prog_reduction - standard_reduction) > 0.01:  # >1% difference
                print(f"  ℹ️  Program reported {prog_reduction*100:.1f}% reduction, "
                      f"FULL workload: {standard_reduction*100:.1f}% reduction")
        
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
        # Valid programs scored on (weighted) cost reduction (the actual goal)
        if constraint_score == 0.0:
            combined_score = 0.0
            print(f"⚠️  COMBINED SCORE = 0.0 due to constraint violation")
        else:
            combined_score = cost_reduction  # Uses weighted reduction when enabled
        
        # === PERIODIC LATENCY VALIDATION ===
        # At certain intervals, validate that cost reduction actually helps latency
        # CRITICAL: For TPC-H and TPC-DS, cost model has poor correlation with latency!
        latency_validation_result = None
        latency_penalty_applied = False
        latency_reduction_actual = None
        
        # === TPC-DS LATENCY VALIDATION ===
        # Auto-detect: tpcds_77 always uses latency, others use env var
        if (should_use_tpcds_latency(benchmark) and 
            benchmark in ("tpcds", "tpcds_27", "tpcds_77") and 
            constraint_score > 0):  # Only validate valid programs
            
            print(f"🔬 Running TPC-DS latency validation (skipping queries {TPCDS_SKIP_QUERIES})...")
            
            latency_validation_result = run_tpcds_latency_validation(
                indexes_data, 
                benchmark="tpcds",  # Always use full tpcds config
                skip_queries=TPCDS_SKIP_QUERIES,
                query_timeout=TPCDS_QUERY_TIMEOUT
            )
            
            if "error" not in latency_validation_result:
                latency_red = latency_validation_result["latency_reduction"]
                latency_reduction_actual = latency_red
                val_time = latency_validation_result.get("validation_time", 0)
                queries_eval = latency_validation_result.get("queries_evaluated", 0)
                
                # Use latency reduction as the PRIMARY score for TPC-DS
                old_score = combined_score
                combined_score = max(0.0, latency_red)  # Use latency as score
                
                print(f"  📊 TPC-DS LATENCY FITNESS:")
                print(f"     Queries evaluated: {queries_eval}/79")
                print(f"     Latency reduction: {latency_red*100:.1f}%")
                print(f"     (cost-based score was: {old_score*100:.1f}%)")
                print(f"     Validation time: {val_time/60:.1f} min")
                
                if latency_red > TPCDS_BASELINE_LATENCY_REDUCTION:
                    print(f"     🎉 BEATS autoadmin BASELINE ({TPCDS_BASELINE_LATENCY_REDUCTION*100:.1f}%)!")
                elif latency_red < 0:
                    print(f"     ⚠️ WORSE than no indexes!")
                    latency_penalty_applied = True
            else:
                print(f"  ⚠️ TPC-DS latency validation error: {latency_validation_result.get('error')}")
        
        # === TPC-H LATENCY VALIDATION ===
        elif (ENABLE_LATENCY_VALIDATION and 
            benchmark == "tpch" and 
            constraint_score > 0):  # Only validate valid programs
            
            # Always run latency validation for TPC-H when enabled
            # (Cost model has poor correlation with latency for TPC-H)
            print(f"🔬 Running latency validation...")
            
            latency_validation_result = run_latency_validation(indexes_data, benchmark="tpch")
            
            if "error" not in latency_validation_result:
                latency_red = latency_validation_result["latency_reduction"]
                latency_reduction_actual = latency_red
                val_time = latency_validation_result.get("validation_time", 0)
                
                # Stricter validation for TPC-H:
                # 1. If latency < 0 (worse than no indexes): ZERO score
                # 2. If latency < threshold: heavy penalty
                # 3. If TPCH_LATENCY_PRIMARY: use latency as the score
                
                if latency_red < LATENCY_DEGRADATION_THRESHOLD:
                    # Latency got worse - apply strict penalty
                    old_score = combined_score
                    if LATENCY_PENALTY_FACTOR == 0.0:
                        combined_score = 0.0
                    else:
                        combined_score *= LATENCY_PENALTY_FACTOR
                    latency_penalty_applied = True
                    print(f"  ❌ LATENCY DEGRADATION: {latency_red*100:.1f}% (threshold: {LATENCY_DEGRADATION_THRESHOLD*100:.1f}%)")
                    print(f"     Penalty applied: {old_score:.3f} → {combined_score:.3f}")
                else:
                    # Latency is positive - potential success!
                    if TPCH_LATENCY_PRIMARY:
                        # Use latency reduction as the PRIMARY score for TPC-H
                        # This ensures evolution optimizes for actual performance
                        old_score = combined_score
                        combined_score = max(0.0, latency_red)  # Use latency as score
                        print(f"  📊 Using LATENCY as primary metric: {latency_red*100:.1f}%")
                        print(f"     (cost-based score was: {old_score:.3f})")
                        if latency_red > TPCH_BASELINE_LATENCY_REDUCTION:
                            print(f"     🎉 BEATS BASELINE ({TPCH_BASELINE_LATENCY_REDUCTION*100:.1f}%)!")
                    else:
                        print(f"  ✅ Latency validation passed: {latency_red*100:.1f}% reduction")
                        if latency_red > TPCH_BASELINE_LATENCY_REDUCTION:
                            print(f"     🎉 BEATS BASELINE ({TPCH_BASELINE_LATENCY_REDUCTION*100:.1f}%)!")
                    
                    print(f"     Validation time: {val_time:.1f}s")
            else:
                print(f"  ⚠️ Latency validation error: {latency_validation_result.get('error')}")
        
        # === JOB LATENCY VALIDATION ===
        # When USE_JOB_LATENCY_FITNESS is enabled, use actual latency as fitness
        elif (USE_JOB_LATENCY_FITNESS and 
            benchmark == "job" and 
            constraint_score > 0):  # Only validate valid programs
            
            skip_msg = f"skipping queries {JOB_SKIP_QUERIES}" if JOB_SKIP_QUERIES else "all 33 queries"
            print(f"🔬 Running JOB latency validation ({skip_msg})...")
            
            latency_validation_result = run_job_latency_validation(
                indexes_data, 
                benchmark="job",
                skip_queries=JOB_SKIP_QUERIES,
                query_timeout=JOB_QUERY_TIMEOUT
            )
            
            if "error" not in latency_validation_result:
                latency_red = latency_validation_result["latency_reduction"]
                latency_reduction_actual = latency_red
                val_time = latency_validation_result.get("validation_time", 0)
                queries_eval = latency_validation_result.get("queries_evaluated", 0)
                
                # Use latency reduction as the PRIMARY score for JOB
                old_score = combined_score
                combined_score = max(0.0, latency_red)  # Use latency as score
                
                print(f"  📊 JOB LATENCY FITNESS:")
                print(f"     Queries evaluated: {queries_eval}/33")
                print(f"     Latency reduction: {latency_red*100:.1f}%")
                print(f"     (cost-based score was: {old_score*100:.1f}%)")
                print(f"     Validation time: {val_time:.1f}s")
                
                if latency_red > JOB_BASELINE_LATENCY_REDUCTION:
                    print(f"     🎉 BEATS extend BASELINE ({JOB_BASELINE_LATENCY_REDUCTION*100:.1f}%)!")
                elif latency_red < 0:
                    print(f"     ⚠️ WORSE than no indexes!")
                    latency_penalty_applied = True
            else:
                print(f"  ⚠️ JOB latency validation error: {latency_validation_result.get('error')}")
        
        constraints_info = BENCHMARK_CONSTRAINTS.get(benchmark, {})
        num_queries = constraints_info.get("num_queries", "?")
        
        # Build status message
        weighted_flag = "⚖️ weighted" if use_weighted else "standard"
        print(f"✅ FULL Evaluation ({benchmark.upper()}, {num_queries} queries, {weighted_flag}): "
              f"constraint={constraint_score:.3f}, cost_reduction={cost_reduction:.3f}, "
              f"time={selection_time:.2f}s, score={combined_score:.3f}")
        
        result = {
            "query_cost_reduction": float(standard_reduction),  # Always report standard for comparison
            "weighted_cost_reduction": float(weighted_reduction),  # Reliability-weighted reduction
            "constraint_score": float(constraint_score),
            "storage_used_mb": float(storage_used_mb),
            "num_indexes": float(num_indexes),
            "selection_time": float(selection_time),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
            "used_reliability_weights": use_weighted,  # Flag to indicate which scoring was used
        }
        
        # Add latency validation results if performed
        if latency_validation_result is not None:
            # Determine baseline threshold based on benchmark
            if benchmark == "tpch":
                baseline_threshold = TPCH_BASELINE_LATENCY_REDUCTION
                used_as_primary = TPCH_LATENCY_PRIMARY
            elif benchmark == "job":
                baseline_threshold = JOB_BASELINE_LATENCY_REDUCTION
                used_as_primary = USE_JOB_LATENCY_FITNESS
            elif benchmark in ("tpcds", "tpcds_27", "tpcds_77"):
                baseline_threshold = TPCDS_BASELINE_LATENCY_REDUCTION
                used_as_primary = should_use_tpcds_latency(benchmark)
            else:
                baseline_threshold = 0.0
                used_as_primary = False
            
            result["latency_validation"] = {
                "performed": True,
                "latency_reduction": latency_validation_result.get("latency_reduction", 0.0),
                "validation_time": latency_validation_result.get("validation_time", 0.0),
                "penalty_applied": latency_penalty_applied,
                "baseline_threshold": baseline_threshold,
                "used_as_primary": used_as_primary,
            }
            # Also add latency_reduction as top-level metric for easy access
            if latency_reduction_actual is not None:
                result["latency_reduction"] = float(latency_reduction_actual)
        else:
            result["latency_validation"] = {"performed": False}
        
        return result
    
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "query_cost_reduction": 0.0,
            "weighted_cost_reduction": 0.0,
            "constraint_score": 0.0,
            "storage_used_mb": 0.0,
            "num_indexes": 0.0,
            "selection_time": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "used_reliability_weights": False,
            "error": str(e)
        }


# Cascade evaluation functions for faster filtering
def evaluate_stage1(program_path, benchmark=None):
    """
    Stage 1: Quick validation check (no timeout).
    Filters out programs that fail fast or violate basic constraints.
    Uses simplified scoring to quickly identify promising programs.
    """
    # Read benchmark from env var if not explicitly passed
    if benchmark is None:
        benchmark = os.environ.get("EVAL_BENCHMARK", "tpch")
    
    try:
        # No timeout for full evaluation
        indexes_data, selection_time, baseline_cost, optimized_cost = run_program(
            program_path, benchmark=benchmark
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
        combined_score = 0.0 if constraint_score == 0.0 else cost_reduction
        
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


def evaluate_stage2(program_path, benchmark=None):
    """
    Stage 2: Full evaluation.
    Only called on programs that pass stage 1.
    """
    # Read benchmark from env var if not explicitly passed
    if benchmark is None:
        benchmark = os.environ.get("EVAL_BENCHMARK", "tpch")
    
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
    Evaluate program on ALL benchmarks with FULL workloads (tpch, tpcds, job).
    
    This is the FULL evaluator version - runs all 3 benchmarks with complete query sets:
    - TPC-H: 18 queries
    - TPC-DS: 79 queries (full, not 30)
    - JOB: 33 queries (full, not 10)
    
    Args:
        program_path: Path to the program file to evaluate
        parallel: If True, run benchmarks in parallel (default: True)
        
    Returns:
        Dictionary with aggregated metrics and per-benchmark breakdown
    """
    start_time = time.time()
    results_by_benchmark = {}
    
    # Calculate total queries for logging
    total_queries = sum(BENCHMARK_CONSTRAINTS.get(bm, {}).get("num_queries", 0) for bm in ALL_BENCHMARKS)
    
    if parallel:
        # Run all benchmarks in parallel using ThreadPoolExecutor
        # (ThreadPoolExecutor is safe here since actual work happens in subprocess via run_with_timeout)
        print(f"🚀 FULL EVALUATOR: Running {len(ALL_BENCHMARKS)} benchmarks ({total_queries} total queries) in parallel...")
        with ThreadPoolExecutor(max_workers=len(ALL_BENCHMARKS)) as executor:
            futures = {
                executor.submit(_evaluate_single_benchmark, (program_path, bm)): bm 
                for bm in ALL_BENCHMARKS
            }
            for future in as_completed(futures):
                benchmark, result = future.result()
                results_by_benchmark[benchmark] = result
                num_q = BENCHMARK_CONSTRAINTS.get(benchmark, {}).get("num_queries", "?")
                print(f"  ✓ {benchmark.upper()} ({num_q} queries): score={result.get('combined_score', 0):.3f}")
    else:
        # Run sequentially
        print(f"🔄 FULL EVALUATOR: Running {len(ALL_BENCHMARKS)} benchmarks ({total_queries} total queries) sequentially...")
        for benchmark in ALL_BENCHMARKS:
            num_q = BENCHMARK_CONSTRAINTS.get(benchmark, {}).get("num_queries", "?")
            print(f"  Running {benchmark.upper()} ({num_q} queries)...")
            results_by_benchmark[benchmark] = evaluate(program_path, benchmark=benchmark)
            print(f"  ✓ {benchmark.upper()}: score={results_by_benchmark[benchmark].get('combined_score', 0):.3f}")
    
    total_time = time.time() - start_time
    
    # Aggregate scores - INCLUDE ALL benchmarks, errors/timeouts count as 0
    # This prevents rewarding programs that timeout on some benchmarks
    all_scores = [r.get("combined_score", 0.0) for r in results_by_benchmark.values()]
    all_cost_reductions = [r.get("query_cost_reduction", 0.0) for r in results_by_benchmark.values()]
    
    # Average over ALL benchmarks (not just successful ones)
    avg_score = sum(all_scores) / len(ALL_BENCHMARKS)
    avg_cost_reduction = sum(all_cost_reductions) / len(ALL_BENCHMARKS)
    
    # Count successful benchmarks for logging
    num_successful = sum(1 for r in results_by_benchmark.values() if "error" not in r)
    
    # Check if ALL benchmarks succeeded with constraint satisfaction
    all_constraints_satisfied = (
        num_successful == len(ALL_BENCHMARKS) and
        all(r.get("constraint_score", 0) == 1.0 for r in results_by_benchmark.values())
    )
    
    print(f"\n📊 Aggregate: avg_score={avg_score:.3f}, avg_cost_reduction={avg_cost_reduction:.3f}, "
          f"benchmarks={num_successful}/{len(ALL_BENCHMARKS)}, time={total_time:.1f}s")
    
    return {
        # Aggregated metrics
        "combined_score": float(avg_score),
        "query_cost_reduction": float(avg_cost_reduction),
        "constraint_score": 1.0 if all_constraints_satisfied else 0.0,
        "eval_time": float(total_time),
        # Per-benchmark breakdown (all 3 benchmarks for FULL evaluator)
        "tpch": results_by_benchmark.get("tpch", {}),
        "tpcds": results_by_benchmark.get("tpcds", {}),
        "job": results_by_benchmark.get("job", {}),
    }


def main():
    """Command-line interface for FULL evaluator (all queries, all benchmarks)."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="FULL Evaluator: Evaluate index selection on complete workloads (all queries)."
    )
    parser.add_argument("program_path", help="Path to the program file to evaluate.")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Evaluation stage (for cascade)")
    parser.add_argument("--benchmark", choices=["tpch", "tpcds", "tpcds_27", "tpcds_77", "job", "dsb", "tpch_skew", "tpch_sf10", "tpcds_sf10", "all"], default=None,
                       help="Benchmark workload to evaluate on (default: tpch, 'all' runs tpch/tpcds/job in parallel, 'tpcds_27' uses 27 reliable queries, 'tpcds_77' uses 77 queries with latency fitness)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run benchmarks sequentially instead of parallel (only with --benchmark all)")
    args = parser.parse_args()
    
    # Support environment variable for benchmark (for OpenEvolve integration)
    # Priority: CLI arg > env var > default (tpch)
    benchmark = args.benchmark or os.environ.get("EVAL_BENCHMARK", "tpch")
    
    if benchmark == "all":
        print(f"FULL EVALUATOR: Running ALL {len(ALL_BENCHMARKS)} benchmarks with FULL workloads ({'sequential' if args.sequential else 'parallel'})...")
        results = evaluate_all_benchmarks(args.program_path, parallel=not args.sequential)
    elif args.stage == 1:
        print(f"Evaluating on benchmark: {benchmark.upper()} (Stage 1)")
        results = evaluate_stage1(args.program_path, benchmark=benchmark)
    elif args.stage == 2:
        print(f"Evaluating on benchmark: {benchmark.upper()} (Stage 2)")
        results = evaluate_stage2(args.program_path, benchmark=benchmark)
    else:
        print(f"Evaluating on benchmark: {benchmark.upper()}")
        results = evaluate(args.program_path, benchmark=benchmark)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

