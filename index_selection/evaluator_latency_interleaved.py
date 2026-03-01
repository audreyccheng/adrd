#!/usr/bin/env python3
"""
Latency-based evaluator for OpenEvolve - INTERLEAVED WARMUP VERSION.

This evaluator uses ACTUAL query latency instead of optimizer cost estimates.

KEY DIFFERENCE FROM evaluator_latency.py:
- Warmup and benchmark are INTERLEAVED per query:
  warmup Q1 → benchmark Q1 → warmup Q2 → benchmark Q2 → ...
- This ensures index pages are fresh in cache when measured
- Original version: warmup ALL queries, then benchmark ALL queries

SCORING: combined_score = -optimized_latency_seconds
- Lower latency = higher score (OpenEvolve maximizes)
- 10s latency → score = -10 (good)
- 100s latency → score = -100 (bad)
- Perfect differentiation: 10× better latency = 10× better score

Design:
- Real index creation + ANALYZE per iteration
- Full query execution for latency measurement
- NO baseline needed (score is just negative latency)

Expected runtime: ~3-5 min per iteration

VARIANCE REDUCTION (Environment Variables):
- LATENCY_WARMUP_INDEXES=1 (default): Run query once before measurement
  to warm index cache. Reduces variance from ~20% to ~10%.
  Set to 0 to disable (faster but noisier).
  
- LATENCY_NUM_RUNS=1 (default): Number of times to execute each query.
  Set to 3 for final evaluation to get median latency (reduces variance to ~3%).
  Trade-off: 3 runs = 3x measurement time.

NOISE REDUCTION (Environment Variables):
- EVAL_PREWARM=1 (default): Use pg_prewarm to pin all tables and indexes
  in shared_buffers before measurement. Eliminates cache eviction noise.
  Set to 0 to disable.

- EVAL_SUPPRESS_BG=1 (default): Force CHECKPOINT and disable autovacuum
  on all tables before measurement. Prevents background I/O interference.
  Set to 0 to disable.

SPEED (Environment Variables):
- EVAL_TOP_QUERIES=0 (default: 0 = all): Only measure the top N queries
  by frequency. Top 15-20 queries typically cover 75%+ of total frequency
  weight. Cuts measurement time by ~40% while preserving ranking signal.
  Set to 0 or unset to measure all queries.

RECOMMENDED SETTINGS:
- During evolution: LATENCY_WARMUP_INDEXES=1 LATENCY_NUM_RUNS=1 (default, fast)
- Final comparison: LATENCY_WARMUP_INDEXES=1 LATENCY_NUM_RUNS=3 (reliable)
- Fast evolution: EVAL_TOP_QUERIES=20 EVAL_PREWARM=1 EVAL_SUPPRESS_BG=1
"""

import configparser
import importlib.util
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

# Ctrl+C: set flag so we exit between queries (shell also kills our process group if we don't exit fast)
_interrupt_requested = False

def _sigint_handler(signum, frame):
    global _interrupt_requested
    _interrupt_requested = True

def _check_interrupt():
    """Exit with 130 if user pressed Ctrl+C (so shell sees we stopped on INT)."""
    if _interrupt_requested:
        sys.exit(130)

# Project root for imports
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", "Index_EAB")
sys.path.insert(0, PROJECT_ROOT)

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index

# Benchmark configurations (SF=1, per paper)
BENCHMARK_CONFIG = {
    "tpch": {
        "db_name": "benchbase",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpch.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpch_work_temp_multi_freq.json",
        "num_queries": 18,
        "budget_mb": 500,
        "skip_queries": [],  # No queries to skip for TPC-H
    },
    "tpcds": {
        "db_name": "benchbase_tpcds",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_tpcds.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/tpcds_work_temp_multi_freq.json",
        "num_queries": 79,
        "budget_mb": 500,
        # Default skip_queries: [0, 5, 56, 63] - extremely slow with no index improvement
        # Query 0: Very slow, minimal improvement
        # Query 5: ~32 min, 1.01x speedup, freq=75
        # Query 56: ~11.6 min, 1.00x speedup (actually worse with indexes), freq=16
        # Query 63: Very slow, minimal improvement
        # Can be overridden via TPCDS_SKIP_QUERIES or SKIP_QUERIES environment variable
        "skip_queries": [],
        #"skip_queries": [0, 5, 56, 63],
    },
    "job": {
        "db_name": "benchbase_job",
        "schema_file": f"{PROJECT_ROOT}/configuration_loader/database/schema_job.json",
        "workload_file": f"{PROJECT_ROOT}/workload_generator/template_based/job_work_temp_multi_freq.json",
        "num_queries": 33,
        "budget_mb": 2000,
        "skip_queries": [],  # No queries to skip for JOB
    },
}

def log(msg):
    """Print with flush for real-time output.
    
    Can be disabled by setting EVALUATOR_QUIET=1 environment variable.
    By default, logging is enabled.
    
    Handles BrokenPipeError gracefully - this can happen when the parent
    process (OpenEvolve) closes stdout/stderr pipes, especially in parallel mode.
    """
    # Check if quiet mode is enabled (default: verbose/on)
    if os.environ.get("EVALUATOR_QUIET", "0") != "1":
        try:
            print(msg, flush=True)
        except BrokenPipeError:
            # Parent process closed the pipe - this is OK, just continue silently
            # This can happen in parallel evaluation mode when pipes are redirected
            pass
        except OSError:
            # Handle other pipe-related errors (e.g., EPIPE on some systems)
            pass


def load_workload(benchmark: str) -> Tuple[Workload, PostgresDatabaseConnector, list]:
    """Load workload and create database connector."""
    config = BENCHMARK_CONFIG[benchmark]
    
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host=os.environ.get("PGHOST", "127.0.0.1"), 
        port=os.environ.get("PGPORT", "5432"),
        db_name=config["db_name"], 
        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), 
        password=""
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


def measure_query_latency_single(query, connector: PostgresDatabaseConnector) -> float:
    """Execute a query ONCE and return actual execution time in milliseconds."""
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
            return 0.0


def measure_query_latency(query, connector: PostgresDatabaseConnector, num_runs: int = 1) -> float:
    """
    Execute a query and return actual execution time in milliseconds.
    
    Args:
        query: Query to execute
        connector: Database connector
        num_runs: Number of times to execute (default: 1). 
                  If > 1, returns MEDIAN of all runs for stability.
                  
    Returns:
        Execution time in milliseconds (median if num_runs > 1)
    """
    if num_runs <= 1:
        return measure_query_latency_single(query, connector)
    
    # Multiple executions: collect all latencies and return median
    latencies = []
    for _ in range(num_runs):
        latency = measure_query_latency_single(query, connector)
        latencies.append(latency)
    
    # Return median (more robust to outliers than mean)
    latencies.sort()
    mid = len(latencies) // 2
    if len(latencies) % 2 == 0:
        return (latencies[mid - 1] + latencies[mid]) / 2.0
    else:
        return latencies[mid]


def warmup_cache(benchmark: str, skip_queries: List[int] = None, force: bool = False):
    """
    Warmup database cache by running all queries once (without indexes).
    
    Call this ONCE at the start of an evolution run to ensure consistent measurements.
    Without warmup, early iterations will have artificially high latencies.
    
    Args:
        benchmark: Benchmark name
        skip_queries: Queries to skip
        force: Unused (kept for API compatibility)
    """
    port = os.environ.get("PGPORT", "5432")
    
    config = BENCHMARK_CONFIG[benchmark]
    
    if skip_queries is None:
        skip_queries = config.get("skip_queries", [])
    
    log(f"\n🔥 Warming up cache for {benchmark}...")
    workload, connector, _ = load_workload(benchmark)
    
    try:
        # Clean up any leftover indexes
        cleanup_indexes(connector)
        
        # Note: ANALYZE is done AFTER warmup (not here) to ensure fresh stats right before index selection
        filtered_count = len([q for i, q in enumerate(workload.queries) if i not in skip_queries])
        log(f"   Running {filtered_count} queries to warm cache...")
        
        warmup_start = time.time()
        executed = 0
        for i, query in enumerate(workload.queries):
            _check_interrupt()
            if i in skip_queries:
                continue
            try:
                measure_query_latency(query, connector)
            except SystemExit:
                raise
            except:
                pass
            executed += 1
            if executed % 10 == 0 or executed == filtered_count:
                log(f"   Progress: {executed}/{filtered_count}")
        
        warmup_elapsed = time.time() - warmup_start
        log(f"✅ Warmup complete ({warmup_elapsed:.1f}s)")
        
    finally:
        connector.close()


def measure_workload_latency(
    workload: Workload, 
    connector: PostgresDatabaseConnector,
    skip_queries: List[int] = None,
    num_runs: int = 1,
    silent: bool = False
) -> float:
    """
    Measure total frequency-weighted latency for workload (NO warmup, just measurement).
    
    Args:
        workload: Workload to measure
        connector: Database connector
        skip_queries: Query indices to skip
        num_runs: Number of times to execute each query (default: 1).
                  If > 1, uses MEDIAN latency for each query.
        silent: If True, suppress progress logging (used for warmup pass)
    
    Returns:
        Total frequency-weighted latency in milliseconds
    """
    if skip_queries is None:
        skip_queries = []
    
    total_latency = 0.0
    skip_set = set(skip_queries)
    filtered_queries = [(i, q) for i, q in enumerate(workload.queries) if i not in skip_set]
    filtered_count = len(filtered_queries)
    
    measure_start = time.time()
    measure_start_time = datetime.now().strftime("%H:%M:%S")
    
    if not silent:
        runs_str = f" x{num_runs}" if num_runs > 1 else ""
        log(f"     Measuring latency ({filtered_count} queries{runs_str})...")
        log(f"     Started at {measure_start_time}")
    
    executed = 0
    
    for i, query in filtered_queries:
        _check_interrupt()
        latency_ms = measure_query_latency(query, connector, num_runs=num_runs)
        total_latency += latency_ms * query.frequency
        executed += 1
        
        if not silent and (executed % 10 == 0 or executed == filtered_count):
            log(f"       Progress: {executed}/{filtered_count} (total latency so far: {total_latency/1000:.1f}s)")
    
    measure_elapsed = time.time() - measure_start
    measure_end_time = datetime.now().strftime("%H:%M:%S")
    
    if not silent:
        log(f"     Measurement complete at {measure_end_time} (took {measure_elapsed:.1f}s)")
        log(f"     Total weighted latency: {total_latency/1000:.2f}s")
    
    return total_latency


def _compute_latency_stats(run_totals_ms: List[float]) -> Dict:
    """Compute summary statistics over per-run total latencies."""
    n = len(run_totals_ms)
    sorted_vals = sorted(run_totals_ms)
    mid = n // 2
    if n % 2 == 0:
        median_ms = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median_ms = sorted_vals[mid]
    mean_ms = sum(run_totals_ms) / n
    min_ms = sorted_vals[0]
    max_ms = sorted_vals[-1]
    if n > 1:
        variance = sum((x - mean_ms) ** 2 for x in run_totals_ms) / (n - 1)
        std_ms = math.sqrt(variance)
    else:
        std_ms = 0.0
    std_pct = (std_ms / mean_ms * 100) if mean_ms > 0 else 0.0
    return {
        "run_totals_ms": list(run_totals_ms),
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_pct": std_pct,
        "num_runs": n,
    }


def measure_workload_latency_interleaved(
    workload: Workload, 
    connector: PostgresDatabaseConnector,
    skip_queries: List[int] = None,
    num_runs: int = 1,
    do_warmup: bool = True
) -> Tuple[float, Optional[Dict]]:
    """
    Measure total frequency-weighted latency for workload with INTERLEAVED warmup.
    
    For each query: warmup (1 run) → benchmark (num_runs) → next query
    
    This ensures index pages are fresh in buffer cache when measured,
    avoiding cache eviction issues that can occur with sequential warmup/benchmark.
    
    Args:
        workload: Workload to measure
        connector: Database connector
        skip_queries: Query indices to skip
        num_runs: Number of times to execute each query for benchmark (default: 1).
                  If > 1, uses MEDIAN latency for each query.
        do_warmup: If True, run each query once before benchmark measurement.
                   If False, skip warmup (just benchmark).
    
    Returns:
        Tuple of (total_latency_ms, run_stats):
        - total_latency_ms: Frequency-weighted latency in ms (sum of per-query medians)
        - run_stats: Dict with per-run totals and stats (None when num_runs <= 1)
    """
    if skip_queries is None:
        skip_queries = []
    
    total_latency = 0.0
    skip_set = set(skip_queries)
    filtered_queries = [(i, q) for i, q in enumerate(workload.queries) if i not in skip_set]
    filtered_count = len(filtered_queries)
    
    measure_start = time.time()
    measure_start_time = datetime.now().strftime("%H:%M:%S")
    
    warmup_str = "warmup+" if do_warmup else ""
    runs_str = f" x{num_runs}" if num_runs > 1 else ""
    log(f"     Measuring latency INTERLEAVED ({filtered_count} queries, {warmup_str}{runs_str} per query)...")
    log(f"     Started at {measure_start_time}")
    
    executed = 0
    track_runs = num_runs > 1
    run_totals = [0.0] * num_runs if track_runs else None
    
    for i, query in filtered_queries:
        _check_interrupt()
        if do_warmup:
            _ = measure_query_latency_single(query, connector)
        
        if track_runs:
            query_latencies = []
            for r in range(num_runs):
                latency_ms = measure_query_latency_single(query, connector)
                query_latencies.append(latency_ms)
                run_totals[r] += latency_ms * query.frequency
            query_latencies.sort()
            mid = len(query_latencies) // 2
            if len(query_latencies) % 2 == 0:
                median_latency = (query_latencies[mid - 1] + query_latencies[mid]) / 2.0
            else:
                median_latency = query_latencies[mid]
            total_latency += median_latency * query.frequency
        else:
            latency_ms = measure_query_latency_single(query, connector)
            total_latency += latency_ms * query.frequency
        
        executed += 1
        if executed % 10 == 0 or executed == filtered_count:
            log(f"       Progress: {executed}/{filtered_count} (total latency so far: {total_latency/1000:.1f}s)")
    
    measure_elapsed = time.time() - measure_start
    measure_end_time = datetime.now().strftime("%H:%M:%S")
    
    log(f"     Measurement complete at {measure_end_time} (took {measure_elapsed:.1f}s)")
    log(f"     Total weighted latency (median-of-runs): {total_latency/1000:.2f}s")
    
    run_stats = None
    if track_runs:
        run_stats = _compute_latency_stats(run_totals)
        log(f"     Per-run totals (s): {', '.join(f'{t/1000:.2f}' for t in run_totals)}")
        log(f"     Latency (s):  min={run_stats['min_ms']/1000:.2f}  max={run_stats['max_ms']/1000:.2f}  "
            f"mean={run_stats['mean_ms']/1000:.2f}  median={run_stats['median_ms']/1000:.2f}  "
            f"std={run_stats['std_ms']/1000:.2f}")
        log(f"     Std/mean:     {run_stats['std_pct']:.1f}%")
    
    return total_latency, run_stats


def create_real_indexes(indexes: Set[Index], connector: PostgresDatabaseConnector) -> Tuple[float, List[str]]:
    """Create real B-tree indexes in PostgreSQL."""
    index_names = []
    total_time = 0.0
    total_indexes = len(indexes)
    
    for i, idx in enumerate(indexes, 1):
        table_name = idx.table().name if hasattr(idx.table(), 'name') else str(idx.table())
        col_names = "_".join([c.name for c in idx.columns])
        index_name = f"evolve_{table_name}_{col_names}"[:63]
        
        columns_str = ", ".join([c.name for c in idx.columns])
        create_sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_name} ({columns_str})'
        
        try:
            log(f"     Creating index {i}/{total_indexes}: {table_name}({columns_str[:50]}{'...' if len(columns_str) > 50 else ''})...")
            start = time.time()
            connector.exec_only(create_sql)
            connector.commit()
            elapsed = time.time() - start
            total_time += elapsed
            index_names.append(index_name)
            log(f"     ✓ Index {i}/{total_indexes} created in {elapsed:.1f}s")
        except Exception as e:
            log(f"     ⚠️  Index {i}/{total_indexes} failed: {str(e)[:50]}")
    
    # ANALYZE is critical - ensures optimizer knows about new indexes
    log(f"     Running ANALYZE...")
    analyze_start = time.time()
    connector.exec_only("ANALYZE")
    connector.commit()
    analyze_elapsed = time.time() - analyze_start
    log(f"     ✓ ANALYZE complete ({analyze_elapsed:.1f}s)")
    
    return total_time, index_names


def cleanup_indexes(connector: PostgresDatabaseConnector):
    """Clean up ALL evolve_* indexes."""
    try:
        result = connector.exec_fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname LIKE 'evolve_%'",
            one=False
        )
        if result:
            for row in result:
                connector.exec_only(f'DROP INDEX IF EXISTS "{row[0]}"')
                connector.commit()
    except:
        pass


def get_top_queries_by_frequency(workload: Workload, top_n: int, existing_skip: List[int] = None) -> List[int]:
    """
    Compute which query indices to skip so only the top-N by frequency are measured.
    
    Args:
        workload: Workload with queries
        top_n: Number of top-frequency queries to keep
        existing_skip: Query indices already marked for skipping
    
    Returns:
        Combined skip list (existing + low-frequency queries)
    """
    if existing_skip is None:
        existing_skip = []
    skip_set = set(existing_skip)
    
    # Build (index, frequency) for non-skipped queries
    query_freqs = [(i, q.frequency) for i, q in enumerate(workload.queries)
                   if i not in skip_set]
    
    # Sort by frequency descending, keep top N
    query_freqs.sort(key=lambda x: x[1], reverse=True)
    keep_indices = set(i for i, _ in query_freqs[:top_n])
    
    # Everything not in keep_indices and not already skipped gets added to skip
    all_indices = set(range(len(workload.queries)))
    combined_skip = sorted(all_indices - keep_indices)
    
    kept_freq = sum(f for i, f in query_freqs[:top_n])
    total_freq = sum(f for i, f in query_freqs)
    pct = (kept_freq / total_freq * 100) if total_freq > 0 else 0
    
    log(f"   📋 Query subset: keeping top {top_n} of {len(query_freqs)} queries "
        f"({pct:.0f}% of frequency weight)")
    
    return combined_skip


def prewarm_cache(connector: PostgresDatabaseConnector, benchmark: str, index_names: List[str] = None):
    """
    Pin all tables and indexes in shared_buffers using pg_prewarm.
    
    This eliminates cache eviction noise by ensuring all data is in memory
    before measurement. Requires pg_prewarm extension.
    
    Args:
        connector: Database connector
        benchmark: Benchmark name (for logging)
        index_names: List of created index names to also prewarm
    """
    # Ensure pg_prewarm extension exists
    try:
        connector.exec_only("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        connector.commit()
    except Exception as e:
        log(f"   ⚠️  Could not enable pg_prewarm: {str(e)[:80]}")
        return
    
    prewarm_start = time.time()
    
    # Get all tables in public schema and prewarm them
    tables_prewarmed = 0
    try:
        result = connector.exec_fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename",
            one=False
        )
        if result:
            for row in result:
                try:
                    connector.exec_only(f"SELECT pg_prewarm('{row[0]}')")
                    tables_prewarmed += 1
                except Exception:
                    pass  # Table may not support prewarm (e.g., temp tables)
    except Exception as e:
        log(f"   ⚠️  Could not list tables for prewarm: {str(e)[:80]}")
    
    # Prewarm created indexes
    indexes_prewarmed = 0
    if index_names:
        for idx_name in index_names:
            try:
                connector.exec_only(f"SELECT pg_prewarm('{idx_name}')")
                indexes_prewarmed += 1
            except Exception:
                pass  # Index may have been dropped or renamed
    
    prewarm_elapsed = time.time() - prewarm_start
    log(f"   ✓ pg_prewarm: {tables_prewarmed} tables + {indexes_prewarmed} indexes "
        f"pinned in shared_buffers ({prewarm_elapsed:.1f}s)")


def suppress_background_activity(connector: PostgresDatabaseConnector):
    """
    Force CHECKPOINT and disable autovacuum to reduce measurement noise.
    
    - CHECKPOINT flushes all dirty pages so no background writes during measurement.
    - Disabling autovacuum prevents vacuum from running mid-measurement.
    
    Call restore_background_activity() after measurement completes.
    """
    # Force checkpoint to flush all dirty pages
    try:
        connector.exec_only("CHECKPOINT")
        connector.commit()
        log(f"   ✓ Forced CHECKPOINT (dirty pages flushed)")
    except Exception as e:
        log(f"   ⚠️  CHECKPOINT failed: {str(e)[:50]}")
    
    # Disable autovacuum on all public tables
    tables_disabled = 0
    try:
        result = connector.exec_fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
            one=False
        )
        if result:
            for row in result:
                try:
                    connector.exec_only(
                        f"ALTER TABLE \"{row[0]}\" SET (autovacuum_enabled = false)"
                    )
                    tables_disabled += 1
                except Exception:
                    pass
            connector.commit()
            log(f"   ✓ Disabled autovacuum on {tables_disabled} tables")
    except Exception as e:
        log(f"   ⚠️  Could not disable autovacuum: {str(e)[:50]}")


def restore_background_activity(connector: PostgresDatabaseConnector):
    """Re-enable autovacuum on all public tables after measurement."""
    try:
        result = connector.exec_fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
            one=False
        )
        if result:
            for row in result:
                try:
                    connector.exec_only(
                        f"ALTER TABLE \"{row[0]}\" SET (autovacuum_enabled = true)"
                    )
                except Exception:
                    pass
            connector.commit()
    except Exception:
        pass  # Non-fatal, autovacuum will re-enable on next restart anyway


def refresh_database_statistics(benchmark: str):
    """
    Refresh database statistics by running ANALYZE.
    This ensures the query planner has accurate statistics for index selection.
    
    Call this before running index selection algorithms to ensure consistent results.
    """
    config = BENCHMARK_CONFIG[benchmark]
    
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host=os.environ.get("PGHOST", "127.0.0.1"), 
        port=os.environ.get("PGPORT", "5432"),
        db_name=config["db_name"], 
        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), 
        password=""
    )
    
    try:
        # Clean up any leftover indexes first
        cleanup_indexes(connector)
        
        # Run ANALYZE to refresh statistics
        log(f"   📊 Refreshing database statistics (ANALYZE)...")
        analyze_start = time.time()
        connector.exec_only("ANALYZE")
        connector.commit()
        analyze_elapsed = time.time() - analyze_start
        log(f"   ✓ Statistics refreshed ({analyze_elapsed:.1f}s)")
    except Exception as e:
        log(f"   ⚠️  Could not refresh statistics: {str(e)[:50]}")
    finally:
        connector.close()


def cleanup_hypothetical_indexes(benchmark: str):
    """
    Clean up all hypothetical indexes (hypopg) to prevent OID exhaustion.
    
    This is critical when evaluating many candidate indexes - hypopg has a limit
    on the number of hypothetical indexes it can create (~1000-2000).
    """
    config = BENCHMARK_CONFIG[benchmark]
    
    db_conf = configparser.ConfigParser()
    db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
    
    connector = PostgresDatabaseConnector(
        db_conf, autocommit=True,
        host=os.environ.get("PGHOST", "127.0.0.1"), 
        port=os.environ.get("PGPORT", "5432"),
        db_name=config["db_name"], 
        user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")), 
        password=""
    )
    
    try:
        # Reset all hypothetical indexes
        connector.drop_hypo_indexes()
        log(f"   🧹 Cleaned up hypothetical indexes (hypopg_reset)")
    except Exception as e:
        # Non-fatal - hypopg might not be enabled or already clean
        log(f"   ⚠️  Could not clean hypopg indexes: {str(e)[:50]}")
    finally:
        connector.close()


def run_selection_algorithm(program_path: str, benchmark: str) -> Tuple[Set[Index], float, float, float]:
    """Run a selection algorithm and return selected indexes."""
    os.environ["BENCHMARK"] = benchmark
    os.environ["FULL_WORKLOAD"] = "true"
    
    # CRITICAL: Refresh statistics before running algorithm
    # This ensures consistent index selection across different database states
    refresh_database_statistics(benchmark)
    
    # CRITICAL: Clean up hypothetical indexes before running algorithm
    # Prevents "hypopg: not more oid available" error when evaluating many candidates
    # Note: hypopg indexes are per-connection, but cleanup helps remove any stale indexes
    cleanup_hypothetical_indexes(benchmark)
    
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    try:
        selected_indexes, selection_time, baseline_cost, optimized_cost = program.run_index_selection()
    finally:
        # Clean up after algorithm completes (in case it left indexes behind)
        # This is especially important if the algorithm crashed or was interrupted
        cleanup_hypothetical_indexes(benchmark)
    
    return selected_indexes, selection_time, baseline_cost, optimized_cost


def evaluate(program_path: str, benchmark: str = None, skip_queries: List[int] = None) -> Dict:
    """
    Evaluate an evolved program using ACTUAL latency with INTERLEAVED warmup.
    
    NOTE: This function is called by OpenEvolve for each program evaluation.
    
    This is the main entry point for OpenEvolve.
    
    KEY DIFFERENCE: Uses interleaved warmup/benchmark per query:
    - warmup Q1 → benchmark Q1 → warmup Q2 → benchmark Q2 → ...
    - Ensures index pages are fresh in cache when measured
    
    Scoring: combined_score = -optimized_latency_seconds
    - 10s latency → score = -10 (better)
    - 100s latency → score = -100 (worse)
    - Perfect differentiation: 10× better latency = 10× better score
    
    Args:
        program_path: Path to the evolved program
        benchmark: Benchmark to evaluate on (default: from EVAL_BENCHMARK env var)
        skip_queries: List of query IDs to skip (for slow queries). 
                      If None, uses default from BENCHMARK_CONFIG.
    
    Returns:
        Dictionary with:
        - combined_score: -optimized_latency_seconds (OpenEvolve maximizes this)
        - optimized_latency_ms: Absolute query latency in milliseconds
        - constraint_score: 1.0 if valid, 0.0 if constraints violated
        - storage_used_mb, num_indexes, etc.
    """
    # Immediate flush to ensure OpenEvolve sees we're running
    import sys
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (BrokenPipeError, OSError):
        # Pipe closed - parent process may have stopped reading, continue anyway
        pass
    
    if benchmark is None:
        benchmark = os.environ.get("EVAL_BENCHMARK", "tpch")
    
    config = BENCHMARK_CONFIG[benchmark]
    
    # Use default skip_queries from config if not provided
    if skip_queries is None:
        # Check environment variable first (allows override from script/config)
        skip_queries_env = os.environ.get(f"{benchmark.upper()}_SKIP_QUERIES") or os.environ.get("SKIP_QUERIES")
        if skip_queries_env:
            # Parse comma-separated list: "0,5,56,63" -> [0, 5, 56, 63]
            try:
                skip_queries = [int(q.strip()) for q in skip_queries_env.split(",") if q.strip()]
            except ValueError:
                log(f"⚠️  Invalid SKIP_QUERIES format: {skip_queries_env}, using default")
                skip_queries = config.get("skip_queries", [])
        else:
            # Use default from BENCHMARK_CONFIG
            skip_queries = config.get("skip_queries", [])
    
    if skip_queries:
        log(f"⏭️  Skipping queries: {skip_queries}")
    BUDGET_MB = config["budget_mb"]
    MAX_INDEXES = 15
    
    try:
        start_time = time.time()
        eval_start_time = datetime.now().strftime("%H:%M:%S")
        
        # Extract program name for visibility
        program_name = os.path.basename(program_path)
        is_initial = "initial_program" in program_name
        program_type = "INITIAL" if is_initial else "EVOLVED"
        
        log(f"\n{'='*60}")
        log(f"🔬 EVALUATION STARTED at {eval_start_time} [{program_type}] (INTERLEAVED)")
        log(f"   Program: {program_name}")
        log(f"{'='*60}")
        
        # Step 1: Run selection algorithm
        log(f"\n📋 Step 1/3: Running index selection algorithm...")
        algo_start = time.time()
        selected_indexes, selection_time, baseline_cost, optimized_cost = run_selection_algorithm(
            program_path, benchmark
        )
        algo_elapsed = time.time() - algo_start
        
        num_indexes = len(selected_indexes)
        storage_bytes = sum(idx.estimated_size or 0 for idx in selected_indexes)
        storage_mb = storage_bytes / (1024 * 1024)
        
        log(f"   ✓ Selected {num_indexes} indexes, {storage_mb:.1f} MB (took {algo_elapsed:.1f}s)")
        log(f"   ✓ Cost estimate: {baseline_cost:.0f} → {optimized_cost:.0f} ({(1 - optimized_cost/baseline_cost)*100:.1f}% reduction)")
        
        # Step 2: Check constraints FIRST (before creating indexes)
        log(f"\n📋 Step 2/3: Checking constraints...")
        storage_violated = storage_mb > BUDGET_MB
        count_violated = num_indexes > MAX_INDEXES
        
        if storage_violated or count_violated:
            # HARD constraint violation - skip expensive latency measurement
            if storage_violated:
                log(f"   ❌ CONSTRAINT VIOLATION: Storage {storage_mb:.1f} MB > {BUDGET_MB} MB budget")
            if count_violated:
                log(f"   ❌ CONSTRAINT VIOLATION: {num_indexes} indexes > {MAX_INDEXES} max")
            log(f"   ⏭️  Skipping latency measurement, returning penalty score")
            return {
                "combined_score": -999999.0,  # Very negative = very bad
                "constraint_score": 0.0,
                "storage_used_mb": storage_mb,
                "num_indexes": num_indexes,
                "selection_time": selection_time,
                "optimized_latency_ms": 0,
                "error": "CONSTRAINT_VIOLATION",
            }
        log(f"   ✓ Constraints OK: {num_indexes} indexes, {storage_mb:.1f}/{BUDGET_MB} MB")
        
        # Step 3: Load workload, create indexes, and measure with INTERLEAVED warmup
        log(f"\n📋 Step 3/3: Creating {num_indexes} indexes and measuring latency (INTERLEAVED)...")
        workload, connector, _ = load_workload(benchmark)
        
        # Configuration for measurement reliability
        # LATENCY_NUM_RUNS: Number of times to execute each query (default: 1)
        #   - 1: Fast (for evolution iterations)
        #   - 3: Reliable (for final evaluation, uses median)
        # LATENCY_WARMUP_INDEXES: Whether to warmup before each query measurement (default: 1/true)
        #   - 1/true: Run query once before benchmark (interleaved)
        #   - 0/false: Skip warmup (faster but more variance)
        num_runs = int(os.environ.get("LATENCY_NUM_RUNS", "1"))
        do_index_warmup = os.environ.get("LATENCY_WARMUP_INDEXES", "1").lower() in ("1", "true", "yes")
        
        # Noise/speed reduction settings
        do_prewarm = os.environ.get("EVAL_PREWARM", "1").lower() in ("1", "true", "yes")
        do_suppress_bg = os.environ.get("EVAL_SUPPRESS_BG", "1").lower() in ("1", "true", "yes")
        top_queries = int(os.environ.get("EVAL_TOP_QUERIES", "0"))
        
        try:
            # Clean up any leftover indexes
            cleanup_indexes(connector)
            
            # Create real indexes
            creation_time, index_names = create_real_indexes(selected_indexes, connector)
            log(f"   ✓ Created {len(index_names)} indexes (took {creation_time:.1f}s)")
            
            # --- Noise reduction: suppress background activity ---
            if do_suppress_bg:
                log(f"\n   🔇 Suppressing background activity...")
                suppress_background_activity(connector)
            
            # --- Noise reduction: pin cache with pg_prewarm ---
            if do_prewarm:
                log(f"\n   🔥 Pinning data in shared_buffers (pg_prewarm)...")
                prewarm_cache(connector, benchmark, index_names)
            
            # --- Speed: query subset by frequency ---
            effective_skip = skip_queries
            if top_queries > 0 and top_queries < len(workload.queries):
                effective_skip = get_top_queries_by_frequency(
                    workload, top_queries, existing_skip=skip_queries
                )
            
            # Measure latency with INTERLEAVED warmup
            # For each query: warmup (if enabled) → benchmark
            num_queries = len([q for i, q in enumerate(workload.queries) if i not in (effective_skip or [])])
            warmup_str = "warmup+" if do_index_warmup else ""
            runs_str = f"x{num_runs}" if num_runs > 1 else "1 run"
            log(f"\n   📊 Measuring {num_queries} queries ({warmup_str}{runs_str} per query, INTERLEAVED)...")
            
            optimized_latency, run_stats = measure_workload_latency_interleaved(
                workload, connector, effective_skip, 
                num_runs=num_runs, 
                do_warmup=do_index_warmup
            )
            log(f"   ✓ Latency measurement complete: {optimized_latency/1000:.2f}s")
            
        finally:
            # Always cleanup (with timeout protection)
            try:
                # Re-enable autovacuum before dropping indexes
                if do_suppress_bg:
                    restore_background_activity(connector)
                
                log(f"\n🧹 Cleaning up indexes...")
                cleanup_start = time.time()
                cleanup_indexes(connector)
                cleanup_elapsed = time.time() - cleanup_start
                if cleanup_elapsed > 5.0:
                    log(f"   ⚠️  Cleanup took {cleanup_elapsed:.1f}s (unusually slow)")
                log(f"   ✓ Cleanup complete")
            except Exception as cleanup_err:
                log(f"   ⚠️  Cleanup error (non-fatal): {cleanup_err}")
            finally:
                try:
                    connector.close()
                except:
                    pass
        
        # Step 4: Calculate score (AFTER cleanup to avoid hanging)
        log(f"\n📊 Calculating final score...")
        # PRIMARY SCORE: Negative latency (lower latency = higher score)
        optimized_latency_seconds = optimized_latency / 1000.0
        combined_score = -optimized_latency_seconds  # e.g., 50s → -50, 10s → -10 (better)
        
        eval_time = time.time() - start_time
        eval_end_time = datetime.now().strftime("%H:%M:%S")
        
        log(f"\n{'='*60}")
        log(f"✅ EVALUATION COMPLETE at {eval_end_time} (took {eval_time:.1f}s)")
        log(f"   📊 Score: {combined_score:.2f} (latency: {optimized_latency_seconds:.2f}s)")
        if run_stats:
            log(f"   📊 Runs: min={run_stats['min_ms']/1000:.2f}s  max={run_stats['max_ms']/1000:.2f}s  "
                f"mean={run_stats['mean_ms']/1000:.2f}s  std={run_stats['std_ms']/1000:.2f}s  "
                f"std/mean={run_stats['std_pct']:.1f}%")
        log(f"   📦 Indexes: {num_indexes}, Storage: {storage_mb:.1f} MB")
        log(f"{'='*60}")
        
        result = {
            "combined_score": combined_score,  # PRIMARY: -latency_seconds (OpenEvolve maximizes)
            "optimized_latency_ms": optimized_latency,
            "optimized_latency_seconds": optimized_latency_seconds,
            "constraint_score": 1.0,
            "storage_used_mb": storage_mb,
            "num_indexes": num_indexes,
            "selection_time": selection_time,
            "index_creation_time": creation_time,
            "eval_time": eval_time,
            "query_cost_reduction": (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0.0,
        }
        if run_stats:
            result["latency_std_seconds"] = run_stats["std_ms"] / 1000.0
            result["latency_std_pct"] = run_stats["std_pct"]
            result["latency_mean_seconds"] = run_stats["mean_ms"] / 1000.0
            result["latency_min_seconds"] = run_stats["min_ms"] / 1000.0
            result["latency_max_seconds"] = run_stats["max_ms"] / 1000.0
            result["latency_per_run_seconds"] = [t / 1000.0 for t in run_stats["run_totals_ms"]]
        
        # Final flush to ensure OpenEvolve sees the result
        import sys
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            # Pipe closed - parent process may have stopped reading, continue anyway
            pass
        
        log(f"   🚀 Returning result to OpenEvolve...")
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
        
        return result
        
    except Exception as e:
        import traceback
        try:
            traceback.print_exc()
        except (BrokenPipeError, OSError):
            # Can't print traceback if pipe is closed - log to stderr as fallback
            try:
                import sys
                sys.stderr.write(f"Error in evaluator: {str(e)}\n")
                sys.stderr.flush()
            except:
                pass  # Even stderr is closed, just continue
        return {
            "combined_score": -999999.0,  # Very negative = very bad (error case)
            "constraint_score": 0.0,
            "error": str(e),
        }


if __name__ == "__main__":
    """Test the latency evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Latency-based evaluator for OpenEvolve (INTERLEAVED warmup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INTERLEAVED WARMUP:
  This version uses interleaved warmup/benchmark per query:
    warmup Q1 → benchmark Q1 → warmup Q2 → benchmark Q2 → ...
  
  This ensures index pages are fresh in buffer cache when measured.
  
VARIANCE REDUCTION OPTIONS:
  For reliable comparisons, use: --num-runs 3 --index-warmup
  
  Example (fast, for evolution):
    python evaluator_latency_interleaved.py -p program.py -b job --warmup
    
  Example (reliable, for final comparison):
    python evaluator_latency_interleaved.py -p program.py -b job --warmup --num-runs 3 --index-warmup

NOISE/SPEED REDUCTION:
  --top-queries N       Only measure top N queries by frequency (default: all)
  --prewarm             Pin all tables+indexes in cache via pg_prewarm (default: on)
  --no-prewarm          Disable pg_prewarm
  --suppress-bg         CHECKPOINT + disable autovacuum during measurement (default: on)
  --no-suppress-bg      Disable background suppression
  
  Example (fast evolution with noise reduction):
    python evaluator_latency_interleaved.py -p program.py -b job --warmup --top-queries 20
    
ENVIRONMENT VARIABLES:
  LATENCY_NUM_RUNS=N       Same as --num-runs
  LATENCY_WARMUP_INDEXES=1 Same as --index-warmup (default: enabled)
  EVAL_TOP_QUERIES=N       Same as --top-queries (default: 0 = all)
  EVAL_PREWARM=1           Same as --prewarm (default: enabled)
  EVAL_SUPPRESS_BG=1       Same as --suppress-bg (default: enabled)
        """
    )
    parser.add_argument("--program", "-p", help="Program to evaluate")
    parser.add_argument("--benchmark", "-b", default="tpch", help="Benchmark (tpch, tpcds, job)")
    parser.add_argument("--skip-queries", type=str, 
                        help="Comma-separated query IDs to skip (overrides default). Use 'none' to skip nothing.")
    parser.add_argument("--warmup", action="store_true", 
                        help="Warmup database cache (run once before evolution)")
    parser.add_argument("--force-warmup", action="store_true",
                        help="Force warmup even if done recently")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of times to execute each query (default: 1). Use 3 for reliable median.")
    parser.add_argument("--index-warmup", action="store_true", default=True,
                        help="Warmup index cache before each query (default: enabled). Reduces variance.")
    parser.add_argument("--no-index-warmup", action="store_true",
                        help="Disable index warmup (faster but higher variance)")
    parser.add_argument("--top-queries", type=int, default=0,
                        help="Only measure top N queries by frequency (default: 0 = all). Use 20 for ~40%% speedup.")
    parser.add_argument("--prewarm", action="store_true", default=True,
                        help="Pin all tables+indexes in shared_buffers via pg_prewarm (default: enabled)")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Disable pg_prewarm cache pinning")
    parser.add_argument("--suppress-bg", action="store_true", default=True,
                        help="CHECKPOINT + disable autovacuum during measurement (default: enabled)")
    parser.add_argument("--no-suppress-bg", action="store_true",
                        help="Disable background activity suppression")
    parser.add_argument("--show-config", action="store_true", help="Show benchmark configuration")
    
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _sigint_handler)
    
    if args.show_config:
        config = BENCHMARK_CONFIG.get(args.benchmark, {})
        log(f"\n📋 Configuration for {args.benchmark}:")
        for k, v in config.items():
            log(f"   {k}: {v}")
        sys.exit(0)
    
    # Set environment variables for variance reduction options
    if args.num_runs > 1:
        os.environ["LATENCY_NUM_RUNS"] = str(args.num_runs)
        log(f"📊 Using {args.num_runs} runs per query (median for stability)")
    
    if args.no_index_warmup:
        os.environ["LATENCY_WARMUP_INDEXES"] = "0"
        log(f"⚠️  Index warmup DISABLED (higher variance expected)")
    else:
        os.environ["LATENCY_WARMUP_INDEXES"] = "1"
        log(f"🔥 Index warmup ENABLED (INTERLEAVED: warmup+benchmark per query)")
    
    # Set noise/speed reduction env vars from CLI args
    if args.top_queries > 0:
        os.environ["EVAL_TOP_QUERIES"] = str(args.top_queries)
        log(f"📋 Query subset: measuring top {args.top_queries} queries by frequency")
    
    if args.no_prewarm:
        os.environ["EVAL_PREWARM"] = "0"
        log(f"⚠️  pg_prewarm DISABLED")
    else:
        os.environ["EVAL_PREWARM"] = "1"
        log(f"🔥 pg_prewarm ENABLED (pin tables+indexes in shared_buffers)")
    
    if args.no_suppress_bg:
        os.environ["EVAL_SUPPRESS_BG"] = "0"
        log(f"⚠️  Background suppression DISABLED")
    else:
        os.environ["EVAL_SUPPRESS_BG"] = "1"
        log(f"🔇 Background suppression ENABLED (CHECKPOINT + disable autovacuum)")
    
    # Handle skip_queries: CLI --skip-queries > env var > BENCHMARK_CONFIG default
    skip_queries = None  # Will use default from config
    if args.skip_queries:
        if args.skip_queries.lower() == 'none':
            skip_queries = []  # Explicitly skip nothing
        else:
            skip_queries = [int(q.strip()) for q in args.skip_queries.split(',')]
    else:
        skip_queries_env = os.environ.get(f"{args.benchmark.upper()}_SKIP_QUERIES") or os.environ.get("SKIP_QUERIES")
        if skip_queries_env:
            try:
                skip_queries = [int(q.strip()) for q in skip_queries_env.split(",") if q.strip()]
            except ValueError:
                log(f"⚠️  Invalid skip queries env var: {skip_queries_env}, using default")
                skip_queries = None
    
    # Show what will be skipped
    config = BENCHMARK_CONFIG.get(args.benchmark, {})
    default_skip = config.get("skip_queries", [])
    effective_skip = skip_queries if skip_queries is not None else default_skip
    if effective_skip:
        log(f"⏭️  Queries to skip: {effective_skip}")
    
    # Refresh statistics - timing depends on whether warmup is done
    # If warmup is skipped, do ANALYZE before (ensures fresh stats for algorithm)
    # If warmup is done, do ANALYZE after warmup (ensures fresh stats right before index selection)
    if not args.warmup and os.environ.get("REFRESH_STATS", "0") == "1":
        refresh_database_statistics(args.benchmark)
    
    # Run warmup if requested
    if args.warmup:
        warmup_cache(args.benchmark, skip_queries, force=args.force_warmup)
        # ANALYZE after warmup ensures fresh statistics right before index selection
        log(f"\n📊 Refreshing database statistics after warmup (ANALYZE)...")
        refresh_database_statistics(args.benchmark)
    
    if args.program:
        log(f"\n🔬 Evaluating {args.program} on {args.benchmark} (INTERLEAVED warmup)...")
        result = evaluate(args.program, args.benchmark, skip_queries)
        log(f"\n📊 Results:")
        for k, v in result.items():
            if isinstance(v, float):
                log(f"   {k}: {v:.4f}")
            else:
                log(f"   {k}: {v}")
