"""
Evaluator for OpenEvolve PBM policy optimization.

Evaluates evolved buffer replacement policies using the minimal_postgres_simulator.
Returns EvaluationResult with metrics AND artifacts for LLM feedback.
"""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

# Try to import EvaluationResult for artifact support
try:
    from openevolve.evaluation_result import EvaluationResult
    HAS_EVALUATION_RESULT = True
except ImportError:
    HAS_EVALUATION_RESULT = False
    class EvaluationResult:
        """Fallback when openevolve not installed"""
        def __init__(self, metrics, artifacts=None):
            self.metrics = metrics
            self.artifacts = artifacts or {}


class TimeoutError(Exception):
    pass


# Workload configurations for evolution
WORKLOAD_CONFIGS = {
    # Sequential microbench (BRIN bitmap scans)
    "default": {
        "workload": "sequential",
        "num_streams": 2,
        "queries_per_stream": 8,
        "table_scale": 0.40,
        "buffer_pool_mb": 256,
        "selectivity": 0.30,
    },
    "low_parallel": {
        "workload": "sequential",
        "num_streams": 1,
        "queries_per_stream": 16,
        "table_scale": 0.40,
        "buffer_pool_mb": 256,
        "selectivity": 0.30,
    },
    "high_parallel": {
        "workload": "sequential",
        "num_streams": 8,
        "queries_per_stream": 4,
        "table_scale": 0.40,
        "buffer_pool_mb": 256,
        "selectivity": 0.30,
    },
    "small_buffer": {
        "workload": "sequential",
        "num_streams": 2,
        "queries_per_stream": 8,
        "table_scale": 0.40,
        "buffer_pool_mb": 64,
        "selectivity": 0.30,
    },
    # TPC-H (analytical, sequential/bitmap scans)
    "tpch_mixed": {
        "workload": "tpch",
        "num_streams": 4,
        "table_scale": 0.10,
        "buffer_pool_mb": 512,
    },
    "tpch_full": {
        "workload": "tpch",
        "num_streams": 8,
        "table_scale": 0.05,
        "buffer_pool_mb": 256,
    },
    # Hard TPC-H for evolution (more eviction pressure)
    # 64MB buffer = 12.5% of data = forces more evictions
    "tpch_hard": {
        "workload": "tpch",
        "num_streams": 8,
        "table_scale": 0.05,
        "buffer_pool_mb": 64,  # 4x smaller than tpch_full
    },
    # Fast TPC-H for evolution (7x faster than tpch_full)
    # Use for rapid iteration, validate winners on tpch_full
    "tpch_fast": {
        "workload": "tpch",
        "num_streams": 4,
        "table_scale": 0.02,
        "buffer_pool_mb": 128,
    },
    # TPC-C (OLTP, point lookups)
    # Data size: ~37GB for 100W, so buffer must be large enough for reasonable hit rate
    "tpcc": {  # Default TPC-C: 100 warehouses, 8GB buffer (~20% of data)
        "workload": "tpcc",
        "num_warehouses": 100,
        "num_terminals": 10,
        "transactions_per_terminal": 50,
        "buffer_pool_mb": 8192,  # 8GB for ~50-60% hit rate
    },
    "tpcc_100w": {  # Same as tpcc but explicit name
        "workload": "tpcc",
        "num_warehouses": 100,
        "num_terminals": 10,
        "transactions_per_terminal": 50,
        "buffer_pool_mb": 8192,  # 8GB for ~50-60% hit rate
    },
    "tpcc_100w_small": {  # 100 warehouses, smaller buffer (stress test)
        "workload": "tpcc",
        "num_warehouses": 100,
        "num_terminals": 10,
        "transactions_per_terminal": 50,
        "buffer_pool_mb": 2048,  # 2GB = ~5% of data, lower hit rate
    },
    "tpcc_10w": {  # 10 warehouses with buffer pressure
        "workload": "tpcc",
        "num_warehouses": 10,
        "num_terminals": 5,
        "transactions_per_terminal": 100,
        "buffer_pool_mb": 128,  # 12.5% of ~1GB data - creates eviction pressure
    },
    # TPC-C with buffer pressure - tests eviction policy properly
    # 64MB buffer = 6.25% of ~1GB data, creates real eviction pressure
    # Shows dirty eviction rate (8-14%) unlike high-buffer configs
    "tpcc_10w_stress": {
        "workload": "tpcc",
        "num_warehouses": 10,
        "num_terminals": 5,
        "transactions_per_terminal": 100,
        "buffer_pool_mb": 64,  # 6.25% ratio - creates eviction pressure
    },
    # P3: Calibrated TPC-C config matching real PostgreSQL experiments (RUN_EXPS.md)
    # Real experiment: 100W, 60 terminals, 256MB buffer (2.5% of ~10GB data)
    # This creates similar buffer pressure to real PostgreSQL tests
    "tpcc_100w_calibrated": {
        "workload": "tpcc",
        "num_warehouses": 100,
        "num_terminals": 60,  # Match real experiment
        "transactions_per_terminal": 50,  # ~3000 txn total, matches ~3 min experiment
        "buffer_pool_mb": 256,  # Match real 2.5% buffer ratio
    },
    # Mixed TPC-H + TPC-C (interleaved)
    "mixed": {
        "workload": "mixed",
        "tpch_num_streams": 4,
        "tpch_table_scale": 0.03,
        "tpcc_num_warehouses": 10,
        "tpcc_num_terminals": 5,
        "tpcc_transactions_per_terminal": 100,
        "tpch_probability": 0.5,
        "buffer_pool_mb": 256,
    },
    # CH-benchmark: TPC-C + TPC-H-like queries on SAME schema (realistic HTAP)
    "chbenchmark": {
        "workload": "chbenchmark",
        "num_warehouses": 10,
        "num_terminals": 5,
        "tpcc_transactions_per_terminal": 200,
        "num_olap_streams": 2,
        "olap_probability": 0.05,  # 5% OLAP
        "buffer_pool_mb": 2400,  # 2.35GB - target ~60% hit rate
    },
    "chbenchmark_heavy_olap": {
        "workload": "chbenchmark",
        "num_warehouses": 10,
        "num_terminals": 5,
        "tpcc_transactions_per_terminal": 200,
        "num_olap_streams": 4,
        "olap_probability": 0.10,  # 10% OLAP
        "buffer_pool_mb": 2048,
    },
    "chbenchmark_stress": {
        "workload": "chbenchmark",
        "num_warehouses": 10,
        "num_terminals": 5,
        "tpcc_transactions_per_terminal": 200,
        "num_olap_streams": 2,
        "olap_probability": 0.05,
        "olap_batch_size": 1000,
        "buffer_pool_mb": 512,  # Smaller buffer = more eviction pressure
    },
    # CH-benchmark with REALISTIC 3:1 OLTP:OLAP terminal ratio
    # (Matches real PostgreSQL experiments from REMOTE_EXPS.md)
    #
    # Real setup: SF=100, 12 OLTP + 4 OLAP terminals (3:1 ratio), 256MB buffer
    # Observed: 72-87% hit rate, 15-30M disk reads per 5min run
    #
    # Terminal ratio 12:4 = 3:1 means:
    #   - 75% of terminal time goes to OLTP (12/16)
    #   - 25% of terminal time goes to OLAP (4/16)
    # But OLAP queries do ~10x more I/O per unit time, so ACCESS ratio is ~1:10
    #
    # Model: olap_probability=0.25 (25% OLAP time), olap_batch_size=40
    #        Every 4 OLTP accesses -> 40 OLAP accesses = 1:10 ratio
    #
    # IMPORTANT: Simulator hit rates will be LOWER than real PostgreSQL because:
    #   1. No synchronized seqscans (real PG shares I/O between concurrent scans)
    #   2. No OS page cache (real systems get additional caching)
    #   3. Simplified scan patterns (no query optimizer / index selection)
    # This is INTENTIONAL - creates more eviction pressure for policy testing.
    #
    # Recommended config for policy comparison: 10 warehouses (3.6GB data) + 2.4GB buffer
    # This gives ~45-55% hit rate which creates meaningful eviction pressure.
    "chbenchmark_3to1": {
        "workload": "chbenchmark",
        "num_warehouses": 10,            # 10W = ~3.6GB data (calibrated for hit rate)
        "num_terminals": 12,             # Match real 12 OLTP terminals
        "tpcc_transactions_per_terminal": 100,  # More transactions for longer run
        "num_olap_streams": 4,           # Match real 4 OLAP terminals
        "olap_probability": 0.25,        # 4/16 = 25% OLAP time share
        "olap_batch_size": 40,           # ~10x I/O rate -> 1:10 access ratio
        "buffer_pool_mb": 2400,          # 2.4GB = 67% of data -> ~45-55% hit rate
    },
    # Full scale (100W) - lower hit rate, more stress testing
    "chbenchmark_3to1_100w": {
        "workload": "chbenchmark",
        "num_warehouses": 100,           # Full SF=100 (~35GB data)
        "num_terminals": 12,
        "tpcc_transactions_per_terminal": 50,
        "num_olap_streams": 4,
        "olap_probability": 0.25,
        "olap_batch_size": 40,
        "buffer_pool_mb": 2400,          # ~7% of data -> lower hit rate
    },
    # Variant with real buffer size (256MB) - extreme stress test
    "chbenchmark_3to1_256mb": {
        "workload": "chbenchmark",
        "num_warehouses": 10,
        "num_terminals": 12,
        "tpcc_transactions_per_terminal": 100,
        "num_olap_streams": 4,
        "olap_probability": 0.25,
        "olap_batch_size": 40,
        "buffer_pool_mb": 256,           # 256MB = 7% of data -> ~10-20% hit rate
    },
    # YCSB (key-value, Zipfian distribution)
    "ycsb_a": {
        "workload": "ycsb",
        "num_records": 1_000_000,
        "read_pct": 0.50,
        "update_pct": 0.50,
        "zipfian_theta": 0.9,
        "num_operations": 50_000,
        "num_threads": 8,
        "buffer_pool_mb": 512,
    },
    "ycsb_b": {
        "workload": "ycsb",
        "num_records": 1_000_000,
        "read_pct": 0.95,
        "update_pct": 0.05,
        "zipfian_theta": 0.9,
        "num_operations": 50_000,
        "num_threads": 8,
        "buffer_pool_mb": 512,
    },
}

# Which workload config to use (can be changed via environment variable)
# Default to "combined" which runs TPC-H, TPC-C, and YCSB with weighted scoring
ACTIVE_WORKLOAD = os.environ.get("EVOLVE_WORKLOAD", "combined")

# LATENCY METRICS: Toggle with EVOLVE_LATENCY=1 environment variable
# When enabled, measures realistic latency instead of just hit rate
# Grounded on postgres-pbm thesis results (see LATENCY_METRICS_PROPOSAL.md)
LATENCY_ENABLED = os.environ.get("EVOLVE_LATENCY", "0") == "1"

# WAL FLUSH COST: Toggle with EVOLVE_WAL_COST=1 environment variable
# When enabled, adds latency for recently dirtied pages (need WAL flush)
# Models PostgreSQL's XLogNeedsFlush() check - recently dirtied pages are expensive
# Expected discovery: Prefer "old dirty" pages over "new dirty" pages
WAL_COST_ENABLED = os.environ.get("EVOLVE_WAL_COST", "0") == "1"

# RE-EVICTION TRACKING: Toggle with EVOLVE_REEVICT=1 environment variable
# When enabled, tracks pages that are re-loaded soon after eviction (bad decisions)
# Provides direct feedback on prediction quality
# Expected discovery: Better Belady predictions, fewer thrashing decisions
REEVICTION_ENABLED = os.environ.get("EVOLVE_REEVICT", "0") == "1"

# P2: DIRTY EVICTION PENALTY: Toggle with EVOLVE_DIRTY_PENALTY=1 environment variable
# When enabled, adds an explicit penalty for sync dirty evictions to the score
# This makes the scoring more sensitive to dirty evictions, encouraging evolution
# to discover clean-over-dirty preference (dual victim tracking)
# Penalty: score -= sync_dirty_rate * DIRTY_PENALTY_WEIGHT
DIRTY_PENALTY_ENABLED = os.environ.get("EVOLVE_DIRTY_PENALTY", "0") == "1"
DIRTY_PENALTY_WEIGHT = float(os.environ.get("EVOLVE_DIRTY_PENALTY_WEIGHT", "0.15"))  # 15% penalty at 100% dirty rate

# Scoring mode: "hit_rate" (default) or "latency" (when EVOLVE_LATENCY=1)
# "latency" mode captures dirty eviction cost (sync write = 200μs)
SCORING_MODE = "latency" if LATENCY_ENABLED else "hit_rate"

# Combined workload weights
COMBINED_WEIGHTS = {
    "tpch_full": 0.5,   # Analytical scans (PBM strength)
    "tpcc_10w": 0.3,    # OLTP point lookups (use 10W for speed)
    "ycsb_a": 0.2,      # Zipfian hot/cold patterns
}

# Fast combined weights for evolution (use EVOLVE_WORKLOAD=combined_fast)
# Total runtime ~10s vs ~90s for combined
COMBINED_FAST_WEIGHTS = {
    "tpch_fast": 0.5,   # 4 streams, table_scale=0.02 (~4s)
    "tpcc_10w": 0.3,    # Already fast (~0.6s)
    "ycsb_a": 0.2,      # Already fast (~0.2s)
}

# TPC-H + TPC-C only (50/50) - use EVOLVE_WORKLOAD=tpch_tpcc
# Focuses on the two main workloads, excludes YCSB
# Total runtime ~27s (tpch_full: ~27s, tpcc_10w: ~0.5s)
TPCH_TPCC_WEIGHTS = {
    "tpch_full": 0.5,   # Analytical scans
    "tpcc_10w": 0.5,    # OLTP point lookups (with buffer pressure)
}


def _log_metrics(code: str, metrics: dict):
    """Log code and metrics to a jsonl file for later analysis"""
    dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openevolve_metrics.jsonl")
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_length": len(code),
        "metrics": metrics
    }
    with open(dst_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def run_with_timeout(program_path, timeout_seconds=90, workload_config=None, 
                     latency_enabled=None, wal_cost_enabled=None, reeviction_enabled=None):
    """
    Run the program in a separate process with timeout.
    
    Args:
        program_path: Path to the policy file
        timeout_seconds: Timeout for evaluation
        workload_config: Workload configuration dict
        latency_enabled: Enable latency metrics (if None, uses LATENCY_ENABLED global)
        wal_cost_enabled: Enable WAL flush cost (if None, uses WAL_COST_ENABLED global)
        reeviction_enabled: Enable re-eviction tracking (if None, uses REEVICTION_ENABLED global)
    
    Returns:
        dict with hit_rate, latency_score, latency metrics, etc.
    """
    simulator_dir = os.path.dirname(os.path.abspath(__file__))
    program_dir = os.path.dirname(os.path.abspath(program_path))
    
    # Get workload config
    if workload_config is None:
        workload_config = WORKLOAD_CONFIGS.get(ACTIVE_WORKLOAD, WORKLOAD_CONFIGS["default"])
    
    # Resolve feature flags
    use_latency = latency_enabled if latency_enabled is not None else LATENCY_ENABLED
    use_wal_cost = wal_cost_enabled if wal_cost_enabled is not None else WAL_COST_ENABLED
    use_reeviction = reeviction_enabled if reeviction_enabled is not None else REEVICTION_ENABLED
    latency_str = "True" if use_latency else "False"
    wal_cost_str = "True" if use_wal_cost else "False"
    reeviction_str = "True" if use_reeviction else "False"
    
    # Serialize config for subprocess
    config_json = json.dumps(workload_config)
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
        # Use string concatenation to avoid f-string brace issues
        script = """
import sys
import os
import pickle
import traceback
import json
import random

# Set fixed random seed for reproducible evaluations
random.seed(42)

# Feature flags
LATENCY_ENABLED = """ + latency_str + """
WAL_COST_ENABLED = """ + wal_cost_str + """
REEVICTION_ENABLED = """ + reeviction_str + """

# Set up paths
sys.path.insert(0, r'""" + simulator_dir + """')
sys.path.insert(0, r'""" + program_dir + """')

# Parse workload config
wl_config = json.loads(r'""" + config_json + """')

# Clear cached imports
for module_name in list(sys.modules.keys()):
    if any(x in module_name for x in ['core', 'scan_tracker', 'workload', 'policies']):
        del sys.modules[module_name]

try:
    # Import the evolved policy
    spec = __import__('importlib.util').util.spec_from_file_location("evolved_policy", r'""" + program_path + """')
    policy_module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(policy_module)
    
    # Get the policy function (check multiple aliases)
    policy_func = None
    for name in ['evolved_policy', 'pbm_sampling_policy', 'pbm_pq_policy', 
                 'clock_sweep_policy', 'clock_pro_policy', 'pbm_horizon_policy',
                 'tinylfu_clock_policy', 'lru_k2_policy']:
        if hasattr(policy_module, name):
            policy_func = getattr(policy_module, name)
            break
    if policy_func is None:
        raise RuntimeError("No policy function found in module")
    
    # Import simulator components
    from core import BufferManager
    from scan_tracker import ScanRegistry, NextAccessEstimator
    
    # Configure workload based on type
    workload_type = wl_config.get("workload", "sequential")
    buffer_pool_blocks = wl_config.get("buffer_pool_mb", 256) * 128  # MB to 8KB blocks
    
    if workload_type == "tpch":
        from workload import TPCHWorkload, TPCHConfig
        config = TPCHConfig(
            num_streams=wl_config.get("num_streams", 4),
            buffer_pool_blocks=buffer_pool_blocks,
            table_scale=wl_config.get("table_scale", 0.10),
            synchronized_seqscans=wl_config.get("synchronized_seqscans", True),  # Match real PostgreSQL
        )
        workload = TPCHWorkload(config)
    elif workload_type == "tpcc":
        from workload import TPCCWorkload, TPCCConfig
        config = TPCCConfig(
            num_warehouses=wl_config.get("num_warehouses", 100),
            num_terminals=wl_config.get("num_terminals", 10),
            transactions_per_terminal=wl_config.get("transactions_per_terminal", 50),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = TPCCWorkload(config)
    elif workload_type == "ycsb":
        from workload import YCSBWorkload, YCSBConfig
        config = YCSBConfig(
            num_records=wl_config.get("num_records", 1000000),
            read_pct=wl_config.get("read_pct", 0.50),
            update_pct=wl_config.get("update_pct", 0.50),
            zipfian_theta=wl_config.get("zipfian_theta", 0.9),
            num_operations=wl_config.get("num_operations", 50000),
            num_threads=wl_config.get("num_threads", 8),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = YCSBWorkload(config)
    elif workload_type == "mixed":
        from workload import MixedWorkload, MixedWorkloadConfig
        config = MixedWorkloadConfig(
            tpch_num_streams=wl_config.get("tpch_num_streams", 4),
            tpch_table_scale=wl_config.get("tpch_table_scale", 0.03),
            tpcc_num_warehouses=wl_config.get("tpcc_num_warehouses", 10),
            tpcc_num_terminals=wl_config.get("tpcc_num_terminals", 5),
            tpcc_transactions_per_terminal=wl_config.get("tpcc_transactions_per_terminal", 100),
            tpch_probability=wl_config.get("tpch_probability", 0.5),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = MixedWorkload(config)
    elif workload_type == "chbenchmark":
        from workload import CHBenchmarkWorkload, CHBenchmarkConfig
        config = CHBenchmarkConfig(
            num_warehouses=wl_config.get("num_warehouses", 10),
            num_terminals=wl_config.get("num_terminals", 5),
            tpcc_transactions_per_terminal=wl_config.get("tpcc_transactions_per_terminal", 200),
            num_olap_streams=wl_config.get("num_olap_streams", 2),
            olap_probability=wl_config.get("olap_probability", 0.05),
            olap_batch_size=wl_config.get("olap_batch_size", 1000),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = CHBenchmarkWorkload(config)
    else:
        from workload import SequentialMicrobench, SequentialMicrobenchConfig
        table_blocks = int(1100000 * wl_config.get("table_scale", 0.40))
        config = SequentialMicrobenchConfig(
            num_streams=wl_config.get("num_streams", 2),
            queries_per_stream=wl_config.get("queries_per_stream", 8),
            table_blocks=table_blocks,
            buffer_pool_blocks=buffer_pool_blocks,
            selectivity=wl_config.get("selectivity", 0.30),
        )
        workload = SequentialMicrobench(config)
    
    # Run evaluation with optional latency and extended tracking
    manager = BufferManager(
        buffer_pool_blocks, 
        policy_func, 
        latency_enabled=LATENCY_ENABLED,
        wal_cost_enabled=WAL_COST_ENABLED,
        reeviction_enabled=REEVICTION_ENABLED
    )
    scans = ScanRegistry(manager.block_groups)
    estimator = NextAccessEstimator(scans, manager.block_groups)
    
    # P4: Wrap estimator to count calls for overhead tracking
    # avg_estimator_calls_per_eviction shows policy efficiency
    # COMBINED with fast paths: ~2-5 calls/eviction
    # SAMPLING without fast paths: ~16 calls/eviction
    class InstrumentedEstimator:
        # Wrapper that counts estimator calls for overhead analysis
        def __init__(self, inner, stats):
            self._inner = inner
            self._stats = stats
        
        def estimate_for_buffer(self, buf):
            self._stats.estimator_calls += 1
            return self._inner.estimate_for_buffer(buf)
        
        def __getattr__(self, name):
            return getattr(self._inner, name)
    
    instrumented_estimator = InstrumentedEstimator(estimator, manager.stats)
    manager.set_scan_tracker(scans, instrumented_estimator)
    
    active_scans = {}
    t = 0.0
    count = 0
    start_time = __import__('time').time()
    
    for access in workload.generate():
        t += 0.0001
        manager.set_time(t)
        count += 1
        
        if access.scan_context:
            ctx = access.scan_context
            if ctx.scan_id not in active_scans:
                if ctx.bitmap:
                    registry_scan_id = scans.register_bitmap_scan(ctx.relation_id, ctx.bitmap)
                else:
                    registry_scan_id = scans.register_sequential_scan(
                        ctx.relation_id, ctx.current_position,
                        ctx.current_position + ctx.total_blocks
                    )
                active_scans[ctx.scan_id] = (registry_scan_id, ctx.current_position)
            
            registry_scan_id, last_pos = active_scans[ctx.scan_id]
            if ctx.current_position - last_pos >= 32:
                scans.update_scan_position(registry_scan_id, ctx.current_position)
                active_scans[ctx.scan_id] = (registry_scan_id, ctx.current_position)
        
        buf_id = manager.read_buffer(access.tag, access.scan_context, is_write=access.is_write)
        manager.unpin_buffer(buf_id)
    
    elapsed = __import__('time').time() - start_time
    
    # Build results with core and latency metrics
    stats = manager.stats
    results = {
        'hit_rate': stats.hit_rate,
        'io_bytes': stats.bytes_read,
        'accesses': count,
        'elapsed': elapsed,
        'hits': stats.buffer_hits,
        'misses': stats.buffer_misses,
        'dirty_writes': stats.buffer_writes,
        # P4: Overhead tracking
        'eviction_calls': stats.eviction_calls,
        'estimator_calls': stats.estimator_calls,
        'avg_estimator_calls_per_eviction': stats.avg_estimator_calls_per_eviction,
    }
    
    # Add latency metrics if enabled
    if LATENCY_ENABLED:
        results.update({
            'latency_enabled': True,
            'latency_score': stats.latency_score,
            'avg_latency_us': stats.avg_access_latency_us,
            'sync_dirty_writes': stats.sync_dirty_writes,
            'async_dirty_writes': stats.async_dirty_writes,
            'sync_dirty_rate': stats.sync_dirty_rate,
            'dirty_eviction_rate': stats.dirty_eviction_rate,
            'sequential_reads': stats.sequential_reads,
            'random_reads': stats.random_reads,
            'latency_breakdown': stats.latency_breakdown,
        })
        log_msg = f"Evaluation complete: hit_rate={stats.hit_rate:.4f}, latency_score={stats.latency_score:.4f}, avg_latency={stats.avg_access_latency_us:.1f}us, sync_dirty={stats.sync_dirty_rate:.2%}"
        
        # Add WAL flush metrics if enabled
        if WAL_COST_ENABLED:
            results.update({
                'wal_cost_enabled': True,
                'wal_flush_rate': stats.wal_flush_rate,
                'wal_flush_evictions': stats.wal_flush_evictions,
            })
            log_msg += f", wal_flush={stats.wal_flush_rate:.2%}"
        
        # Add re-eviction metrics if enabled
        if REEVICTION_ENABLED:
            results.update({
                'reeviction_enabled': True,
                're_eviction_rate': stats.re_eviction_rate,
                're_evictions': stats.re_evictions,
            })
            log_msg += f", re_evict={stats.re_eviction_rate:.2%}"
        
        log_msg += f", accesses={count}, time={elapsed:.1f}s"
        print(log_msg)
    else:
        results['latency_enabled'] = False
        print(f"Evaluation complete: hit_rate={results['hit_rate']:.4f}, accesses={count}, time={elapsed:.1f}s")
    
    results_path = r'""" + temp_file.name + """.results'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()
    results_path = r'""" + temp_file.name + """.results'
    with open(results_path, 'wb') as f:
        pickle.dump({'error': str(e)}, f)
"""
        temp_file.write(script)
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"
    process = None
    
    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
            
            print(f"Subprocess output: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")
            
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")
            
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                
                if "error" in results:
                    raise RuntimeError(f"Evaluation failed: {results['error']}")
                
                return results
            else:
                raise RuntimeError("Results file not found")
                
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")
            
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def _create_artifacts(workload_results, combined_score):
    """Create detailed artifacts for LLM feedback."""
    artifacts = {
        "overall_score": f"{combined_score:.4f}",
    }
    
    # Per-workload feedback
    feedback = []
    best_workload = None
    worst_workload = None
    best_rate = -1
    worst_rate = 2
    
    for name, result in workload_results.items():
        if 'error' in result:
            feedback.append(f"❌ {name}: FAILED - {result['error']}")
            if 'timeout' in result['error'].lower():
                feedback.append(f"   → Algorithm too slow for {name}. Simplify or sample fewer buffers.")
        else:
            rate = result.get('hit_rate', 0)
            elapsed = result.get('elapsed', 0)
            feedback.append(f"✓ {name}: hit_rate={rate:.2%}, time={elapsed:.1f}s")
            if rate > best_rate:
                best_rate = rate
                best_workload = name
            if rate < worst_rate:
                worst_rate = rate
                worst_workload = name
    
    artifacts["workload_results"] = "\n".join(feedback)
    
    # Targeted suggestions
    suggestions = []
    if worst_workload == "tpch_full":
        suggestions.append("TPC-H weakness: Analytical scans. Try prioritizing eviction of scan-accessed buffers (check scan_ids).")
    elif worst_workload == "tpcc_10w":
        suggestions.append("TPC-C weakness: OLTP point lookups. Try better frequency-based eviction (access_count, LRU-K).")
    elif worst_workload == "ycsb_a":
        suggestions.append("YCSB weakness: Zipfian hot keys. Try better hot/cold separation or LFU-style counting.")
    
    if combined_score < 0.50:
        suggestions.append("Score low overall. Ensure you're returning valid buf_id with refcount==0.")
    elif combined_score < 0.55:
        suggestions.append("Slightly below baseline. Check if prediction (estimator) is being used effectively.")
    elif combined_score > 0.60:
        suggestions.append("Good score! Try tuning parameters or combining strategies for further gains.")
    
    if suggestions:
        artifacts["improvement_suggestions"] = " | ".join(suggestions)
    
    return artifacts


def evaluate(program_path, latency_enabled=None, wal_cost_enabled=None, reeviction_enabled=None):
    """
    Evaluate the evolved policy.
    
    For "combined" workload: runs TPC-H, TPC-C, and YCSB with weighted scoring.
    Otherwise: runs single specified workload.
    
    Scoring modes (controlled by EVOLVE_LATENCY env var or latency_enabled param):
    - hit_rate (default): Score based on buffer hit rate only
    - latency: Score based on latency_score which accounts for dirty eviction cost
    
    Extended features (controlled by env vars or params):
    - WAL flush cost (EVOLVE_WAL_COST=1): Adds latency for recently dirtied pages
    - Re-eviction tracking (EVOLVE_REEVICT=1): Tracks bad eviction decisions
    
    Returns:
        EvaluationResult with metrics and artifacts for LLM feedback.
    """
    # Resolve feature flags
    use_latency = latency_enabled if latency_enabled is not None else LATENCY_ENABLED
    use_wal_cost = wal_cost_enabled if wal_cost_enabled is not None else WAL_COST_ENABLED
    use_reeviction = reeviction_enabled if reeviction_enabled is not None else REEVICTION_ENABLED
    score_key = 'latency_score' if use_latency else 'hit_rate'
    
    try:
        code_str = open(program_path, 'r', encoding="utf-8").read()
        
        try:
            if ACTIVE_WORKLOAD in ("combined", "combined_fast", "tpch_tpcc"):
                # Run multiple workloads with weighted scoring
                # Use different weight configs based on ACTIVE_WORKLOAD
                if ACTIVE_WORKLOAD == "combined_fast":
                    weights = COMBINED_FAST_WEIGHTS
                elif ACTIVE_WORKLOAD == "tpch_tpcc":
                    weights = TPCH_TPCC_WEIGHTS
                else:
                    weights = COMBINED_WEIGHTS
                
                workload_results = {}
                combined_score = 0.0
                num_passed = 0
                
                for workload_name, weight in weights.items():
                    config = WORKLOAD_CONFIGS.get(workload_name)
                    if config is None:
                        print(f"Warning: workload {workload_name} not found, skipping")
                        continue
                    
                    try:
                        # 90s timeout per workload - penalize timeouts with score=0
                        results = run_with_timeout(program_path, timeout_seconds=90, 
                                                   workload_config=config,
                                                   latency_enabled=use_latency,
                                                   wal_cost_enabled=use_wal_cost,
                                                   reeviction_enabled=use_reeviction)
                        hit_rate = results['hit_rate']
                        
                        # Store all results for this workload
                        workload_results[workload_name] = {
                            'hit_rate': hit_rate,
                            'accesses': results['accesses'],
                            'elapsed': results['elapsed'],
                        }
                        
                        # Add latency metrics if enabled
                        if use_latency and results.get('latency_enabled'):
                            workload_results[workload_name].update({
                                'latency_score': results['latency_score'],
                                'avg_latency_us': results['avg_latency_us'],
                                'sync_dirty_rate': results['sync_dirty_rate'],
                            })
                            # Add WAL flush metrics if enabled
                            if use_wal_cost and results.get('wal_cost_enabled'):
                                workload_results[workload_name].update({
                                    'wal_flush_rate': results['wal_flush_rate'],
                                })
                            # Add re-eviction metrics if enabled
                            if use_reeviction and results.get('reeviction_enabled'):
                                workload_results[workload_name].update({
                                    're_eviction_rate': results['re_eviction_rate'],
                                })
                            # Use latency_score for combined scoring
                            score = results['latency_score']
                            
                            # P2: Apply dirty eviction penalty if enabled
                            if DIRTY_PENALTY_ENABLED:
                                dirty_penalty = results['sync_dirty_rate'] * DIRTY_PENALTY_WEIGHT
                                score = max(0.0, score - dirty_penalty)
                            
                            log_parts = [f"latency_score={score:.4f}", f"hit_rate={hit_rate:.4f}", f"sync_dirty={results['sync_dirty_rate']:.2%}"]
                            if use_wal_cost and results.get('wal_cost_enabled'):
                                log_parts.append(f"wal_flush={results.get('wal_flush_rate', 0):.2%}")
                            if use_reeviction and results.get('reeviction_enabled'):
                                log_parts.append(f"re_evict={results.get('re_eviction_rate', 0):.2%}")
                            log_parts.append(f"weight={weight}")
                            print(f"  {workload_name}: {', '.join(log_parts)}")
                        else:
                            # Use hit_rate for combined scoring
                            score = hit_rate
                            print(f"  {workload_name}: hit_rate={hit_rate:.4f} (weight={weight})")
                        
                        combined_score += weight * score
                        num_passed += 1
                    except Exception as e:
                        # Timeout or error: score = 0 for this workload (no renormalization!)
                        print(f"  {workload_name}: FAILED (score=0) - {e}")
                        workload_results[workload_name] = {'error': str(e), 'hit_rate': 0.0}
                        # combined_score += weight * 0.0  (implicit)
                
                result_dict = {
                    "runs_successfully": 1.0 if num_passed > 0 else 0.0,
                    "combined_score": float(combined_score),
                    "workload_results": workload_results,
                    "workload": "combined",
                    "scoring_mode": "latency" if use_latency else "hit_rate",
                }
                
                _log_metrics(code_str, result_dict)
                
                score_mode_str = "latency" if use_latency else "hit_rate"
                print(f"Combined Score: {combined_score:.4f} (mode={score_mode_str}, weights: tpch=0.5, tpcc=0.3, ycsb=0.2)")
                
                # Create artifacts for LLM feedback
                artifacts = _create_artifacts(workload_results, combined_score)
                if use_latency:
                    artifacts["scoring_mode"] = "latency"
                
                metrics = {
                    "runs_successfully": 1.0 if num_passed > 0 else 0.0,
                    "combined_score": float(combined_score),
                }
                
                return EvaluationResult(metrics=metrics, artifacts=artifacts)
            else:
                # Single workload mode
                results = run_with_timeout(program_path, timeout_seconds=90, 
                                          latency_enabled=use_latency,
                                          wal_cost_enabled=use_wal_cost,
                                          reeviction_enabled=use_reeviction)
                
                hit_rate = results['hit_rate']
                
                # Use latency_score if enabled, otherwise hit_rate
                if use_latency and results.get('latency_enabled'):
                    combined_score = results['latency_score']
                    
                    # P2: Apply dirty eviction penalty if enabled
                    if DIRTY_PENALTY_ENABLED:
                        dirty_penalty = results['sync_dirty_rate'] * DIRTY_PENALTY_WEIGHT
                        combined_score = max(0.0, combined_score - dirty_penalty)
                    
                    print(f"Score: {combined_score:.4f} (latency_score, hit_rate={hit_rate:.4f}, sync_dirty={results['sync_dirty_rate']:.2%}, workload={ACTIVE_WORKLOAD})")
                else:
                    combined_score = hit_rate
                    print(f"Score: {combined_score:.4f} (hit_rate={hit_rate:.4f}, workload={ACTIVE_WORKLOAD})")
                
                result_dict = {
                    "runs_successfully": 1.0,
                    "combined_score": float(combined_score),
                    "hit_rate": float(hit_rate),
                    "accesses": results['accesses'],
                    "elapsed": results['elapsed'],
                    "workload": ACTIVE_WORKLOAD,
                    "scoring_mode": "latency" if use_latency else "hit_rate",
                }
                
                _log_metrics(code_str, result_dict)
                
                metrics = {"runs_successfully": 1.0, "combined_score": float(combined_score)}
                artifacts = {"workload": ACTIVE_WORKLOAD, "hit_rate": f"{hit_rate:.4f}"}
                if use_latency and results.get('latency_enabled'):
                    artifacts["latency_score"] = f"{results['latency_score']:.4f}"
                    artifacts["avg_latency_us"] = f"{results['avg_latency_us']:.1f}"
                    artifacts["sync_dirty_rate"] = f"{results['sync_dirty_rate']:.2%}"
                return EvaluationResult(metrics=metrics, artifacts=artifacts)
            
        except TimeoutError:
            metrics = {"runs_successfully": 0.0, "combined_score": 0.0}
            artifacts = {
                "error_type": "Timeout",
                "error_message": "Policy execution exceeded timeout",
                "suggestion": "Algorithm too slow. Try: reduce estimator calls, sample fewer buffers, simplify logic"
            }
            return EvaluationResult(metrics=metrics, artifacts=artifacts)
        except Exception as e:
            traceback.print_exc()
            metrics = {"runs_successfully": 0.0, "combined_score": 0.0}
            artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()[-500:],  # Last 500 chars
                "suggestion": "Check for: syntax errors, missing imports, invalid buffer access"
            }
            return EvaluationResult(metrics=metrics, artifacts=artifacts)
    
    except Exception as e:
        metrics = {"runs_successfully": 0.0, "combined_score": 0.0}
        artifacts = {
            "error_type": "LoadError",
            "error_message": str(e),
            "suggestion": "Program failed to load. Check syntax and imports."
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a PBM policy.")
    parser.add_argument("program_path", help="Path to the policy file to evaluate.")
    args = parser.parse_args()
    
    result = evaluate(args.program_path)
    
    # Handle both EvaluationResult and dict returns
    if hasattr(result, 'metrics'):
        output = {"metrics": result.metrics, "artifacts": result.artifacts}
    else:
        output = result
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
