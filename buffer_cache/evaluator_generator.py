"""
EvaluatorGenerator — Generates evaluator.py + initial_program.py from a SimulatorConfig.

This is the bridge between the outer loop (simulator evolution) and the inner loop
(OpenEvolve policy evolution). It generates files that:
1. Import the existing V5 simulator modules (no duplication)
2. Wrap BufferDescriptor/estimator to mask fields disabled by the config
3. Configure workload mix, scoring, and feedback from config settings
4. Produce a valid OpenEvolve evaluator contract (evaluate(program_path) -> EvaluationResult)

The generated evaluator loads SimulatorConfig from sim_config.json at runtime.
"""

import json
import os
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    from .simulator_config import SimulatorConfig
except ImportError:
    from simulator_config import SimulatorConfig


# Path to the PBM buffer simulator.
# Override with SIMEVOLVER_SIMULATOR_DIR env var, or symlink/clone into pg_clean/simulator/
V5_SIMULATOR_DIR = os.environ.get(
    "SIMEVOLVER_SIMULATOR_DIR",
    str(Path(__file__).parent / "simulator"),
)


def generate_evaluator(config: SimulatorConfig, output_dir: str) -> str:
    """
    Generate evaluator.py from a SimulatorConfig.

    Args:
        config: SimulatorConfig defining the simulator variant
        output_dir: Directory to write generated files into

    Returns:
        Path to generated evaluator.py
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config as JSON for runtime loading
    config_path = output_path / "sim_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Write evaluator.py (template with V5_DIR injected)
    evaluator_code = _EVALUATOR_TEMPLATE.replace("__V5_DIR_PLACEHOLDER__", V5_SIMULATOR_DIR)
    evaluator_path = output_path / "evaluator.py"
    with open(evaluator_path, "w") as f:
        f.write(evaluator_code)

    # Generate initial_program.py
    initial_code = _generate_initial_program(config)
    initial_path = output_path / "initial_program.py"
    with open(initial_path, "w") as f:
        f.write(initial_code)

    return str(evaluator_path)


# ---- Evaluator template (standalone, config-driven) ----
# This is written as a regular string, NOT an f-string, to avoid brace issues.
# The only placeholder is __V5_DIR_PLACEHOLDER__ which gets replaced at generation time.

_EVALUATOR_TEMPLATE = r'''"""
Auto-generated evaluator for sim_evolver.
Reads SimulatorConfig from sim_config.json and wraps V5 simulator accordingly.
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

# Add V5 simulator to path
V5_DIR = r"__V5_DIR_PLACEHOLDER__"
if V5_DIR not in sys.path:
    sys.path.insert(0, V5_DIR)

# Try to import EvaluationResult
try:
    # Add openevolve to path if available
    _oe_dir = os.environ.get("SIMEVOLVER_OPENEVOLVE_DIR",
                             os.path.join(os.path.dirname(V5_DIR), os.pardir, "openevolve"))
    if os.path.isdir(_oe_dir) and _oe_dir not in sys.path:
        sys.path.insert(0, _oe_dir)
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    class EvaluationResult:
        def __init__(self, metrics, artifacts=None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

# Load config from co-located JSON
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_config.json")
with open(_CONFIG_PATH) as _f:
    CONFIG = json.load(_f)

# Import V5 workload configs — try simulator's evaluator, fall back to built-in defaults
try:
    from evaluator import WORKLOAD_CONFIGS
except Exception:
    # Built-in subset covering the most common workloads
    WORKLOAD_CONFIGS = {
        "tpch_full": {"workload": "tpch", "num_streams": 8, "table_scale": 0.05, "buffer_pool_mb": 256},
        "tpch_fast": {"workload": "tpch", "num_streams": 4, "table_scale": 0.02, "buffer_pool_mb": 128},
        "tpch_hard": {"workload": "tpch", "num_streams": 8, "table_scale": 0.05, "buffer_pool_mb": 64},
        "tpcc_10w": {"workload": "tpcc", "num_warehouses": 10, "num_terminals": 5, "transactions_per_terminal": 100, "buffer_pool_mb": 128},
        "tpcc_10w_stress": {"workload": "tpcc", "num_warehouses": 10, "num_terminals": 5, "transactions_per_terminal": 100, "buffer_pool_mb": 64},
        "tpcc_100w_calibrated": {"workload": "tpcc", "num_warehouses": 100, "num_terminals": 60, "transactions_per_terminal": 50, "buffer_pool_mb": 256},
        "ycsb_a": {"workload": "ycsb", "num_records": 1000000, "read_pct": 0.50, "update_pct": 0.50, "zipfian_theta": 0.9, "num_operations": 50000, "num_threads": 8, "buffer_pool_mb": 512},
        "ycsb_b": {"workload": "ycsb", "num_records": 1000000, "read_pct": 0.95, "update_pct": 0.05, "zipfian_theta": 0.9, "num_operations": 50000, "num_threads": 8, "buffer_pool_mb": 512},
        "chbenchmark_3to1": {"workload": "chbenchmark", "num_warehouses": 10, "num_terminals": 12, "tpcc_transactions_per_terminal": 100, "num_olap_streams": 4, "olap_probability": 0.25, "olap_batch_size": 40, "buffer_pool_mb": 2400},
    }


def _run_single_workload(program_path, workload_config, config):
    """Run a single workload evaluation in a subprocess."""
    simulator_dir = V5_DIR
    program_dir = os.path.dirname(os.path.abspath(program_path))
    config_json = json.dumps(workload_config)
    sim_config_json = json.dumps(config)

    latency_enabled = config.get("scoring_mode", "hit_rate") in ("latency", "combined")
    wal_cost_enabled = config.get("wal_cost_enabled", False)

    # Build subprocess script as a list of string parts to avoid f-string issues
    script_parts = [
        "import sys, os, pickle, traceback, json, random",
        "random.seed(42)",
        "sys.path.insert(0, %r)" % simulator_dir,
        "sys.path.insert(0, %r)" % program_dir,
        "wl_config = json.loads(%r)" % config_json,
        "sim_config = json.loads(%r)" % sim_config_json,
        "LATENCY_ENABLED = %s" % latency_enabled,
        "WAL_COST_ENABLED = %s" % wal_cost_enabled,
        "REEVICTION_ENABLED = False",
        "",
        "for mn in list(sys.modules.keys()):",
        "    if any(x in mn for x in ['core', 'scan_tracker', 'workload', 'policies']):",
        "        del sys.modules[mn]",
    ]

    # The rest of the subprocess script uses exec() with a heredoc-style approach
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
        # Write the setup
        temp_file.write("\n".join(script_parts) + "\n\n")

        # Write the main evaluation logic
        temp_file.write(_SUBPROCESS_SCRIPT.replace(
            "__PROGRAM_PATH__", program_path
        ).replace(
            "__RESULTS_PATH__", temp_file.name + ".results"
        ))
        temp_file_path = temp_file.name

    results_path = temp_file_path + ".results"
    process = None

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=90)
        exit_code = process.returncode

        print("Subprocess output: " + stdout.decode())
        if stderr:
            print("Subprocess stderr: " + stderr.decode())

        if exit_code != 0:
            raise RuntimeError("Process exited with code %d" % exit_code)

        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            if "error" in results:
                raise RuntimeError("Evaluation failed: " + results["error"])
            return results
        else:
            raise RuntimeError("Results file not found")

    except subprocess.TimeoutExpired:
        raise TimeoutError("Process timed out after 90 seconds")
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


# Subprocess evaluation script template
_SUBPROCESS_SCRIPT = r"""
try:
    spec = __import__('importlib.util').util.spec_from_file_location(
        "evolved_policy", r"__PROGRAM_PATH__")
    policy_module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(policy_module)

    policy_func = None
    for name in ['evolved_policy', 'pbm_sampling_policy', 'pbm_pq_policy',
                 'clock_sweep_policy', 'clock_pro_policy', 'pbm_horizon_policy',
                 'tinylfu_clock_policy', 'lru_k2_policy']:
        if hasattr(policy_module, name):
            policy_func = getattr(policy_module, name)
            break
    if policy_func is None:
        raise RuntimeError("No policy function found in module")

    from core import BufferManager
    from scan_tracker import ScanRegistry, NextAccessEstimator

    workload_type = wl_config.get("workload", "sequential")
    buffer_pool_blocks = wl_config.get("buffer_pool_mb", 256) * 128

    if workload_type == "tpch":
        from workload import TPCHWorkload, TPCHConfig
        wl_cfg = TPCHConfig(
            num_streams=wl_config.get("num_streams", 4),
            buffer_pool_blocks=buffer_pool_blocks,
            table_scale=wl_config.get("table_scale", 0.10),
            synchronized_seqscans=wl_config.get("synchronized_seqscans", True),
        )
        workload = TPCHWorkload(wl_cfg)
    elif workload_type == "tpcc":
        from workload import TPCCWorkload, TPCCConfig
        wl_cfg = TPCCConfig(
            num_warehouses=wl_config.get("num_warehouses", 100),
            num_terminals=wl_config.get("num_terminals", 10),
            transactions_per_terminal=wl_config.get("transactions_per_terminal", 50),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = TPCCWorkload(wl_cfg)
    elif workload_type == "ycsb":
        from workload import YCSBWorkload, YCSBConfig
        wl_cfg = YCSBConfig(
            num_records=wl_config.get("num_records", 1000000),
            read_pct=wl_config.get("read_pct", 0.50),
            update_pct=wl_config.get("update_pct", 0.50),
            zipfian_theta=wl_config.get("zipfian_theta", 0.9),
            num_operations=wl_config.get("num_operations", 50000),
            num_threads=wl_config.get("num_threads", 8),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = YCSBWorkload(wl_cfg)
    elif workload_type == "chbenchmark":
        from workload import CHBenchmarkWorkload, CHBenchmarkConfig
        wl_cfg = CHBenchmarkConfig(
            num_warehouses=wl_config.get("num_warehouses", 10),
            num_terminals=wl_config.get("num_terminals", 5),
            tpcc_transactions_per_terminal=wl_config.get("tpcc_transactions_per_terminal", 200),
            num_olap_streams=wl_config.get("num_olap_streams", 2),
            olap_probability=wl_config.get("olap_probability", 0.05),
            olap_batch_size=wl_config.get("olap_batch_size", 1000),
            buffer_pool_blocks=buffer_pool_blocks,
        )
        workload = CHBenchmarkWorkload(wl_cfg)
    else:
        from workload import SequentialMicrobench, SequentialMicrobenchConfig
        table_blocks = int(1100000 * wl_config.get("table_scale", 0.40))
        wl_cfg = SequentialMicrobenchConfig(
            num_streams=wl_config.get("num_streams", 2),
            queries_per_stream=wl_config.get("queries_per_stream", 8),
            table_blocks=table_blocks,
            buffer_pool_blocks=buffer_pool_blocks,
            selectivity=wl_config.get("selectivity", 0.30),
        )
        workload = SequentialMicrobench(wl_cfg)

    # --- Information masking based on sim_config ---
    class MaskedBlockGroup:
        # Wraps a BlockGroup to hide scan_ids when expose_scan_ids is False.
        def __init__(self, inner, cfg):
            self._inner = inner
            if cfg.get("expose_scan_ids", True):
                self.scan_ids = inner.scan_ids
            else:
                self.scan_ids = set()
        def __getattr__(self, name):
            return getattr(self._inner, name)

    class MaskedBuffer:
        __slots__ = ['buf_id', 'tag', 'refcount', 'usage_count', 'is_dirty',
                     'block_group', 'last_access_time', 'access_count', 'dirty_time', '_inner']
        def __init__(self, inner, cfg):
            self._inner = inner
            self.buf_id = inner.buf_id
            self.tag = inner.tag
            self.refcount = inner.refcount
            self.usage_count = inner.usage_count if cfg.get("expose_usage_count", True) else 0
            self.is_dirty = inner.is_dirty if cfg.get("expose_is_dirty", True) else False
            if not cfg.get("expose_block_group", True):
                self.block_group = None
            elif inner.block_group is not None and not cfg.get("expose_scan_ids", True):
                self.block_group = MaskedBlockGroup(inner.block_group, cfg)
            else:
                self.block_group = inner.block_group
            self.last_access_time = inner.last_access_time if cfg.get("expose_last_access_time", True) else 0.0
            self.access_count = inner.access_count if cfg.get("expose_access_count", True) else 0
            self.dirty_time = inner.dirty_time if cfg.get("expose_dirty_time", True) else 0.0
        def is_valid(self): return self.tag is not None
        def is_pinned(self): return self.refcount > 0
        def can_evict(self): return self.refcount == 0

    class MaskedBufferList:
        def __init__(self, buffers, cfg):
            self._buffers = buffers
            self._cfg = cfg
        def __getitem__(self, idx):
            return MaskedBuffer(self._buffers[idx], self._cfg)
        def __len__(self): return len(self._buffers)
        def __iter__(self):
            for i in range(len(self._buffers)): yield self[i]

    class MaskedEstimator:
        def __init__(self, inner, stats, cfg):
            self._inner = inner
            self._stats = stats
            self._cfg = cfg
        def estimate_for_buffer(self, buf):
            if not self._cfg.get("enable_estimator", True):
                return (float('inf'), False)
            self._stats.estimator_calls += 1
            actual_buf = buf._inner if hasattr(buf, '_inner') else buf
            return self._inner.estimate_for_buffer(actual_buf)
        def estimate_for_buffer_with_confidence(self, buf):
            if not self._cfg.get("enable_confidence", False):
                return (float('inf'), False, 0.0)
            self._stats.estimator_calls += 1
            actual_buf = buf._inner if hasattr(buf, '_inner') else buf
            return self._inner.estimate_for_buffer_with_confidence(actual_buf)
        def estimate_for_block_group(self, rel_id, group_num):
            if not self._cfg.get("enable_block_group_estimate", False):
                return (float('inf'), False)
            return self._inner.estimate_for_block_group(rel_id, group_num)

    _original_policy = policy_func
    _sim_cfg = sim_config

    def wrapping_policy(buffers, buffer_table, estimator, scan_context):
        masked_bufs = MaskedBufferList(buffers, _sim_cfg)
        masked_est = MaskedEstimator(estimator, manager.stats, _sim_cfg) if estimator else None
        ctx = scan_context if _sim_cfg.get("policy_sees_scan_context", True) else None
        return _original_policy(masked_bufs, buffer_table, masked_est, ctx)

    manager = BufferManager(
        buffer_pool_blocks,
        wrapping_policy,
        latency_enabled=LATENCY_ENABLED,
        wal_cost_enabled=WAL_COST_ENABLED,
        reeviction_enabled=REEVICTION_ENABLED,
    )
    scans = ScanRegistry(manager.block_groups)
    estimator = NextAccessEstimator(scans, manager.block_groups)
    manager.set_scan_tracker(scans, estimator)

    active_scans = {}
    t = 0.0
    count = 0
    import time as _time
    start_time = _time.time()

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

    elapsed = _time.time() - start_time
    stats = manager.stats
    results = {
        'hit_rate': stats.hit_rate,
        'io_bytes': stats.bytes_read,
        'accesses': count,
        'elapsed': elapsed,
        'hits': stats.buffer_hits,
        'misses': stats.buffer_misses,
    }

    if LATENCY_ENABLED:
        results.update({
            'latency_enabled': True,
            'latency_score': stats.latency_score,
            'avg_latency_us': stats.avg_access_latency_us,
            'sync_dirty_writes': stats.sync_dirty_writes,
            'async_dirty_writes': stats.async_dirty_writes,
            'sync_dirty_rate': stats.sync_dirty_rate,
            'dirty_eviction_rate': stats.dirty_eviction_rate,
        })

    with open(r"__RESULTS_PATH__", 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    print("Error: " + str(e))
    traceback.print_exc()
    with open(r"__RESULTS_PATH__", 'wb') as f:
        pickle.dump({'error': str(e)}, f)
"""


def _create_artifacts(workload_results, combined_score, config):
    """Create detailed artifacts for LLM feedback."""
    artifacts = {"overall_score": "%.4f" % combined_score}

    if not config.get("per_workload_breakdown", True):
        return artifacts

    feedback = []
    worst_workload = None
    worst_rate = 2.0

    for name, result in workload_results.items():
        if 'error' in result:
            feedback.append("FAIL %s: %s" % (name, result['error']))
        else:
            rate = result.get('hit_rate', 0)
            feedback.append("%s: hit_rate=%.2f%%" % (name, rate * 100))
            if rate < worst_rate:
                worst_rate = rate
                worst_workload = name

    artifacts["workload_results"] = "\n".join(feedback)

    if config.get("artifact_suggestions", True):
        suggestions = []
        if worst_workload and "tpch" in worst_workload:
            suggestions.append("TPC-H weakness: Try scan-aware eviction.")
        elif worst_workload and "tpcc" in worst_workload:
            suggestions.append("TPC-C weakness: Try frequency-based eviction.")
        elif worst_workload and "ycsb" in worst_workload:
            suggestions.append("YCSB weakness: Try hot/cold separation.")
        if suggestions:
            artifacts["improvement_suggestions"] = " | ".join(suggestions)

    return artifacts


def evaluate(program_path):
    """
    Evaluate an evolved policy using the configured simulator variant.
    Returns EvaluationResult compatible with OpenEvolve.
    """
    try:
        workload_weights = CONFIG.get("workload_weights", {"tpch_full": 1.0})
        scoring_mode = CONFIG.get("scoring_mode", "hit_rate")
        score_key = 'latency_score' if scoring_mode in ('latency', 'combined') else 'hit_rate'

        if len(workload_weights) > 1:
            workload_results = {}
            combined_score = 0.0
            num_passed = 0

            for workload_name, weight in workload_weights.items():
                wl_config = WORKLOAD_CONFIGS.get(workload_name)
                if wl_config is None:
                    print("Warning: workload %s not found, skipping" % workload_name)
                    continue

                try:
                    results = _run_single_workload(program_path, wl_config, CONFIG)
                    score = results.get(score_key, results.get('hit_rate', 0.0))

                    dirty_penalty_w = CONFIG.get("dirty_penalty_weight", 0.0)
                    if dirty_penalty_w > 0 and 'sync_dirty_rate' in results:
                        score = max(0.0, score - results['sync_dirty_rate'] * dirty_penalty_w)

                    workload_results[workload_name] = {
                        'hit_rate': results['hit_rate'],
                        'elapsed': results.get('elapsed', 0),
                    }
                    combined_score += weight * score
                    num_passed += 1
                    print("  %s: score=%.4f (weight=%.2f)" % (workload_name, score, weight))
                except Exception as e:
                    print("  %s: FAILED - %s" % (workload_name, e))
                    workload_results[workload_name] = {'error': str(e)}

            metrics = {
                "runs_successfully": 1.0 if num_passed > 0 else 0.0,
                "combined_score": float(combined_score),
            }
            artifacts = _create_artifacts(workload_results, combined_score, CONFIG)
            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        else:
            workload_name = list(workload_weights.keys())[0]
            wl_config = WORKLOAD_CONFIGS.get(workload_name)
            if wl_config is None:
                metrics = {"runs_successfully": 0.0, "combined_score": 0.0}
                artifacts = {"error_type": "ConfigError",
                             "error_message": "Workload '%s' not found in WORKLOAD_CONFIGS" % workload_name}
                return EvaluationResult(metrics=metrics, artifacts=artifacts)
            results = _run_single_workload(program_path, wl_config, CONFIG)
            score = results.get(score_key, results.get('hit_rate', 0.0))

            metrics = {"runs_successfully": 1.0, "combined_score": float(score)}
            artifacts = {"workload": workload_name, "hit_rate": "%.4f" % results['hit_rate']}
            return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        traceback.print_exc()
        metrics = {"runs_successfully": 0.0, "combined_score": 0.0}
        artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()[-500:],
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path")
    args = parser.parse_args()
    result = evaluate(args.program_path)
    output = {"metrics": result.metrics, "artifacts": result.artifacts}
    print(json.dumps(output, indent=2))
'''


def _generate_initial_program(config: SimulatorConfig) -> str:
    """Generate an initial_program.py appropriate for the config."""
    if config.enable_estimator:
        return textwrap.dedent('''\
            """
            Initial evolvable policy - PBM sampling baseline.
            Auto-generated by sim_evolver for inner-loop evolution.
            """
            import random
            from typing import List, Dict, Optional, Any, Tuple


            # EVOLVE-BLOCK-START
            def evolved_policy(
                buffers: List,
                buffer_table: Dict,
                estimator: Any,
                scan_context: Any,
                num_samples: int = 7,
            ) -> Optional[int]:
                """
                Sample N random unpinned buffers, evict the one with largest
                predicted next-access time (furthest future access).
                """
                if estimator is None:
                    return None

                num_buffers = len(buffers)
                if num_buffers == 0:
                    return None

                _random = random.random

                # Fast path: find empty buffer
                for _ in range(3):
                    buf_id = int(_random() * num_buffers)
                    buf = buffers[buf_id]
                    if buf.refcount == 0 and buf.tag is None:
                        return buf_id

                # Sample and score
                best_id = None
                best_score = -1.0
                seen = set()
                attempts = 0

                while len(seen) < num_samples and attempts < num_samples * 3:
                    attempts += 1
                    buf_id = int(_random() * num_buffers)
                    if buf_id in seen:
                        continue
                    seen.add(buf_id)

                    buf = buffers[buf_id]
                    if buf.refcount > 0:
                        continue
                    if buf.tag is None:
                        return buf_id

                    next_access, is_requested = estimator.estimate_for_buffer(buf)
                    if next_access > best_score:
                        best_score = next_access
                        best_id = buf_id

                return best_id
            # EVOLVE-BLOCK-END
        ''')
    else:
        return textwrap.dedent('''\
            """
            Initial evolvable policy - random eviction baseline (no estimator).
            Auto-generated by sim_evolver for inner-loop evolution.
            """
            import random
            from typing import List, Dict, Optional, Any


            # EVOLVE-BLOCK-START
            def evolved_policy(
                buffers: List,
                buffer_table: Dict,
                estimator: Any,
                scan_context: Any,
            ) -> Optional[int]:
                """
                Simple random eviction: pick a random unpinned buffer.
                Prefer empty buffers and low usage_count.
                """
                num_buffers = len(buffers)
                if num_buffers == 0:
                    return None

                _random = random.random

                # Fast path: find empty buffer
                for _ in range(5):
                    buf_id = int(_random() * num_buffers)
                    buf = buffers[buf_id]
                    if buf.refcount == 0 and buf.tag is None:
                        return buf_id

                # Sample and pick lowest usage_count
                best_id = None
                best_uc = 999

                for _ in range(10):
                    buf_id = int(_random() * num_buffers)
                    buf = buffers[buf_id]
                    if buf.refcount > 0:
                        continue
                    if buf.tag is None:
                        return buf_id
                    if buf.usage_count < best_uc:
                        best_uc = buf.usage_count
                        best_id = buf_id

                return best_id
            # EVOLVE-BLOCK-END
        ''')


def generate_openevolve_config(config: SimulatorConfig, output_dir: str) -> str:
    """Generate an OpenEvolve config YAML for the inner loop."""
    import yaml

    oe_config = {
        "max_iterations": config.inner_iterations,
        "language": "python",
        "diff_based_evolution": True,
        "max_code_length": 30000,
        "early_stopping_patience": max(20, config.inner_iterations // 3),
        "llm": {
            "temperature": 0.85,
            "max_tokens": 30000,
            "timeout": 300,
        },
        "prompt": {
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "include_artifacts": True,
        },
        "database": {
            "population_size": config.inner_population_size,
            "num_islands": config.inner_num_islands,
            "migration_interval": 20,
            "feature_dimensions": ["combined_score", "complexity", "diversity"],
            "feature_bins": 10,
        },
        "evaluator": {
            "timeout": 300,
            "parallel_evaluations": 2,
            "enable_artifacts": True,
        },
    }

    output_path = Path(output_dir) / "openevolve_config.yaml"
    with open(output_path, "w") as f:
        yaml.dump(oe_config, f, default_flow_style=False)

    return str(output_path)
