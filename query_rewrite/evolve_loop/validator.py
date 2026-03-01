"""
Regression validator: tests RuleSelector against all queries to catch regressions.

Phase 4 of the evolution loop.

Validation runs in a subprocess to get a fresh JVM with the rebuilt JAR,
since JPype cannot reload Java classes in the same process.
"""

import json
import logging
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig
from evolve_loop.features import load_query
from evolve_loop.utils.java_bridge import init_jvm, rewrite_sql
from evolve_loop.utils.jar_builder import sync_and_rebuild
from evolve_loop.utils.pg_runner import get_connection, measure_latency_robust

logger = logging.getLogger(__name__)


def collect_baselines(
    queries: List[Tuple[str, str]],
    config: EvolutionConfig,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Measure baseline latencies for all queries (no rewrite rules).

    Args:
        queries: List of (benchmark, query_id) tuples
        config: Evolution config
        output_path: Optional path to save baselines JSON

    Returns:
        Dict mapping query_id to median baseline latency (seconds)
    """
    baselines = {}
    start_time = time.time()

    for i, (benchmark, query_id) in enumerate(queries, 1):
        elapsed = time.time() - start_time
        logger.info(
            "[%d/%d] Baseline %s/%s (elapsed: %.1fmin)",
            i, len(queries), benchmark, query_id, elapsed / 60,
        )

        try:
            sql, _create_tables = load_query(config.query_dirs[benchmark], query_id)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", query_id, e)
            continue

        conn = get_connection(config.pg_configs[benchmark])
        lat, status = measure_latency_robust(
            conn, sql,
            runs=config.validation_runs,
            warmup=config.validation_warmup,
            timeout_sec=config.search_timeout_sec,
        )
        conn.close()

        if lat is not None:
            baselines[query_id] = round(lat, 4)
            logger.info("  %s: %.4fs", query_id, lat)
        else:
            logger.info("  %s: %s (excluded from baselines)", query_id, status)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(baselines, f, indent=2)
        logger.info("Baselines saved to %s (%d queries)", output_path, len(baselines))

    return baselines


def validate_all(
    ruleselector_java: str,
    queries: List[Tuple[str, str]],
    baselines: Dict[str, float],
    config: EvolutionConfig,
    output_path: Optional[str] = None,
) -> Dict:
    """Validate a RuleSelector.java against all queries.

    Rebuilds JAR with the new code, then spawns a subprocess with a fresh
    JVM to test all queries and compare to baselines.

    Args:
        ruleselector_java: New RuleSelector.java source code
        queries: List of (benchmark, query_id) tuples
        baselines: Dict mapping query_id to baseline latency
        config: Evolution config
        output_path: Optional path to save validation report

    Returns:
        Validation report dict with wins, regressions, neutrals, summary
    """
    # Step 1: Sync source and rebuild JAR
    logger.info("Rebuilding JAR with new RuleSelector.java...")
    success, build_msg = sync_and_rebuild(
        ruleselector_java,
        config.canonical_ruleselector,
        config.rebuild_script,
        timeout_sec=config.build_timeout_sec,
    )
    if not success:
        logger.error("JAR rebuild failed: %s", build_msg)
        return {
            "error": f"JAR rebuild failed: {build_msg}",
            "wins": [],
            "regressions": [],
            "neutrals": [],
            "summary": {"total_wins": 0, "total_regressions": 0, "net_improvement": 0},
        }

    # Step 2: Run validation in subprocess (fresh JVM with rebuilt JAR)
    logger.info("Spawning validation subprocess...")
    return _run_validation_subprocess(queries, baselines, config, output_path)


def _run_validation_subprocess(
    queries: List[Tuple[str, str]],
    baselines: Dict[str, float],
    config: EvolutionConfig,
    output_path: Optional[str] = None,
) -> Dict:
    """Spawn validate_worker.py as a subprocess with a fresh JVM.

    Communicates via temp JSON files for input/output.
    """
    # Build input params
    params = {
        "jar_dir": config.jar_dir,
        "queries": [list(q) for q in queries],
        "baselines": baselines,
        "query_dirs": config.query_dirs,
        "pg_configs": config.pg_configs,
        "validation_runs": config.validation_runs,
        "validation_warmup": config.validation_warmup,
        "search_timeout_sec": config.search_timeout_sec,
        "win_threshold": config.win_threshold,
        "regression_threshold": config.regression_threshold,
        "train_suffix": config.train_suffix,
    }

    # Write input to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_val_input.json", delete=False
    ) as f:
        json.dump(params, f)
        input_path = f.name

    # Output path
    if output_path:
        out_path = output_path
    else:
        out_path = tempfile.mktemp(suffix="_val_output.json")

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "evolve_loop.validate_worker",
                "--input", input_path,
                "--output", out_path,
            ],
            capture_output=True,
            text=True,
            timeout=config.validation_timeout_sec,
            cwd=str(Path(__file__).parent.parent),
        )

        # Log subprocess output
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.info("[worker] %s", line)
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                if "[ERROR]" in line or "[WARNING]" in line:
                    logger.warning("[worker] %s", line)
                else:
                    logger.info("[worker] %s", line)

        if result.returncode != 0:
            logger.error(
                "Validation subprocess failed (exit code %d)", result.returncode
            )
            return {
                "error": f"Subprocess failed (exit {result.returncode}): {result.stderr[:500]}",
                "wins": [],
                "regressions": [],
                "neutrals": [],
                "summary": {"total_wins": 0, "total_regressions": 0, "net_improvement": 0},
            }

        # Read output
        with open(out_path) as f:
            report = json.load(f)

        logger.info(
            "Validation complete: %d wins, %d regressions",
            report["summary"]["total_wins"],
            report["summary"]["total_regressions"],
        )

        return report

    except subprocess.TimeoutExpired:
        logger.error(
            "Validation subprocess timed out after %ds",
            config.validation_timeout_sec,
        )
        return {
            "error": f"Validation timed out after {config.validation_timeout_sec}s",
            "wins": [],
            "regressions": [],
            "neutrals": [],
            "summary": {"total_wins": 0, "total_regressions": 0, "net_improvement": 0},
        }
    except Exception as e:
        logger.error("Validation subprocess error: %s", e)
        return {
            "error": f"Subprocess error: {e}",
            "wins": [],
            "regressions": [],
            "neutrals": [],
            "summary": {"total_wins": 0, "total_regressions": 0, "net_improvement": 0},
        }
    finally:
        # Clean up temp input file
        try:
            Path(input_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Clean up temp output file (only if we created it)
        if not output_path:
            try:
                Path(out_path).unlink(missing_ok=True)
            except Exception:
                pass


# Re-export for tests that call _apply_ruleselector in-process
from evolve_loop.validate_worker import _apply_ruleselector  # noqa: F401, E402
