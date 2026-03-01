"""
Validation worker subprocess for the evolution loop.

Runs in a separate Python process to get a fresh JVM with the rebuilt JAR.
This is necessary because JPype cannot reload Java classes after the JVM
has been started — the main process's JVM always has the OLD RuleSelector.

Usage (called by validator.py, not directly):
    python -m evolve_loop.validate_worker --input /path/to/input.json --output /path/to/output.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_worker")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _apply_ruleselector(sql: str, create_tables: str) -> Optional[str]:
    """Apply the RuleSelector to a query (empty rules triggers RuleSelector)."""
    import jpype as jp

    if not jp.isJVMStarted():
        raise RuntimeError("JVM not started")

    try:
        from evolve_loop.utils.java_bridge import _parse_create_tables, _to_java_list, _Rewriter

        if _Rewriter is None:
            raise RuntimeError("Rewriter not loaded")

        create_stmts = _parse_create_tables(create_tables)
        result = _Rewriter.rewrite(
            jp.JString(sql),
            _to_java_list(create_stmts),
            _to_java_list([]),  # Empty rules triggers RuleSelector
            jp.JInt(0),
            jp.JString("postgresql"),
        )

        if result and hasattr(result, "sql"):
            rewritten = str(result.sql)
            orig_norm = " ".join(sql.split()).lower()
            new_norm = " ".join(rewritten.split()).lower()
            if orig_norm == new_norm:
                return None
            return rewritten
        return None
    except Exception as e:
        logger.debug("RuleSelector application failed: %s", e)
        return None


def validate_queries(params: dict) -> dict:
    """Run validation on all queries with a fresh JVM.

    Args:
        params: Dict with keys: jar_dir, queries, baselines, query_dirs,
                pg_configs, validation_runs, validation_warmup,
                search_timeout_sec, win_threshold, regression_threshold

    Returns:
        Validation report dict.
    """
    from evolve_loop.utils.java_bridge import init_jvm
    from evolve_loop.features import load_query
    from evolve_loop.utils.pg_runner import get_connection, measure_latency_robust

    # Start fresh JVM with the rebuilt JAR
    jar_dir = params["jar_dir"]
    logger.info("Starting JVM with JAR dir: %s", jar_dir)
    init_jvm(jar_dir)

    queries = [tuple(q) for q in params["queries"]]
    baselines = params["baselines"]
    query_dirs = params["query_dirs"]
    pg_configs = params["pg_configs"]
    validation_runs = params.get("validation_runs", 5)
    validation_warmup = params.get("validation_warmup", 1)
    search_timeout_sec = params.get("search_timeout_sec", 120)
    win_threshold = params.get("win_threshold", 1.10)
    regression_threshold = params.get("regression_threshold", 0.95)
    train_suffix = params.get("train_suffix", "")

    wins = []
    regressions = []
    neutrals = []
    errors = []
    start_time = time.time()

    for i, (benchmark, query_id) in enumerate(queries, 1):
        if query_id not in baselines:
            continue

        elapsed = time.time() - start_time
        logger.info(
            "[%d/%d] Validating %s/%s (elapsed: %.1fmin)",
            i, len(queries), benchmark, query_id, elapsed / 60,
        )

        try:
            sql, create_tables = load_query(query_dirs[benchmark], query_id)
        except FileNotFoundError:
            continue

        rewritten = _apply_ruleselector(sql, create_tables)
        split = "train" if train_suffix and query_id.endswith(train_suffix) else "test"
        if rewritten is None:
            neutrals.append({
                "query": query_id,
                "benchmark": benchmark,
                "baseline": baselines[query_id],
                "new": baselines[query_id],
                "speedup": 1.0,
                "status": "NO_CHANGE",
                "split": split,
            })
            continue

        conn = get_connection(pg_configs[benchmark])
        lat, status = measure_latency_robust(
            conn, rewritten,
            runs=validation_runs,
            warmup=validation_warmup,
            timeout_sec=search_timeout_sec,
        )
        conn.close()

        baseline_lat = baselines[query_id]

        if lat is None:
            errors.append({
                "query": query_id,
                "benchmark": benchmark,
                "baseline": baseline_lat,
                "status": status,
            })
            regressions.append({
                "query": query_id,
                "benchmark": benchmark,
                "baseline": baseline_lat,
                "new": None,
                "speedup": 0.0,
                "status": status,
                "split": split,
            })
            continue

        speedup = baseline_lat / lat if lat > 0 else 0
        split = "train" if train_suffix and query_id.endswith(train_suffix) else "test"
        entry = {
            "query": query_id,
            "benchmark": benchmark,
            "baseline": round(baseline_lat, 4),
            "new": round(lat, 4),
            "speedup": round(speedup, 3),
            "split": split,
        }

        if speedup > win_threshold:
            wins.append(entry)
            logger.info("  WIN: %s %.4fs -> %.4fs (%.2fx)", query_id, baseline_lat, lat, speedup)
        elif speedup < regression_threshold:
            regressions.append(entry)
            logger.warning(
                "  REGRESSION: %s %.4fs -> %.4fs (%.2fx)",
                query_id, baseline_lat, lat, speedup,
            )
        else:
            neutrals.append(entry)

    elapsed = time.time() - start_time

    # Build split summary
    split_summary = {}
    if train_suffix:
        for split_name in ("train", "test"):
            split_summary[split_name] = {
                "wins": sum(1 for e in wins if e.get("split") == split_name),
                "regressions": sum(1 for e in regressions if e.get("split") == split_name),
                "neutrals": sum(1 for e in neutrals if e.get("split") == split_name),
            }

    report = {
        "timestamp": datetime.now().isoformat(),
        "wins": wins,
        "regressions": regressions,
        "neutrals": neutrals,
        "errors": errors,
        "summary": {
            "total_queries": len(queries),
            "queries_with_baselines": len(baselines),
            "total_wins": len(wins),
            "total_regressions": len(regressions),
            "total_neutrals": len(neutrals),
            "total_errors": len(errors),
            "net_improvement": len(wins) - len(regressions),
            "elapsed_sec": round(elapsed, 1),
        },
    }
    if split_summary:
        report["split_summary"] = split_summary

    logger.info(
        "Validation complete: %d wins, %d regressions, %d neutrals, %d errors (%.1fs)",
        len(wins), len(regressions), len(neutrals), len(errors), elapsed,
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Validation worker subprocess")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        params = json.load(f)

    try:
        report = validate_queries(params)
    except Exception as e:
        logger.error("Validation worker failed: %s", e, exc_info=True)
        report = {
            "error": f"Worker failed: {e}",
            "wins": [],
            "regressions": [],
            "neutrals": [],
            "errors": [],
            "summary": {
                "total_wins": 0,
                "total_regressions": 0,
                "net_improvement": 0,
            },
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
