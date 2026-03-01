"""
Combo searcher: tests rule combinations on queries and measures PostgreSQL latency.

Phase 1 of the evolution loop.
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig
from evolve_loop.features import load_query
from evolve_loop.utils.java_bridge import init_jvm, rewrite_sql
from evolve_loop.utils.pg_runner import get_connection, measure_latency

logger = logging.getLogger(__name__)


def generate_bootstrap_combos(config: EvolutionConfig) -> List[List[str]]:
    """Generate single-rule and 2-rule combos for bootstrap iteration.

    Tests each rule individually plus all pairs of main rules.
    This catches combo-dependent wins that no single rule produces.
    """
    combos = []
    seen = set()
    all_rules = config.main_rules + config.pre_finishers + config.post_finishers
    for r in all_rules:
        if r not in seen:
            combos.append([r])
            seen.add(r)

    # Add all 2-rule combos from main rules
    for i, r1 in enumerate(config.main_rules):
        for r2 in config.main_rules[i + 1:]:
            key = f"{r1}|{r2}"
            if key not in seen:
                combos.append([r1, r2])
                seen.add(key)

    return combos


def generate_combos(config: EvolutionConfig) -> List[List[str]]:
    """Generate all rule combinations from the pre/main/post template.

    Returns list of rule combos, where each combo is a list of rule names.
    """
    combos = []
    seen = set()

    all_singles = config.main_rules + config.pre_finishers + config.post_finishers

    # 1. Single rules
    for r in all_singles:
        if r not in seen:
            combos.append([r])
            seen.add(r)

    # 2. MAIN + POST
    for main in config.main_rules:
        for post in config.post_finishers:
            if main != post:
                key = f"{main}|{post}"
                if key not in seen:
                    combos.append([main, post])
                    seen.add(key)

    # 3. PRE + MAIN
    for pre in config.pre_finishers:
        for main in config.main_rules:
            if pre != main:
                key = f"{pre}|{main}"
                if key not in seen:
                    combos.append([pre, main])
                    seen.add(key)

    # 4. PRE + MAIN + POST
    for pre in config.pre_finishers:
        for main in config.main_rules:
            for post in config.post_finishers:
                if len({pre, main, post}) == 3:
                    key = f"{pre}|{main}|{post}"
                    if key not in seen:
                        combos.append([pre, main, post])
                        seen.add(key)

    # 5. Finisher combos (PRE + POST)
    for p1 in config.pre_finishers:
        for p2 in config.post_finishers:
            if p1 != p2:
                key = f"{p1}|{p2}"
                if key not in seen:
                    combos.append([p1, p2])
                    seen.add(key)

    return combos


def _combo_name(rules: List[str]) -> str:
    """Short name for a rule combo."""
    return "_".join(r.replace("_", "")[:3].lower() for r in rules)


def _is_dangerous(
    query_id: str,
    rules: List[str],
    config: EvolutionConfig,
    discovered_dangers: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Check if a rule combo is known-dangerous for this query."""
    has_fsq = any("FILTER_SUB_QUERY_TO_CORRELATE" in r for r in rules)
    has_fij = any("FILTER_INTO_JOIN" in r for r in rules)

    if has_fsq and any(d in query_id for d in config.fsq_dangerous):
        return True
    if has_fij and any(d in query_id for d in config.fij_dangerous):
        return True

    # Check empirically discovered dangers
    if discovered_dangers and query_id in discovered_dangers:
        combo_key = "|".join(rules)
        if combo_key in discovered_dangers[query_id]:
            return True

    return False


def search_query(
    query_id: str,
    benchmark: str,
    combos: List[List[str]],
    config: EvolutionConfig,
    discovered_dangers: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Test all combos on a single query and return results.

    Returns dict with structure:
    {
        "query_id": str,
        "benchmark": str,
        "original_latency": float or None,
        "wins": [{combo, latency, speedup}],
        "regressions": [{combo, latency, speedup}],
        "best_combo": [str] or None,
        "best_speedup": float,
    }
    """
    result = {
        "query_id": query_id,
        "benchmark": benchmark,
        "original_latency": None,
        "wins": [],
        "regressions": [],
        "best_combo": None,
        "best_speedup": 1.0,
    }

    try:
        sql, create_tables = load_query(config.query_dirs[benchmark], query_id)
    except FileNotFoundError as e:
        logger.warning("Skipping %s: %s", query_id, e)
        result["error"] = str(e)
        return result

    # Measure original latency
    conn = get_connection(config.pg_configs[benchmark])
    orig_lat, status = measure_latency(conn, sql, config.search_timeout_sec)
    if orig_lat is None:
        logger.info("%s: original %s", query_id, status)
        conn.close()
        result["error"] = f"original: {status}"
        return result

    result["original_latency"] = orig_lat
    logger.info("%s: original %.3fs", query_id, orig_lat)

    # Test each combo
    for combo in combos:
        if _is_dangerous(query_id, combo, config, discovered_dangers):
            continue

        rewritten = rewrite_sql(sql, create_tables, combo)
        if rewritten is None:
            continue

        lat, status = measure_latency(conn, rewritten, config.search_timeout_sec)
        if lat is None:
            continue

        speedup = orig_lat / lat if lat > 0 else 0
        combo_info = {
            "combo": combo,
            "combo_name": _combo_name(combo),
            "latency": lat,
            "speedup": round(speedup, 3),
        }

        if speedup > config.win_threshold:
            result["wins"].append(combo_info)
            if speedup > result["best_speedup"]:
                result["best_speedup"] = round(speedup, 3)
                result["best_combo"] = combo

        elif speedup < config.regression_threshold:
            result["regressions"].append(combo_info)

    conn.close()
    return result


def search_all(
    queries: List[Tuple[str, str]],
    combos: List[List[str]],
    config: EvolutionConfig,
    output_path: Optional[str] = None,
    discovered_dangers: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Run combo search on all queries (sequentially - JVM is not fork-safe).

    Args:
        queries: List of (benchmark, query_id) tuples
        combos: List of rule combos to test
        config: Evolution config
        output_path: Optional path to save JSON results

    Returns:
        Dict with "queries" (list of results) and "summary"
    """
    init_jvm(config.jar_dir)

    all_results = []
    total_wins = 0
    start_time = time.time()

    for i, (benchmark, query_id) in enumerate(queries, 1):
        elapsed = time.time() - start_time
        logger.info(
            "[%d/%d] %s/%s (elapsed: %.1fmin)",
            i, len(queries), benchmark, query_id, elapsed / 60,
        )

        result = search_query(query_id, benchmark, combos, config, discovered_dangers)
        all_results.append(result)

        n_wins = len(result.get("wins", []))
        total_wins += n_wins
        if result.get("best_combo"):
            logger.info(
                "  BEST: %s -> %.2fx",
                _combo_name(result["best_combo"]),
                result["best_speedup"],
            )

    elapsed = time.time() - start_time

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_queries": len(queries),
            "num_combos": len(combos),
            "search_timeout_sec": config.search_timeout_sec,
        },
        "summary": {
            "total_queries": len(queries),
            "queries_with_wins": sum(1 for r in all_results if r.get("wins")),
            "total_wins": total_wins,
            "elapsed_sec": round(elapsed, 1),
        },
        "queries": all_results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Search results saved to %s", output_path)

    return output


# --- Adaptive search (iteration 2+) ---


def _match_feature_filter(
    query_id: str,
    feature_filter: Dict,
    feature_matrix: Dict[str, Dict],
) -> bool:
    """Check if a query's features match a filter dict.

    Supports exact values, [min, max] ranges, and booleans.
    Returns False if the query has no features or any condition fails.
    """
    features = feature_matrix.get(query_id)
    if features is None:
        return False

    for key, condition in feature_filter.items():
        actual = features.get(key)
        if actual is None:
            return False
        if isinstance(condition, list) and len(condition) == 2:
            # Range: [min, max]
            if not (condition[0] <= actual <= condition[1]):
                return False
        elif isinstance(condition, bool):
            if actual != condition:
                return False
        else:
            # Exact match
            if actual != condition:
                return False
    return True


def _inject_exploration_combos(
    query_plan: Dict[str, List[List[str]]],
    config: EvolutionConfig,
    search_history: "SearchHistory",
    qid_to_benchmark: Dict[str, str],
    iteration: int = 0,
) -> int:
    """Inject broad exploration multi-rule combos into the search plan.

    Ensures every adaptive iteration tests some untried multi-rule combos,
    preventing the analyst from stalling on zero wins.

    Args:
        query_plan: Mutable plan dict to inject into.
        config: Evolution config.
        search_history: Accumulated search history.
        qid_to_benchmark: Query ID to benchmark mapping.
        iteration: Current iteration number (used as RNG seed).

    Returns:
        Number of (query, combo) pairs injected.
    """
    import random

    rng = random.Random(iteration)

    # Generate multi-rule combos only (singles already saturated by bootstrap)
    all_combos = generate_combos(config)
    multi_combos = [c for c in all_combos if len(c) >= 2]
    if not multi_combos:
        return 0

    # Pick candidates: untested first, then neutral, then winning
    candidates = list(search_history.untested_queries)
    candidates += list(search_history.neutral_queries)
    candidates += list(search_history.winning_queries)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for q in candidates:
        if q not in seen and q in qid_to_benchmark:
            unique.append(q)
            seen.add(q)
    candidates = unique

    if not candidates:
        return 0

    # Sample up to 10 queries
    n_queries = min(10, len(candidates))
    selected = rng.sample(candidates, n_queries)

    added = 0
    for qid in selected:
        # Find untested multi-rule combos for this query
        untested = [
            c for c in multi_combos
            if not _already_tested(qid, c, search_history.tested)
        ]
        if not untested:
            continue
        # Sample 2 combos per query
        sampled = rng.sample(untested, min(2, len(untested)))
        for combo in sampled:
            _add_to_plan(query_plan, qid, combo, search_history.tested)
            added += 1

    if added:
        logger.info("Injected %d exploration (query, multi-combo) pairs", added)
    return added


def _already_tested(
    query_id: str,
    combo: List[str],
    tested: Dict[str, Dict[str, float]],
) -> bool:
    """Check if a (query, combo) pair was already tested."""
    combo_key = "|".join(combo)
    return combo_key in tested.get(query_id, {})


def generate_search_plan(
    config: EvolutionConfig,
    directives: List[Dict],
    search_history: "SearchHistory",
    feature_matrix: Dict[str, Dict],
    all_queries: List[Tuple[str, str]],
) -> List[Tuple[str, str, List[List[str]]]]:
    """Generate targeted (query, combos) pairs based on analyst directives.

    Processes each directive by strategy type, deduplicates against
    search history, and returns a search plan where each query gets
    its own combo list.

    Args:
        config: Evolution config
        directives: Search directives from the analyst
        search_history: Accumulated search history
        feature_matrix: Feature dict per query
        all_queries: Complete list of (benchmark, query_id) tuples

    Returns:
        List of (benchmark, query_id, [combos]) tuples to test
    """
    # Build lookup: query_id -> benchmark
    qid_to_benchmark = {qid: bench for bench, qid in all_queries}

    # Accumulate (query_id, combo) pairs from all directives
    # query_plan[query_id] = set of combo_keys
    query_plan: Dict[str, List[List[str]]] = {}

    # Sort directives by priority (lower = higher priority)
    sorted_directives = sorted(
        directives,
        key=lambda d: d.get("priority", 99),
    )

    for directive in sorted_directives:
        strategy = directive.get("strategy", "")
        target_queries = directive.get("target_queries", [])
        target_combos = directive.get("target_combos", [])
        feature_filter = directive.get("feature_filter")
        base_combo = directive.get("base_combo")
        extensions = directive.get("extensions", [])

        try:
            if strategy == "expand_winners":
                _plan_expand_winners(
                    query_plan, target_combos, feature_filter,
                    search_history, feature_matrix, qid_to_benchmark, config,
                )

            elif strategy == "extend_combos":
                _plan_extend_combos(
                    query_plan, base_combo, extensions, target_queries,
                    search_history, qid_to_benchmark, config,
                )

            elif strategy == "gap_analysis":
                _plan_gap_analysis(
                    query_plan, target_queries, target_combos,
                    search_history, qid_to_benchmark, config,
                )

            elif strategy == "verify_hypothesis":
                _plan_verify_hypothesis(
                    query_plan, target_combos, feature_filter,
                    search_history, feature_matrix, qid_to_benchmark, config,
                )

            elif strategy == "regression_isolate":
                _plan_regression_isolate(
                    query_plan, target_queries, target_combos,
                    search_history, qid_to_benchmark, config,
                )

            elif strategy == "broad_sweep":
                _plan_broad_sweep(
                    query_plan, target_queries,
                    search_history, qid_to_benchmark, config,
                )

            else:
                logger.warning("Unknown directive strategy: %s", strategy)

        except Exception as e:
            logger.warning(
                "Error processing directive '%s': %s",
                directive.get("description", strategy), e,
            )

    # Always inject some broad exploration multi-rule combos
    _inject_exploration_combos(
        query_plan, config, search_history, qid_to_benchmark,
    )

    if not query_plan:
        logger.warning("No adaptive search plan generated, falling back to broad sweep on untested queries")
        untested = list(search_history.untested_queries)[:config.adaptive_max_queries_per_iter]
        if untested:
            combos = generate_combos(config)
            for qid in untested:
                if qid in qid_to_benchmark:
                    query_plan[qid] = combos
        else:
            logger.warning("No untested queries either — nothing to search")

    # Convert to output format, applying caps
    result = []
    queries_added = 0
    for qid, combos in query_plan.items():
        if queries_added >= config.adaptive_max_queries_per_iter:
            break
        if qid not in qid_to_benchmark:
            continue
        benchmark = qid_to_benchmark[qid]
        # Cap combos per query
        capped = combos[:config.adaptive_max_combos_per_query]
        if capped:
            result.append((benchmark, qid, capped))
            queries_added += 1

    total_tests = sum(len(combos) for _, _, combos in result)
    logger.info(
        "Adaptive search plan: %d queries, %d total combo tests",
        len(result), total_tests,
    )
    return result


def _add_to_plan(
    query_plan: Dict[str, List[List[str]]],
    qid: str,
    combo: List[str],
    tested: Dict[str, Dict[str, float]],
    dedup: bool = True,
) -> None:
    """Add a (query, combo) pair to the plan, deduplicating against history."""
    if dedup and _already_tested(qid, combo, tested):
        return
    if qid not in query_plan:
        query_plan[qid] = []
    # Deduplicate within the plan too
    combo_key = "|".join(combo)
    for existing in query_plan[qid]:
        if "|".join(existing) == combo_key:
            return
    query_plan[qid].append(combo)


def _plan_expand_winners(
    query_plan, target_combos, feature_filter,
    search_history, feature_matrix, qid_to_benchmark, config,
):
    """Expand winning combos to queries with similar features."""
    combos = target_combos
    if not combos and search_history.best_combos:
        # Use the top winning combos
        sorted_best = sorted(
            search_history.best_combos.values(),
            key=lambda x: x.get("speedup", 0),
            reverse=True,
        )
        combos = [b["combo"] for b in sorted_best[:5]]

    for qid in qid_to_benchmark:
        if feature_filter and not _match_feature_filter(qid, feature_filter, feature_matrix):
            continue
        for combo in combos:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def _plan_extend_combos(
    query_plan, base_combo, extensions, target_queries,
    search_history, qid_to_benchmark, config,
):
    """Try adding extensions to a winning base combo."""
    if not base_combo or not extensions:
        return

    # Generate extended combos
    extended_combos = []
    for ext in extensions:
        if ext not in base_combo:
            # Try as pre-finisher
            extended_combos.append([ext] + base_combo)
            # Try as post-finisher
            extended_combos.append(base_combo + [ext])

    # Target queries: either specified or queries where base already won
    if not target_queries:
        base_key = "|".join(base_combo)
        target_queries = [
            qid for qid, tested in search_history.tested.items()
            if base_key in tested and tested[base_key] > config.win_threshold
        ]

    for qid in target_queries:
        if qid not in qid_to_benchmark:
            continue
        for combo in extended_combos:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def _plan_gap_analysis(
    query_plan, target_queries, target_combos,
    search_history, qid_to_benchmark, config,
):
    """Test specific combos on specific gap queries."""
    if not target_queries:
        # Default to neutral queries (tested but no wins)
        target_queries = list(search_history.neutral_queries)[:20]
    if not target_combos:
        return

    for qid in target_queries:
        if qid not in qid_to_benchmark:
            continue
        for combo in target_combos:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def _plan_verify_hypothesis(
    query_plan, target_combos, feature_filter,
    search_history, feature_matrix, qid_to_benchmark, config,
):
    """Test target combos on ALL queries matching a feature filter."""
    if not target_combos:
        return

    for qid in qid_to_benchmark:
        if feature_filter and not _match_feature_filter(qid, feature_filter, feature_matrix):
            continue
        for combo in target_combos:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def _plan_regression_isolate(
    query_plan, target_queries, target_combos,
    search_history, qid_to_benchmark, config,
):
    """Test subsets of regressing combos to isolate the problematic rule."""
    if target_combos:
        # Use specified subsets
        combos_to_test = target_combos
    else:
        # Auto-generate subsets from regressing combos
        combos_to_test = []
        for qid in (target_queries or []):
            for combo_key in search_history.regressing_combos.get(qid, []):
                combo = combo_key.split("|")
                if len(combo) > 1:
                    for i in range(len(combo)):
                        subset = combo[:i] + combo[i + 1:]
                        combos_to_test.append(subset)

    if not target_queries:
        target_queries = list(search_history.regressing_combos.keys())[:10]

    for qid in target_queries:
        if qid not in qid_to_benchmark:
            continue
        for combo in combos_to_test:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def _plan_broad_sweep(
    query_plan, target_queries,
    search_history, qid_to_benchmark, config,
):
    """Full static combo set on specific target queries."""
    combos = generate_combos(config)

    if not target_queries:
        target_queries = list(search_history.untested_queries)[:config.adaptive_max_queries_per_iter]

    for qid in target_queries:
        if qid not in qid_to_benchmark:
            continue
        for combo in combos:
            _add_to_plan(query_plan, qid, combo, search_history.tested)


def search_planned(
    search_plan: List[Tuple[str, str, List[List[str]]]],
    config: EvolutionConfig,
    output_path: Optional[str] = None,
    discovered_dangers: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Run adaptive search where each query gets its own combo list.

    Like search_all() but each query has a targeted combo list based on
    analyst directives. Same return format for downstream compatibility.

    Args:
        search_plan: List of (benchmark, query_id, [combos]) tuples
        config: Evolution config
        output_path: Optional path to save JSON results

    Returns:
        Dict with "queries" (list of results) and "summary"
    """
    init_jvm(config.jar_dir)

    all_results = []
    total_wins = 0
    total_combos = 0
    start_time = time.time()

    for i, (benchmark, query_id, combos) in enumerate(search_plan, 1):
        elapsed = time.time() - start_time
        logger.info(
            "[%d/%d] %s/%s (%d combos, elapsed: %.1fmin)",
            i, len(search_plan), benchmark, query_id,
            len(combos), elapsed / 60,
        )

        result = search_query(query_id, benchmark, combos, config, discovered_dangers)
        all_results.append(result)
        total_combos += len(combos)

        n_wins = len(result.get("wins", []))
        total_wins += n_wins
        if result.get("best_combo"):
            logger.info(
                "  BEST: %s -> %.2fx",
                _combo_name(result["best_combo"]),
                result["best_speedup"],
            )

    elapsed = time.time() - start_time

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_queries": len(search_plan),
            "num_combos": total_combos,
            "search_timeout_sec": config.search_timeout_sec,
            "mode": "adaptive",
        },
        "summary": {
            "total_queries": len(search_plan),
            "queries_with_wins": sum(1 for r in all_results if r.get("wins")),
            "total_wins": total_wins,
            "elapsed_sec": round(elapsed, 1),
        },
        "queries": all_results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Adaptive search results saved to %s", output_path)

    return output
