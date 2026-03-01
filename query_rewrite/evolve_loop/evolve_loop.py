"""
Main orchestrator for the Calcite rule evolution loop.

Drives 5-phase iterations:
  1. SEARCH: Test rule combos on queries (compute)
  2. ANALYZE: Find patterns in results (Claude API)
  3. IMPLEMENT: Generate Java code (Claude API)
  4. VALIDATE: Regression test (compute)
  5. FIX: Repair regressions (Claude API)

Usage:
    python -m evolve_loop.evolve_loop --config config.yaml
    python -m evolve_loop.evolve_loop --config config.yaml --resume
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig, load_config
from evolve_loop.state import (
    EvolutionState,
    IterationRecord,
    is_converged,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


def _setup_logging(checkpoint_dir: str, verbose: bool = False) -> None:
    """Configure logging to both console and file."""
    log_dir = Path(checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # File handler
    log_file = log_dir / "evolution.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(fh)


def _initialize_state(config: EvolutionConfig, resume: bool) -> EvolutionState:
    """Initialize or resume evolution state."""
    if resume:
        state = load_checkpoint(config)
        if state:
            logger.info("Resumed from iteration %d", state.iteration)
            return state
        logger.warning("No checkpoint found, starting fresh")

    state = EvolutionState()
    state.started_at = datetime.now().isoformat()

    # Load initial RuleSelector.java
    rs_path = Path(config.canonical_ruleselector)
    if rs_path.exists():
        state.ruleselector_code = rs_path.read_text()
        logger.info("Loaded initial RuleSelector.java (%d chars)", len(state.ruleselector_code))
    else:
        logger.warning("Canonical RuleSelector.java not found at %s", config.canonical_ruleselector)

    return state


def _discover_queries(
    config: EvolutionConfig,
    train_only: bool = False,
) -> List[Tuple[str, str]]:
    """Discover all available queries, optionally filtered to a subset.

    Args:
        config: Evolution config
        train_only: If True and config.train_suffix is set, only return
            queries whose ID ends with that suffix (e.g. "_0").
            Used during search phases to halve evaluation cost.
    """
    from evolve_loop.features import discover_queries
    queries = discover_queries(config.query_dirs)

    # Apply subset filter if configured
    if config.query_subset:
        subset_set = set()
        for bench, qids in config.query_subset.items():
            for qid in qids:
                subset_set.add((bench, qid))
        queries = [q for q in queries if q in subset_set]
        logger.info("Filtered to %d subset queries across %d benchmarks",
                     len(queries), len(config.query_subset))
    else:
        logger.info("Discovered %d queries across %d benchmarks", len(queries), len(config.benchmarks))

    # Apply train/test split if configured
    if train_only and config.train_suffix:
        queries = [(bm, qid) for bm, qid in queries if qid.endswith(config.train_suffix)]
        logger.info("Train-only filter (%s): %d queries", config.train_suffix, len(queries))

    return queries


def _collect_baselines_if_needed(
    state: EvolutionState,
    queries: List[Tuple[str, str]],
    config: EvolutionConfig,
) -> None:
    """Collect baselines if not already present.

    Checks three sources in order:
    1. In-memory (from checkpoint resume)
    2. On-disk baselines.json (preserved across runs for slow queries)
    3. Fresh collection (measures all queries, subject to timeout)
    """
    if state.baselines:
        logger.info("Using %d cached baselines (from checkpoint)", len(state.baselines))
        return

    baselines_path = Path(config.checkpoint_dir) / "baselines.json"
    if baselines_path.exists():
        import json
        state.baselines = json.loads(baselines_path.read_text())
        logger.info("Loaded %d baselines from disk (%s)", len(state.baselines), baselines_path)
        return

    from evolve_loop.validator import collect_baselines
    state.baselines = collect_baselines(queries, config, output_path=str(baselines_path))
    logger.info("Collected %d fresh baselines", len(state.baselines))


def _build_features_if_needed(
    state: EvolutionState,
    queries: List[Tuple[str, str]],
    config: EvolutionConfig,
) -> None:
    """Build feature matrix if not already present."""
    if state.feature_matrix:
        logger.info("Using %d cached features (from checkpoint)", len(state.feature_matrix))
        return

    features_path = Path(config.checkpoint_dir) / "feature_matrix.json"
    if features_path.exists():
        import json
        state.feature_matrix = json.loads(features_path.read_text())
        logger.info("Loaded %d features from disk (%s)", len(state.feature_matrix), features_path)
        return

    from evolve_loop.features import load_query, extract_features_for_query
    from evolve_loop.utils.java_bridge import init_jvm

    init_jvm(config.jar_dir)

    for benchmark, query_id in queries:
        try:
            sql, create_tables = load_query(config.query_dirs[benchmark], query_id)
            features = extract_features_for_query(sql, create_tables)
            if features:
                state.feature_matrix[query_id] = features
        except Exception as e:
            logger.debug("Feature extraction failed for %s: %s", query_id, e)

    logger.info("Extracted features for %d queries", len(state.feature_matrix))


def run_iteration(
    state: EvolutionState,
    search_queries: List[Tuple[str, str]],
    all_queries: List[Tuple[str, str]],
    config: EvolutionConfig,
) -> IterationRecord:
    """Run a single evolution iteration (phases 1-5).

    Args:
        state: Current evolution state
        search_queries: Queries to use for search (train split)
        all_queries: All queries for validation (train + test)
        config: Evolution config

    Returns an IterationRecord summarizing what happened.
    """
    state.iteration += 1
    record = IterationRecord(
        iteration=state.iteration,
        timestamp=datetime.now().isoformat(),
        wins_before=state.total_wins,
    )

    logger.info("=" * 60)
    logger.info("ITERATION %d", state.iteration)
    logger.info("=" * 60)

    # ---- Phase 1: SEARCH ----
    logger.info("--- Phase 1: SEARCH ---")
    try:
        from evolve_loop.searcher import (
            generate_bootstrap_combos, generate_combos, search_all,
            generate_search_plan, search_planned,
        )
        from evolve_loop.state import update_search_history
        from evolve_loop.utils.java_bridge import init_jvm

        init_jvm(config.jar_dir)

        search_path = str(
            Path(config.checkpoint_dir) / "iterations"
            / f"iter_{state.iteration:03d}" / "search_results.json"
        )

        use_adaptive = (
            state.iteration > config.bootstrap_iterations
            and state.pending_directives
        )

        if use_adaptive:
            # ADAPTIVE: use analyst's directives from previous iteration
            logger.info(
                "Using adaptive search (%d directives from previous iteration)",
                len(state.pending_directives),
            )
            search_plan = generate_search_plan(
                config, state.pending_directives, state.search_history,
                state.feature_matrix, search_queries,
            )
            record.search_queries = len(search_plan)
            record.search_combos = sum(len(combos) for _, _, combos in search_plan)
            search_results = search_planned(
                search_plan, config, output_path=search_path,
                discovered_dangers=state.search_history.regressing_combos,
            )
        else:
            # BOOTSTRAP: first iteration(s) or no directives — singles only
            logger.info("Using bootstrap search (singles only)")
            combos = generate_bootstrap_combos(config)
            record.search_combos = len(combos)
            record.search_queries = len(search_queries)
            search_results = search_all(
                search_queries, combos, config, output_path=search_path,
                discovered_dangers=state.search_history.regressing_combos,
            )

        # Fold results into search history
        update_search_history(state, search_results, search_queries, config)

        logger.info(
            "Search complete: %d queries, %d combos, %d queries with wins (history: %d tested)",
            record.search_queries, record.search_combos,
            search_results["summary"]["queries_with_wins"],
            len(state.search_history.tested),
        )
    except Exception as e:
        logger.error("Phase 1 (Search) failed: %s", e, exc_info=True)
        record.error = f"Search failed: {e}"
        return record

    # ---- Phase 2: ANALYZE ----
    logger.info("--- Phase 2: ANALYZE ---")
    try:
        from evolve_loop.analyst import analyze_results, _api_usage as analyst_usage
        _before = (analyst_usage["input_tokens"], analyst_usage["output_tokens"])

        hypotheses, directives = analyze_results(
            search_results, state.feature_matrix, state.ruleselector_code, config,
            search_history=state.search_history,
            iteration=state.iteration,
        )
        state.total_api_input_tokens += analyst_usage["input_tokens"] - _before[0]
        state.total_api_output_tokens += analyst_usage["output_tokens"] - _before[1]
        record.hypotheses_proposed = len(hypotheses)
        record.hypotheses_accepted = len(hypotheses)

        # Store directives for next iteration's adaptive search
        state.pending_directives = directives

        hyp_path = str(
            Path(config.checkpoint_dir) / "iterations"
            / f"iter_{state.iteration:03d}" / "hypotheses.json"
        )
        Path(hyp_path).parent.mkdir(parents=True, exist_ok=True)
        with open(hyp_path, "w") as f:
            json.dump({"hypotheses": hypotheses, "directives": directives}, f, indent=2)

        logger.info(
            "Analysis complete: %d hypotheses proposed, %d search directives for next iteration",
            len(hypotheses), len(directives),
        )

        if not hypotheses:
            logger.info("No new hypotheses — iteration complete (no changes)")
            return record

    except Exception as e:
        logger.error("Phase 2 (Analyze) failed: %s", e, exc_info=True)
        record.error = f"Analysis failed: {e}"
        save_checkpoint(state, config, search_results=search_results)
        return record

    # ---- Phase 3: IMPLEMENT ----
    logger.info("--- Phase 3: IMPLEMENT ---")
    try:
        from evolve_loop.implementer import implement_patterns, _api_usage as impl_usage
        _before = (impl_usage["input_tokens"], impl_usage["output_tokens"])

        qa_path = Path(config.canonical_queryanalyzer)
        qa_code = qa_path.read_text() if qa_path.exists() else ""

        new_code = implement_patterns(
            hypotheses, state.ruleselector_code, qa_code, config
        )
        state.total_api_input_tokens += impl_usage["input_tokens"] - _before[0]
        state.total_api_output_tokens += impl_usage["output_tokens"] - _before[1]

        if new_code is None:
            logger.warning("Implementation failed — no compilable code produced")
            record.error = "Implementation failed"
            save_checkpoint(state, config, search_results=search_results, hypotheses=hypotheses)
            return record

        # Count new patterns (approximate)
        old_patterns = state.ruleselector_code.count("return rules.toArray")
        new_patterns = new_code.count("return rules.toArray")
        record.new_patterns_added = max(0, new_patterns - old_patterns)
        logger.info("Implementation complete: %d new patterns added", record.new_patterns_added)

        # Save LLM-generated code for debugging (before guards, before revert)
        iter_dir = (
            Path(config.checkpoint_dir) / "iterations"
            / f"iter_{state.iteration:03d}"
        )
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "RuleSelector_llm.java").write_text(new_code)

    except Exception as e:
        logger.error("Phase 3 (Implement) failed: %s", e, exc_info=True)
        record.error = f"Implementation failed: {e}"
        save_checkpoint(state, config, search_results=search_results, hypotheses=hypotheses)
        return record

    # ---- Phase 4: VALIDATE ----
    logger.info("--- Phase 4: VALIDATE ---")
    try:
        from evolve_loop.validator import validate_all

        # Save Phase 4 (pre-guard) validation as separate file
        val_path_initial = str(
            Path(config.checkpoint_dir) / "iterations"
            / f"iter_{state.iteration:03d}" / "validation_initial.json"
        )
        validation = validate_all(
            new_code, all_queries, state.baselines, config, output_path=val_path_initial
        )

        if "error" in validation:
            raise RuntimeError(validation["error"])

        record.wins_after = validation["summary"]["total_wins"]
        record.regressions_found = validation["summary"]["total_regressions"]

        logger.info(
            "Validation: %d wins, %d regressions",
            record.wins_after, record.regressions_found,
        )

    except Exception as e:
        logger.error("Phase 4 (Validate) failed: %s", e, exc_info=True)
        record.error = f"Validation failed: {e}"
        save_checkpoint(
            state, config,
            search_results=search_results,
            hypotheses=hypotheses,
        )
        return record

    # ---- Phase 5: FIX (if regressions) ----
    if validation["regressions"]:
        logger.info("--- Phase 5: FIX (guards only) ---")

        try:
            from evolve_loop.fixer import fix_with_guards

            guarded_code, unguarded = fix_with_guards(
                validation["regressions"],
                validation.get("wins", []),
                state.feature_matrix,
                new_code,
            )

            if guarded_code != new_code:
                # Save guarded code for debugging
                (iter_dir / "RuleSelector_guarded.java").write_text(guarded_code)

                logger.info("Guards generated, re-validating...")
                re_val = validate_all(
                    guarded_code, all_queries, state.baselines, config
                )

                if "error" not in re_val:
                    new_code = guarded_code
                    pre_guard_regressions = len(validation.get("regressions", []))
                    validation = re_val
                    record.regressions_found = re_val["summary"]["total_regressions"]
                    record.wins_after = re_val["summary"]["total_wins"]
                    record.regressions_fixed = (
                        pre_guard_regressions
                        - re_val["summary"]["total_regressions"]
                    )
                    logger.info(
                        "After guards: %d wins, %d regressions",
                        re_val["summary"]["total_wins"],
                        re_val["summary"]["total_regressions"],
                    )
                else:
                    logger.warning("Guard re-validation failed: %s", re_val["error"])
        except Exception as e:
            logger.error("Guard generation failed: %s", e, exc_info=True)

        # Save the code before potential revert — this is the best version
        # produced by this iteration (with guards), needed for best-so-far tracking.
        best_code_this_iter = new_code

        # Accept/reject decision for remaining regressions
        if validation["regressions"]:
            worst_regression = min(
                r.get("speedup", 1.0) for r in validation["regressions"]
            )
            net_wins = record.wins_after - record.regressions_found

            # Accept if: no catastrophic regressions AND net positive
            if worst_regression >= 0.50 and net_wins > 0:
                logger.info(
                    "Accepting %d regressions (worst %.2fx) — "
                    "net %d wins (%d wins, %d regressions)",
                    len(validation["regressions"]),
                    worst_regression,
                    net_wins,
                    record.wins_after,
                    record.regressions_found,
                )
            else:
                logger.warning(
                    "Reverting: %d regressions (worst %.2fx), net wins %d",
                    len(validation["regressions"]),
                    worst_regression,
                    net_wins,
                )
                new_code = state.ruleselector_code
                record.error = (
                    f"Reverted: {len(validation['regressions'])} regressions "
                    f"(worst {worst_regression:.2f}x)"
                )
                record.patterns_disabled = record.new_patterns_added
                record.new_patterns_added = 0
    else:
        logger.info("No regressions — skipping Phase 5 (Fix)")
        best_code_this_iter = new_code

    # Update state with the (possibly new) code
    if new_code != state.ruleselector_code:
        state.ruleselector_code = new_code
        state.total_wins = record.wins_after

    # Update best-so-far if this iteration is better.
    # Use best_code_this_iter (pre-revert code) so we track the actual best
    # policy, not the canonical code we reverted to.
    current_wins = record.wins_after
    current_regs = record.regressions_found
    is_better = (
        current_wins > state.best_total_wins
        or (current_wins == state.best_total_wins
            and current_regs < state.best_total_regressions)
    )
    if is_better and current_wins > 0:
        state.best_ruleselector_code = best_code_this_iter
        state.best_total_wins = current_wins
        state.best_total_regressions = current_regs
        state.best_iteration = state.iteration
        logger.info(
            "New best policy: %d wins, %d regressions (iter %d)",
            current_wins, current_regs, state.iteration,
        )

    # Checkpoint
    save_checkpoint(
        state, config,
        search_results=search_results,
        hypotheses=hypotheses,
        validation_report=validation,
    )

    return record


def run_evolution(config: EvolutionConfig, resume: bool = False) -> EvolutionState:
    """Run the full evolution loop until convergence.

    Args:
        config: Evolution config
        resume: Whether to resume from last checkpoint

    Returns:
        Final EvolutionState
    """
    state = _initialize_state(config, resume)

    # Discover queries: all variants for baselines/features/validation,
    # train-only for search iterations.
    all_queries = _discover_queries(config, train_only=False)
    search_queries = _discover_queries(config, train_only=True)

    if len(search_queries) < len(all_queries):
        logger.info(
            "Train/test split: %d search queries, %d total queries",
            len(search_queries), len(all_queries),
        )

    # Initialize JVM for baseline collection and feature extraction
    from evolve_loop.utils.java_bridge import init_jvm
    init_jvm(config.jar_dir)

    # Collect baselines if needed (all queries — needed for validation)
    _collect_baselines_if_needed(state, all_queries, config)

    # Build feature matrix if needed (all queries)
    _build_features_if_needed(state, all_queries, config)

    # Save initial checkpoint
    save_checkpoint(state, config)

    # Main loop
    while not is_converged(state, config):
        iter_start = time.time()
        record = run_iteration(state, search_queries, all_queries, config)
        iter_elapsed = time.time() - iter_start

        state.history.append(record)
        logger.info(
            "Iteration %d complete (%.1fmin): wins=%d, regressions=%d, patterns=%d%s",
            record.iteration,
            iter_elapsed / 60,
            record.wins_after,
            record.regressions_found,
            record.new_patterns_added,
            f" ERROR: {record.error}" if record.error else "",
        )

        # Check for convergence
        if is_converged(state, config):
            logger.info("Evolution converged after %d iterations", state.iteration)
            break

    # Final summary
    logger.info("=" * 60)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total iterations: %d", state.iteration)
    logger.info("Total wins: %d", state.total_wins)
    logger.info("Total API tokens: %d input, %d output",
                state.total_api_input_tokens, state.total_api_output_tokens)

    # Save final state
    save_checkpoint(state, config)

    return state


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calcite Rule Evolution Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evolve_loop.evolve_loop --config evolve_loop/default_config.yaml
  python -m evolve_loop.evolve_loop --config config.yaml --resume
  python -m evolve_loop.evolve_loop --config config.yaml --max-iterations 5
        """,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max-iterations", type=int, help="Override max iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_iterations:
        config.max_iterations = args.max_iterations

    _setup_logging(config.checkpoint_dir, verbose=args.verbose)
    logger.info("Starting evolution loop with config: %s", args.config)

    try:
        state = run_evolution(config, resume=args.resume)
        logger.info("Evolution completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")
        return 1
    except Exception as e:
        logger.error("Evolution failed: %s", e, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
