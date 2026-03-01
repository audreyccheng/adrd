"""
State management and checkpointing for the evolution loop.

Provides:
- EvolutionState: Tracks current iteration, RuleSelector code, baselines, history
- save_checkpoint(): Save state to disk
- load_checkpoint(): Load state from latest checkpoint
- is_converged(): Check convergence criteria
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from evolve_loop.config import EvolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchHistory:
    """Accumulated search knowledge across all iterations.

    Tracks what (query, combo) pairs have been tested, their results,
    and categorizes queries by outcome for adaptive search planning.
    """
    # Core data: tested[query_id][combo_key] = speedup
    # combo_key = "|".join(combo), order-preserving since rule order matters
    tested: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Best known result per query: {combo, speedup, iteration}
    best_combos: Dict[str, Dict] = field(default_factory=dict)

    # Queries grouped by outcome
    winning_queries: Set[str] = field(default_factory=set)
    neutral_queries: Set[str] = field(default_factory=set)
    untested_queries: Set[str] = field(default_factory=set)

    # Dangerous combos discovered empirically: query_id -> [combo_keys that regressed]
    regressing_combos: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Record of a single evolution iteration."""
    iteration: int
    timestamp: str
    search_queries: int = 0
    search_combos: int = 0
    hypotheses_proposed: int = 0
    hypotheses_accepted: int = 0
    wins_before: int = 0
    wins_after: int = 0
    regressions_found: int = 0
    regressions_fixed: int = 0
    fix_attempts: int = 0
    new_patterns_added: int = 0
    patterns_disabled: int = 0
    error: Optional[str] = None


@dataclass
class EvolutionState:
    """Full state of the evolution loop."""
    iteration: int = 0
    ruleselector_code: str = ""
    baselines: Dict[str, float] = field(default_factory=dict)
    feature_matrix: Dict[str, Dict] = field(default_factory=dict)
    history: List[IterationRecord] = field(default_factory=list)
    total_wins: int = 0
    total_api_input_tokens: int = 0
    total_api_output_tokens: int = 0
    started_at: str = ""
    last_checkpoint: str = ""
    # Adaptive search state
    search_history: SearchHistory = field(default_factory=SearchHistory)
    pending_directives: List[Dict] = field(default_factory=list)
    # Best-so-far tracking
    best_ruleselector_code: str = ""
    best_total_wins: int = 0
    best_total_regressions: int = 0
    best_iteration: int = 0


def update_search_history(
    state: EvolutionState,
    search_results: Dict,
    all_queries: List[Tuple[str, str]],
    config: EvolutionConfig,
) -> None:
    """Fold new search results into the accumulated search history.

    Called after Phase 1 (search) completes each iteration. Updates
    the tested pairs, best combos, and query category sets.

    Args:
        state: Current evolution state (search_history is modified in-place)
        search_results: Output from searcher.search_all() or search_planned()
        all_queries: Complete list of (benchmark, query_id) tuples
        config: Evolution config (for win_threshold)
    """
    history = state.search_history
    all_query_ids = {qid for _, qid in all_queries}

    for qr in search_results.get("queries", []):
        qid = qr["query_id"]

        if qid not in history.tested:
            history.tested[qid] = {}

        # Record wins
        for w in qr.get("wins", []):
            combo_key = "|".join(w["combo"])
            history.tested[qid][combo_key] = w["speedup"]
            history.winning_queries.add(qid)
            history.neutral_queries.discard(qid)

            # Update best combo for this query
            if (qid not in history.best_combos
                    or w["speedup"] > history.best_combos[qid].get("speedup", 0)):
                history.best_combos[qid] = {
                    "combo": w["combo"],
                    "speedup": w["speedup"],
                    "iteration": state.iteration,
                }

        # Record regressions
        for r in qr.get("regressions", []):
            combo_key = "|".join(r["combo"])
            history.tested[qid][combo_key] = r["speedup"]
            if qid not in history.regressing_combos:
                history.regressing_combos[qid] = []
            if combo_key not in history.regressing_combos[qid]:
                history.regressing_combos[qid].append(combo_key)

        # If query was tested but had no wins, mark neutral (unless already winning)
        if qid in history.tested and qid not in history.winning_queries:
            history.neutral_queries.add(qid)

    # Update untested set
    tested_ids = set(history.tested.keys())
    history.untested_queries = all_query_ids - tested_ids

    logger.info(
        "Search history updated: %d tested, %d winning, %d neutral, %d untested",
        len(history.tested), len(history.winning_queries),
        len(history.neutral_queries), len(history.untested_queries),
    )


def save_checkpoint(
    state: EvolutionState,
    config: EvolutionConfig,
    search_results: Optional[Dict] = None,
    hypotheses: Optional[List[Dict]] = None,
    validation_report: Optional[Dict] = None,
) -> str:
    """Save the current state to a checkpoint directory.

    Args:
        state: Current evolution state
        config: Evolution config
        search_results: Optional search results for this iteration
        hypotheses: Optional hypotheses for this iteration
        validation_report: Optional validation report for this iteration

    Returns:
        Path to the checkpoint directory
    """
    base_dir = Path(config.checkpoint_dir)
    iter_dir = base_dir / "iterations" / f"iter_{state.iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    current_dir = base_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    state.last_checkpoint = datetime.now().isoformat()

    # Save state.json (top-level)
    state_data = {
        "iteration": state.iteration,
        "total_wins": state.total_wins,
        "total_api_input_tokens": state.total_api_input_tokens,
        "total_api_output_tokens": state.total_api_output_tokens,
        "started_at": state.started_at,
        "last_checkpoint": state.last_checkpoint,
        "best_total_wins": state.best_total_wins,
        "best_total_regressions": state.best_total_regressions,
        "best_iteration": state.best_iteration,
        "history": [_iter_record_to_dict(r) for r in state.history],
    }
    _write_json(base_dir / "state.json", state_data)

    # Save baselines (top-level, only once)
    if state.baselines:
        _write_json(base_dir / "baselines.json", state.baselines)

    # Save feature matrix (top-level)
    if state.feature_matrix:
        _write_json(base_dir / "feature_matrix.json", state.feature_matrix)

    # Save search history (top-level)
    if state.search_history.tested:
        history_data = {
            "tested": state.search_history.tested,
            "best_combos": state.search_history.best_combos,
            "winning_queries": sorted(state.search_history.winning_queries),
            "neutral_queries": sorted(state.search_history.neutral_queries),
            "untested_queries": sorted(state.search_history.untested_queries),
            "regressing_combos": state.search_history.regressing_combos,
        }
        _write_json(base_dir / "search_history.json", history_data)

    # Save pending directives (top-level)
    if state.pending_directives:
        _write_json(base_dir / "pending_directives.json", state.pending_directives)

    # Save RuleSelector.java to both iteration dir and current/
    if state.ruleselector_code:
        (iter_dir / "RuleSelector.java").write_text(state.ruleselector_code)
        (current_dir / "RuleSelector.java").write_text(state.ruleselector_code)

    # Save best RuleSelector.java
    if state.best_ruleselector_code:
        (current_dir / "RuleSelector_best.java").write_text(state.best_ruleselector_code)

    # Save per-iteration artifacts
    if search_results:
        _write_json(iter_dir / "search_results.json", search_results)
    if hypotheses:
        _write_json(iter_dir / "hypotheses.json", hypotheses)
    if validation_report:
        _write_json(iter_dir / "validation_report.json", validation_report)

    # Save iteration log
    if state.history:
        _write_json(
            iter_dir / "iteration_log.json",
            _iter_record_to_dict(state.history[-1]),
        )

    logger.info("Checkpoint saved to %s", iter_dir)
    return str(iter_dir)


def load_checkpoint(config: EvolutionConfig) -> Optional[EvolutionState]:
    """Load state from the latest checkpoint.

    Args:
        config: Evolution config

    Returns:
        EvolutionState if checkpoint exists, None otherwise
    """
    base_dir = Path(config.checkpoint_dir)
    state_file = base_dir / "state.json"

    if not state_file.exists():
        logger.info("No checkpoint found at %s", base_dir)
        return None

    state_data = json.loads(state_file.read_text())
    state = EvolutionState()
    state.iteration = state_data.get("iteration", 0)
    state.total_wins = state_data.get("total_wins", 0)
    state.total_api_input_tokens = state_data.get("total_api_input_tokens", 0)
    state.total_api_output_tokens = state_data.get("total_api_output_tokens", 0)
    state.started_at = state_data.get("started_at", "")
    state.last_checkpoint = state_data.get("last_checkpoint", "")
    state.best_total_wins = state_data.get("best_total_wins", 0)
    state.best_total_regressions = state_data.get("best_total_regressions", 0)
    state.best_iteration = state_data.get("best_iteration", 0)

    # Load history
    for rec_data in state_data.get("history", []):
        state.history.append(_dict_to_iter_record(rec_data))

    # Load baselines
    baselines_file = base_dir / "baselines.json"
    if baselines_file.exists():
        state.baselines = json.loads(baselines_file.read_text())

    # Load feature matrix
    features_file = base_dir / "feature_matrix.json"
    if features_file.exists():
        state.feature_matrix = json.loads(features_file.read_text())

    # Load search history (backward-compatible: missing file = empty history)
    history_file = base_dir / "search_history.json"
    if history_file.exists():
        hdata = json.loads(history_file.read_text())
        state.search_history = SearchHistory(
            tested=hdata.get("tested", {}),
            best_combos=hdata.get("best_combos", {}),
            winning_queries=set(hdata.get("winning_queries", [])),
            neutral_queries=set(hdata.get("neutral_queries", [])),
            untested_queries=set(hdata.get("untested_queries", [])),
            regressing_combos=hdata.get("regressing_combos", {}),
        )

    # Load pending directives (backward-compatible: missing file = empty list)
    directives_file = base_dir / "pending_directives.json"
    if directives_file.exists():
        state.pending_directives = json.loads(directives_file.read_text())

    # Load current RuleSelector.java
    current_rs = base_dir / "current" / "RuleSelector.java"
    if current_rs.exists():
        state.ruleselector_code = current_rs.read_text()

    # Load best RuleSelector.java
    best_rs = base_dir / "current" / "RuleSelector_best.java"
    if best_rs.exists():
        state.best_ruleselector_code = best_rs.read_text()

    logger.info(
        "Checkpoint loaded: iteration=%d, wins=%d, baselines=%d queries",
        state.iteration, state.total_wins, len(state.baselines),
    )
    return state


def is_converged(state: EvolutionState, config: EvolutionConfig) -> bool:
    """Check if the evolution loop has converged.

    Convergence criteria (any one triggers):
    1. Max iterations reached
    2. No new wins in last 3 iterations
    3. All iterations produced zero hypotheses

    Args:
        state: Current evolution state
        config: Evolution config

    Returns:
        True if converged
    """
    # Max iterations
    if state.iteration >= config.max_iterations:
        logger.info("Converged: max iterations (%d) reached", config.max_iterations)
        return True

    # No new wins in last 3 iterations
    if len(state.history) >= 3:
        last_3 = state.history[-3:]
        new_wins = sum(
            max(0, r.wins_after - r.wins_before)
            for r in last_3
        )
        if new_wins == 0:
            logger.info("Converged: no new wins in last 3 iterations")
            return True

    # All recent iterations produced zero hypotheses
    if len(state.history) >= 3:
        last_3 = state.history[-3:]
        if all(r.hypotheses_proposed == 0 for r in last_3):
            logger.info("Converged: no hypotheses proposed in last 3 iterations")
            return True

    return False


def _iter_record_to_dict(record: IterationRecord) -> Dict:
    """Convert an IterationRecord to a serializable dict."""
    return {
        "iteration": record.iteration,
        "timestamp": record.timestamp,
        "search_queries": record.search_queries,
        "search_combos": record.search_combos,
        "hypotheses_proposed": record.hypotheses_proposed,
        "hypotheses_accepted": record.hypotheses_accepted,
        "wins_before": record.wins_before,
        "wins_after": record.wins_after,
        "regressions_found": record.regressions_found,
        "regressions_fixed": record.regressions_fixed,
        "fix_attempts": record.fix_attempts,
        "new_patterns_added": record.new_patterns_added,
        "patterns_disabled": record.patterns_disabled,
        "error": record.error,
    }


def _dict_to_iter_record(data: Dict) -> IterationRecord:
    """Convert a dict to an IterationRecord."""
    return IterationRecord(
        iteration=data.get("iteration", 0),
        timestamp=data.get("timestamp", ""),
        search_queries=data.get("search_queries", 0),
        search_combos=data.get("search_combos", 0),
        hypotheses_proposed=data.get("hypotheses_proposed", 0),
        hypotheses_accepted=data.get("hypotheses_accepted", 0),
        wins_before=data.get("wins_before", 0),
        wins_after=data.get("wins_after", 0),
        regressions_found=data.get("regressions_found", 0),
        regressions_fixed=data.get("regressions_fixed", 0),
        fix_attempts=data.get("fix_attempts", 0),
        new_patterns_added=data.get("new_patterns_added", 0),
        patterns_disabled=data.get("patterns_disabled", 0),
        error=data.get("error"),
    )


def _write_json(path: Path, data) -> None:
    """Write data as JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
