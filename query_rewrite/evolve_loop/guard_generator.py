"""
Deterministic guard generator for regression fixing.

Generates early-exit guards from verified feature values to block
known-regressing queries. Guards fire at the top of the EVOLVE-BLOCK,
before any pattern can match, so they work regardless of which pattern
would have caught the query.

This runs as Stage 1 of Phase 5 (before the LLM fixer).
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Core features used in guards. Maps Python feature_matrix keys to Java expressions.
# Features available as local variables use the variable name directly.
# Features not declared as variables use inline QueryAnalyzer calls.
# Using 20 features ensures every query template gets a unique signature,
# eliminating false positive guards that block wins.
GUARD_FEATURES = {
    # Boolean features (local variables in EVOLVE-BLOCK)
    "subquery": "subquery",
    "corr": "corr",
    "union": "union",
    "outerJoin": "outerJoin",
    "distinct": "distinct",
    "filterAboveJoin": "filterAboveJoin",
    "selfJoinSubquery": "selfJoinSubquery",
    # Boolean features (inline QueryAnalyzer calls)
    "hasAggregate": "hasAggregate(root)",
    "hasCaseWhen": "hasCaseWhen(root)",
    "hasLikePattern": "hasLikePattern(root)",
    "hasLimit": "hasLimit(root)",
    "hasSort": "hasSort(root)",
    # Integer features (local variables)
    "joins": "joins",
    "numSubqueries": "numSubqueries",
    "aggs": "aggs",
    "groupByKeys": "groupByKeys",
    "filters": "filters",
    "predicates": "predicates",
    "depth": "depth",
    # Integer features (inline QueryAnalyzer calls)
    "countAggregateCalls": "countAggregateCalls(root)",
}

BOOL_FEATURES = {"subquery", "corr", "union", "outerJoin", "distinct",
                 "filterAboveJoin", "selfJoinSubquery", "hasAggregate",
                 "hasCaseWhen", "hasLikePattern", "hasLimit", "hasSort"}
INT_FEATURES = {"joins", "numSubqueries", "aggs", "groupByKeys", "filters",
                "predicates", "depth", "countAggregateCalls"}


def _features_match(f1: Dict, f2: Dict) -> bool:
    """Check if two feature dicts have identical values for all guard features."""
    for key in GUARD_FEATURES:
        if f1.get(key) != f2.get(key):
            return False
    return True


def _build_condition(features: Dict) -> str:
    """Build a Java boolean condition string from features."""
    parts = []
    for key in GUARD_FEATURES:
        val = features.get(key)
        if val is None:
            continue
        java_name = GUARD_FEATURES[key]
        if key in BOOL_FEATURES:
            parts.append(java_name if val else f"!{java_name}")
        elif key in INT_FEATURES:
            parts.append(f"{java_name} == {val}")
    return " && ".join(parts)


def _guard_matches_features(guard_features: Dict, query_features: Dict) -> bool:
    """Check if a guard's feature conditions would match a query's features."""
    for key in GUARD_FEATURES:
        guard_val = guard_features.get(key)
        query_val = query_features.get(key)
        if guard_val is None:
            continue
        if guard_val != query_val:
            return False
    return True


def generate_guards(
    regressions: List[Dict],
    wins: List[Dict],
    feature_matrix: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Generate deterministic early-exit guards for regressing queries.

    Args:
        regressions: List of regression dicts with 'query', 'speedup', etc.
        wins: List of win dicts with 'query', 'speedup', etc.
        feature_matrix: Features for all queries (from Java QueryAnalyzer).

    Returns:
        (guards, unguarded): guards is a list of guard dicts ready for injection,
        unguarded is a list of regression dicts that couldn't be guarded (variant problems).
    """
    if not regressions:
        return [], []

    # Build win lookup: query_id -> win dict
    win_lookup = {w["query"]: w for w in wins}
    # Build win features list for safety checking
    win_features = {w["query"]: feature_matrix.get(w["query"], {}) for w in wins}

    # Group regressions by feature signature for deduplication
    # Multiple variants (e.g., query014_0, query014_1) with identical features -> one guard
    feature_groups: Dict[str, List[Dict]] = {}
    for reg in regressions:
        qid = reg["query"]
        features = feature_matrix.get(qid)
        if not features:
            logger.warning("No features for regression %s, skipping", qid)
            continue

        # Create a hashable key from guard features
        key = tuple(
            (k, features.get(k)) for k in sorted(GUARD_FEATURES.keys())
        )
        if key not in feature_groups:
            feature_groups[key] = []
        feature_groups[key].append(reg)

    guards = []
    unguarded = []

    for feat_key, group_regs in feature_groups.items():
        features = dict(feat_key)
        query_ids = [r["query"] for r in group_regs]
        worst_speedup = min(r.get("speedup", 0) for r in group_regs)

        # Check for variant problem: any WIN has identical features?
        variant_wins = []
        for wid, wf in win_features.items():
            if wf and _features_match(features, wf):
                variant_wins.append(win_lookup[wid])

        if variant_wins:
            # Variant problem — can't guard without blocking the win
            best_win_speedup = max(w.get("speedup", 1) for w in variant_wins)
            variant_win_ids = [w["query"] for w in variant_wins]

            if best_win_speedup > (1 / worst_speedup):
                # Win outweighs regression — accept tradeoff
                logger.info(
                    "Variant tradeoff accepted: %s (%.2fx regression) vs %s (%.2fx win)",
                    query_ids, worst_speedup, variant_win_ids, best_win_speedup,
                )
            else:
                logger.warning(
                    "Variant conflict: %s regresses %.2fx, wins %s only %.2fx — "
                    "cannot guard, flagging for review",
                    query_ids, worst_speedup, variant_win_ids, best_win_speedup,
                )

            unguarded.extend(group_regs)
            continue

        # Verify guard won't block any winning query
        blocked_wins = []
        for wid, wf in win_features.items():
            if wf and _guard_matches_features(features, wf):
                blocked_wins.append(wid)

        if blocked_wins:
            # This shouldn't happen if variant detection worked, but safety check
            logger.error(
                "Guard for %s would block wins %s — skipping",
                query_ids, blocked_wins,
            )
            unguarded.extend(group_regs)
            continue

        # Build the guard
        condition = _build_condition(features)
        guard = {
            "query_ids": query_ids,
            "condition": condition,
            "features": features,
            "worst_speedup": worst_speedup,
            "reason": f"deterministic guard ({len(query_ids)} queries, {worst_speedup:.2f}x worst)",
        }
        guards.append(guard)
        logger.info(
            "Generated guard for %s (worst %.2fx): %s",
            query_ids, worst_speedup, condition,
        )

    return guards, unguarded


def guard_to_java(guard: Dict) -> str:
    """Generate Java code for a single guard.

    Args:
        guard: Guard dict from generate_guards().

    Returns:
        Java code string (indented with 8 spaces for method body).
    """
    qids = ", ".join(guard["query_ids"])
    speedup = guard["worst_speedup"]
    condition = guard["condition"]

    return (
        f"        // Guard: {qids} ({speedup:.2f}x regression) — deterministic guard\n"
        f"        if ({condition}) {{\n"
        f"            return rules.toArray(new String[0]);\n"
        f"        }}"
    )


def inject_guards(ruleselector_code: str, guards: List[Dict]) -> str:
    """Inject guard code into RuleSelector.java.

    Inserts guards after feature variable extraction, before the first pattern.
    If an existing guard section exists, APPENDS new guards to it (preserving
    existing guards). This is critical: the LLM's code may include canonical
    guards from previous iterations that protect against known regressions.
    Replacing them would lose that protection.

    Args:
        ruleselector_code: Current RuleSelector.java source code.
        guards: List of guard dicts from generate_guards().

    Returns:
        Modified RuleSelector.java source code with guards injected.
    """
    if not guards:
        return ruleselector_code

    lines = ruleselector_code.split("\n")
    in_block = False
    guard_section_start = -1
    guard_section_end = -1
    insertion_point = -1

    for i, line in enumerate(lines):
        stripped = line.strip()

        if "EVOLVE-BLOCK-START" in stripped:
            in_block = True
            continue
        if "EVOLVE-BLOCK-END" in stripped:
            break

        if not in_block:
            continue

        # Detect existing guard section header
        if "EARLY-EXIT SAFETY GUARDS" in stripped:
            guard_section_start = i
            continue

        # Find guard section end: first PATTERN comment after guard section start
        # This is the most reliable boundary — works regardless of guard comment format
        if guard_section_start >= 0 and guard_section_end < 0:
            if stripped.startswith("// PATTERN"):
                guard_section_end = i
                continue

        # Find insertion point: first pattern or if-statement after variable declarations
        # (only used when there's no existing guard section)
        if insertion_point < 0 and guard_section_start < 0:
            if stripped.startswith("// PATTERN") or (
                stripped.startswith("if (")
                and not stripped.startswith("if (!jp")  # skip JVM checks
            ):
                insertion_point = i
                continue

    # Build new guard entries (without section header — will append to existing)
    new_guard_lines = [
        "",
        "        // --- New guards (iteration) ---",
    ]
    for guard in guards:
        new_guard_lines.append(guard_to_java(guard))
        new_guard_lines.append("")
    new_guard_text = "\n".join(new_guard_lines)

    # Build full guard section (with header — for when no existing section)
    full_guard_lines = [
        "",
        "        // ============ EARLY-EXIT SAFETY GUARDS ============",
        "        // Auto-generated: block known-regressing feature signatures.",
        "        // These fire before any pattern, so they work regardless of which",
        "        // pattern would have caught the query.",
        "",
    ]
    for guard in guards:
        full_guard_lines.append(guard_to_java(guard))
        full_guard_lines.append("")
    full_guard_text = "\n".join(full_guard_lines)

    if guard_section_start >= 0 and guard_section_end >= 0:
        # APPEND to existing guard section (preserve existing guards)
        before = "\n".join(lines[:guard_section_end])
        after = "\n".join(lines[guard_section_end:])
        return before + new_guard_text + "\n" + after
    elif guard_section_start >= 0:
        # Found start but no end — find end by looking for first PATTERN comment
        for i in range(guard_section_start + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("// PATTERN"):
                guard_section_end = i
                break
        if guard_section_end >= 0:
            before = "\n".join(lines[:guard_section_end])
            after = "\n".join(lines[guard_section_end:])
            return before + new_guard_text + "\n" + after
    elif insertion_point >= 0:
        # No existing guard section — insert with full header before first pattern
        before = "\n".join(lines[:insertion_point])
        after = "\n".join(lines[insertion_point:])
        return before + full_guard_text + "\n" + after

    # Fallback: couldn't find insertion point, return unchanged
    logger.error("Could not find insertion point for guards in RuleSelector.java")
    return ruleselector_code
