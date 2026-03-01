"""
Regression fixer: uses Claude API to analyze and fix regressions in RuleSelector.

Phase 5 of the evolution loop.
Stage 1: Deterministic guard generation (no API cost).
Stage 2: LLM-based fixing for remaining regressions (existing behavior).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig

logger = logging.getLogger(__name__)

# Accumulated API token usage for this module
_api_usage = {"input_tokens": 0, "output_tokens": 0}

_SYSTEM_PROMPT = None


def fix_with_guards(
    regressions: List[Dict],
    wins: List[Dict],
    feature_matrix: Dict[str, Dict],
    ruleselector_code: str,
) -> Tuple[str, List[Dict]]:
    """Fix regressions using deterministic feature-based guards (Stage 1).

    Generates early-exit guards from verified feature values to block
    known-regressing queries. No LLM call needed.

    Args:
        regressions: List of regression dicts from validation report.
        wins: List of win dicts from validation report.
        feature_matrix: Features for all queries (from Java QueryAnalyzer).
        ruleselector_code: Current RuleSelector.java source code.

    Returns:
        (modified_code, unguarded_regressions): modified code with guards injected,
        and list of regressions that couldn't be guarded (variant problems).
    """
    from evolve_loop.guard_generator import generate_guards, inject_guards

    if not regressions:
        return ruleselector_code, []

    guards, unguarded = generate_guards(regressions, wins, feature_matrix)

    if not guards:
        logger.info("No guards generated (all regressions are variant problems)")
        return ruleselector_code, unguarded

    modified_code = inject_guards(ruleselector_code, guards)

    if modified_code == ruleselector_code:
        logger.warning("Guard injection failed — code unchanged")
        return ruleselector_code, list(regressions)

    logger.info(
        "Injected %d guards (%d regressions guarded, %d unguarded)",
        len(guards),
        len(regressions) - len(unguarded),
        len(unguarded),
    )
    return modified_code, unguarded


def _load_system_prompt() -> str:
    """Load the fixer system prompt from the prompts directory."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        prompt_path = Path(__file__).parent / "prompts" / "fixer_system.md"
        _SYSTEM_PROMPT = prompt_path.read_text()
    return _SYSTEM_PROMPT


def _build_user_message(
    regressions: List[Dict],
    ruleselector_code: str,
    hypotheses: List[Dict],
    feature_matrix: Dict[str, Dict],
    wins: Optional[List[Dict]] = None,
) -> str:
    """Build the user message for the fixer Claude API call."""

    # Build wins section (sorted by impact, highest speedup first)
    win_text = []
    if wins:
        sorted_wins = sorted(wins, key=lambda w: w.get("speedup", 0), reverse=True)
        for w in sorted_wins:
            qid = w["query"]
            baseline = w.get("baseline", "?")
            new_lat = w.get("new", "?")
            speedup = w.get("speedup", 0)
            impact = "CRITICAL" if speedup >= 10 else "HIGH" if speedup >= 2 else "moderate"
            win_text.append(f"  {qid}: {baseline}s -> {new_lat}s ({speedup}x) [{impact}]")

    reg_text = []
    for r in regressions:
        qid = r["query"]
        baseline = r.get("baseline", "?")
        new_lat = r.get("new", "TIMEOUT")
        speedup = r.get("speedup", 0)
        status = r.get("status", "")
        reg_text.append(f"  {qid}: {baseline}s -> {new_lat}s ({speedup}x) {status}")

    # Get features for regressing queries
    feature_text = []
    for r in regressions:
        qid = r["query"]
        features = feature_matrix.get(qid, {})
        if features:
            feat_str = ", ".join(f"{k}={v}" for k, v in sorted(features.items()))
            feature_text.append(f"  {qid}: {feat_str}")

    # Also include features for winning queries
    win_feature_text = []
    if wins:
        for w in wins:
            qid = w["query"]
            features = feature_matrix.get(qid, {})
            if features:
                feat_str = ", ".join(f"{k}={v}" for k, v in sorted(features.items()))
                win_feature_text.append(f"  {qid}: {feat_str}")

    hyp_json = json.dumps(hypotheses, indent=2) if hypotheses else "[]"

    return f"""## Current Wins (MUST PRESERVE)

{chr(10).join(win_text) if win_text else "  (no wins reported)"}

## Regressions Found

{chr(10).join(reg_text)}

## Features of Winning Queries

{chr(10).join(win_feature_text) if win_feature_text else "  (features not available)"}

## Features of Regressing Queries

{chr(10).join(feature_text) if feature_text else "  (features not available)"}

## Pattern Hypotheses That Were Implemented

```json
{hyp_json}
```

## Current RuleSelector.java

```java
{ruleselector_code}
```

## Instructions

1. Review the wins list above. These are VALUABLE — patterns producing them must be preserved.
2. For each regression, identify which pattern in RuleSelector.java is causing it.
3. Check if the regressing pattern also produces wins. If it does:
   - NEVER disable it. Instead, tighten conditions to exclude the regressing query while keeping winning queries.
   - Use the feature differences between winning and regressing queries to find distinguishing conditions.
4. Only disable a pattern if it produces ZERO wins.
5. Return the COMPLETE modified RuleSelector.java with fixes applied.
6. Add comments explaining each fix and confirming which wins are preserved.
"""


def fix_regressions(
    regressions: List[Dict],
    ruleselector_code: str,
    hypotheses: List[Dict],
    feature_matrix: Dict[str, Dict],
    config: EvolutionConfig,
    wins: Optional[List[Dict]] = None,
) -> Optional[str]:
    """Fix regressions in RuleSelector.java using Claude API.

    Args:
        regressions: List of regression dicts from validation report
        ruleselector_code: Current RuleSelector.java source code
        hypotheses: Pattern hypotheses that were implemented
        feature_matrix: Features for all queries
        config: Evolution config
        wins: List of win dicts from validation report (to avoid losing them)

    Returns:
        Fixed RuleSelector.java source code, or None on failure
    """
    import anthropic

    if not regressions:
        logger.info("No regressions to fix")
        return ruleselector_code

    if not config.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in config or environment")

    system_prompt = _load_system_prompt()
    user_message = _build_user_message(
        regressions, ruleselector_code, hypotheses, feature_matrix, wins
    )

    logger.info(
        "Calling Claude API for regression fixing (%d regressions, model=%s)...",
        len(regressions), config.model,
    )

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        response = stream.get_final_message()

    response_text = response.content[0].text
    _api_usage["input_tokens"] += response.usage.input_tokens
    _api_usage["output_tokens"] += response.usage.output_tokens
    logger.info(
        "Fixer response: %d chars, %d input/%d output tokens",
        len(response_text),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    # Extract Java code
    java_code = _extract_java_code(response_text)
    if java_code is None:
        logger.error("Could not extract Java code from fixer response")
        return None

    if "EVOLVE-BLOCK-START" not in java_code:
        logger.error("Fixed code missing EVOLVE-BLOCK-START marker")
        return None

    # Compile check — catch corrupted/malformed code before validation
    try:
        from evolve_loop.implementer import _try_compile
        success, error = _try_compile(java_code, config)
        if not success:
            logger.error("Fixed code does not compile: %s", error[:200])
            return None
    except Exception as e:
        logger.warning("Compile check skipped: %s", e)

    return java_code


def disable_unsafe(
    ruleselector_code: str,
    regressions: List[Dict],
    feature_matrix: Dict[str, Dict],
) -> str:
    """Emergency fallback: disable ALL newly-added patterns in the EVOLVE-BLOCK.

    This is a blunt instrument — it comments out every pattern between the
    EVOLVE-BLOCK markers, not just the ones causing regressions. Used when
    the Claude API fixer fails or as a last resort.

    Note: regressions and feature_matrix are accepted for interface
    compatibility but not used for selective disabling.

    Args:
        ruleselector_code: Current RuleSelector.java source
        regressions: List of regression dicts (used only for logging)
        feature_matrix: Features for queries (unused)

    Returns:
        Modified RuleSelector.java with all evolved patterns disabled
    """
    if not regressions:
        return ruleselector_code

    reg_queries = {r["query"] for r in regressions}
    logger.warning(
        "Emergency disable: commenting out patterns for %d regressing queries: %s",
        len(reg_queries), reg_queries,
    )

    # Find patterns between EVOLVE-BLOCK markers and comment them out
    lines = ruleselector_code.split("\n")
    in_block = False
    in_pattern = False
    pattern_start = -1
    modified_lines = list(lines)
    disabled_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if "EVOLVE-BLOCK-START" in stripped:
            in_block = True
            continue
        if "EVOLVE-BLOCK-END" in stripped:
            in_block = False
            continue

        if not in_block:
            continue

        # Detect pattern start (if statement)
        if stripped.startswith("if (") and not stripped.startswith("//"):
            pattern_start = i
            in_pattern = True

        # Detect pattern end (return statement)
        if in_pattern and "return rules.toArray" in stripped:
            pattern_end = i
            # Find closing brace
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip() == "}":
                    pattern_end = j
                    break

            # Comment out all lines from pattern_start to pattern_end
            for k in range(pattern_start, pattern_end + 1):
                if not modified_lines[k].strip().startswith("//"):
                    modified_lines[k] = "        // DISABLED: " + modified_lines[k].strip()
            disabled_count += 1
            in_pattern = False

    if disabled_count > 0:
        logger.warning("Emergency disable: commented out %d patterns", disabled_count)
    else:
        logger.warning(
            "Emergency disable found no active patterns to disable."
        )
    return "\n".join(modified_lines)


def _extract_java_code(response_text: str) -> Optional[str]:
    """Extract Java source code from Claude's response."""
    from evolve_loop.utils.code_extraction import extract_java_code
    return extract_java_code(response_text)
