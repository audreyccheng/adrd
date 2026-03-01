"""
Pattern analyst: uses Claude API to discover feature patterns from search results.

Phase 2 of the evolution loop.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig

logger = logging.getLogger(__name__)

# Accumulated API token usage for this module
_api_usage = {"input_tokens": 0, "output_tokens": 0}

_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    """Load the analyst system prompt from the prompts directory."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        prompt_path = Path(__file__).parent / "prompts" / "analyst_system.md"
        _SYSTEM_PROMPT = prompt_path.read_text()
    return _SYSTEM_PROMPT


def _build_user_message(
    search_results: Dict,
    feature_matrix: Dict[str, Dict],
    existing_code: str,
    config: EvolutionConfig,
    search_history: Optional[Any] = None,
    iteration: int = 1,
) -> str:
    """Build the user message for the analyst Claude API call."""

    # Summarize search results
    wins_summary = []
    for qr in search_results.get("queries", []):
        qid = qr["query_id"]
        orig = qr.get("original_latency")
        if not qr.get("wins"):
            continue
        for w in qr["wins"]:
            combo_str = "+".join(w["combo"])
            wins_summary.append(
                f"  {qid}: {orig:.3f}s -> {w['latency']:.3f}s "
                f"({combo_str}, {w['speedup']}x)"
            )

    regressions_summary = []
    for qr in search_results.get("queries", []):
        qid = qr["query_id"]
        orig = qr.get("original_latency")
        if not qr.get("regressions"):
            continue
        for r in qr["regressions"]:
            combo_str = "+".join(r["combo"])
            regressions_summary.append(
                f"  {qid}: {orig:.3f}s -> {r['latency']:.3f}s "
                f"({combo_str}, {r['speedup']}x)"
            )

    # Format feature matrix
    features_text = []
    for qid, features in sorted(feature_matrix.items()):
        feat_str = ", ".join(f"{k}={v}" for k, v in sorted(features.items()))
        features_text.append(f"  {qid}: {feat_str}")

    # Count existing patterns (active ones)
    active_patterns = existing_code.count("return rules.toArray")

    msg = f"""## Search Results

### Wins (speedup > {config.win_threshold}x):
{chr(10).join(wins_summary) if wins_summary else "  (none)"}

### Regressions (speedup < {config.regression_threshold}x):
{chr(10).join(regressions_summary) if regressions_summary else "  (none)"}

## Feature Matrix
{chr(10).join(features_text)}

## Existing Patterns
RuleSelector.java currently has {active_patterns} active patterns (return statements).
The following queries are already covered by existing patterns — do NOT propose patterns for them unless you can improve the existing one:

{existing_code}

## Known Dangerous Combos
- FSQ on: {', '.join(config.fsq_dangerous)} (causes severe regression)
- FIJ on: {', '.join(config.fij_dangerous)} (causes timeout/regression)

## Instructions
1. Analyze the wins to find feature patterns that predict which combos help.
2. Group wins by common features and propose pattern hypotheses.
3. Check ALL variants of each query (e.g., _0 and _1) before proposing.
4. Return your hypotheses as a JSON array.
5. Only propose patterns with confidence "medium" or "high".
6. Ensure conditions are tight enough to avoid matching non-winning queries.
"""

    # Add search history section for adaptive iterations
    if search_history and iteration > 1 and search_history.tested:
        history_section = f"""
## Search History (for directing next iteration)

### Coverage
- Queries tested so far: {len(search_history.tested)}
- Queries with wins: {len(search_history.winning_queries)}
- Queries with no wins: {len(search_history.neutral_queries)}
- Queries untested: {len(search_history.untested_queries)}
- Queries with regressing combos: {len(search_history.regressing_combos)}

### Best Results Per Query (top 20 by speedup)
{_format_best_combos(search_history.best_combos)}

### Untested Queries
{', '.join(sorted(search_history.untested_queries)[:30]) if search_history.untested_queries else '(none)'}
{f'  ... and {len(search_history.untested_queries) - 30} more' if len(search_history.untested_queries) > 30 else ''}

### Empirically Discovered Dangerous Combos
{_format_dangers(search_history.regressing_combos)}

## ADDITIONAL OUTPUT: Search Directives

In addition to your pattern hypotheses JSON array, also output a SECOND JSON block
labeled with the marker `SEARCH_DIRECTIVES`. This tells the searcher what to test in
the next iteration. See your system prompt for the available strategies and format.
"""
        msg += history_section

    return msg


def _parse_hypotheses(response_text: str) -> List[Dict]:
    """Parse Claude's response into a list of hypothesis dicts."""
    # Try to find JSON array in the response
    text = response_text.strip()

    # Look for ```json ... ``` block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try to find a JSON array
    if "[" in text:
        arr_start = text.index("[")
        # Find matching closing bracket
        depth = 0
        for i in range(arr_start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    text = text[arr_start : i + 1]
                    break

    try:
        hypotheses = json.loads(text)
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
        return hypotheses
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse analyst response as JSON: %s", e)
        logger.debug("Raw response: %s", response_text[:500])
        return []


def _validate_hypothesis(hyp: Dict) -> bool:
    """Validate that a hypothesis has all required fields."""
    required = ["name", "conditions", "rules", "evidence", "confidence"]
    for field in required:
        if field not in hyp:
            logger.warning("Hypothesis missing field '%s': %s", field, hyp.get("name", "?"))
            return False

    if not isinstance(hyp["rules"], list) or not hyp["rules"]:
        logger.warning("Hypothesis '%s' has empty or invalid rules", hyp["name"])
        return False

    if not isinstance(hyp["conditions"], dict) or not hyp["conditions"]:
        logger.warning("Hypothesis '%s' has empty or invalid conditions", hyp["name"])
        return False

    if hyp.get("variant_risk") == "high":
        logger.warning("Skipping hypothesis '%s': variant_risk=high", hyp["name"])
        return False

    return True


def _format_best_combos(best_combos: Dict[str, Dict]) -> str:
    """Format top best combos for the analyst prompt."""
    sorted_best = sorted(
        best_combos.items(),
        key=lambda x: x[1].get("speedup", 0),
        reverse=True,
    )
    lines = []
    for qid, info in sorted_best[:20]:
        combo_str = "+".join(info.get("combo", []))
        lines.append(f"  {qid}: {combo_str} -> {info.get('speedup', 0):.2f}x")
    return "\n".join(lines) if lines else "  (none yet)"


def _format_dangers(regressing_combos: Dict[str, List[str]]) -> str:
    """Format discovered dangerous combos for the analyst prompt."""
    lines = []
    for qid, combo_keys in sorted(regressing_combos.items()):
        lines.append(f"  {qid}: {', '.join(combo_keys[:3])}")
    return "\n".join(lines[:15]) if lines else "  (none)"


def _parse_directives(response_text: str) -> List[Dict]:
    """Extract SEARCH_DIRECTIVES JSON block from analyst response.

    The analyst outputs directives after its hypotheses, marked with
    the text 'SEARCH_DIRECTIVES' followed by a JSON array.

    Returns:
        List of directive dicts, or [] if not found/parse error.
    """
    marker = "SEARCH_DIRECTIVES"
    if marker not in response_text:
        return []

    idx = response_text.index(marker)
    remaining = response_text[idx:]

    # Find JSON block (```json ... ``` or bare [...])
    json_text = None
    if "```json" in remaining:
        try:
            start = remaining.index("```json") + 7
            end = remaining.index("```", start)
            json_text = remaining[start:end].strip()
        except ValueError:
            pass
    elif "```" in remaining:
        try:
            start = remaining.index("```") + 3
            end = remaining.index("```", start)
            json_text = remaining[start:end].strip()
        except ValueError:
            pass

    if json_text is None and "[" in remaining:
        arr_start = remaining.index("[")
        depth = 0
        for i in range(arr_start, len(remaining)):
            if remaining[i] == "[":
                depth += 1
            elif remaining[i] == "]":
                depth -= 1
                if depth == 0:
                    json_text = remaining[arr_start : i + 1]
                    break

    if json_text is None:
        logger.warning("Found SEARCH_DIRECTIVES marker but no JSON block")
        return []

    try:
        directives = json.loads(json_text)
        if not isinstance(directives, list):
            directives = [directives]
        # Basic validation
        valid = []
        for d in directives:
            if isinstance(d, dict) and "strategy" in d:
                valid.append(d)
            else:
                logger.warning("Skipping invalid directive (missing 'strategy'): %s", d)
        return valid
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse search directives JSON: %s", e)
        return []


def analyze_results(
    search_results: Dict,
    feature_matrix: Dict[str, Dict],
    existing_code: str,
    config: EvolutionConfig,
    search_history: Optional[Any] = None,
    iteration: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    """Analyze search results and propose pattern hypotheses using Claude API.

    Args:
        search_results: Output from searcher.search_all()
        feature_matrix: Dict mapping query_id to feature dict
        existing_code: Current RuleSelector.java source code
        config: Evolution config
        search_history: Accumulated search history (for adaptive search)
        iteration: Current iteration number

    Returns:
        Tuple of (validated hypotheses, search directives for next iteration)
    """
    import anthropic

    if not config.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in config or environment")

    system_prompt = _load_system_prompt()
    user_message = _build_user_message(
        search_results, feature_matrix, existing_code, config,
        search_history=search_history,
        iteration=iteration,
    )

    logger.info("Calling Claude API for analysis (model=%s)...", config.model)

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
        "Analyst response: %d chars, %d input tokens, %d output tokens",
        len(response_text),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    hypotheses = _parse_hypotheses(response_text)
    logger.info("Parsed %d hypotheses from response", len(hypotheses))

    # Validate and filter
    valid = [h for h in hypotheses if _validate_hypothesis(h)]
    logger.info("%d hypotheses passed validation (of %d total)", len(valid), len(hypotheses))

    # Parse search directives (for adaptive search in next iteration)
    directives = _parse_directives(response_text)
    logger.info("Parsed %d search directives for next iteration", len(directives))

    return valid, directives
