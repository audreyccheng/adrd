"""
Mutation strategies for SimulatorConfigs.

Each mutation produces a valid SimulatorConfig variant by modifying
one or more aspects of the simulator's design. Mutations are the
outer loop's exploration mechanism.
"""

import json
import os
import random
from dataclasses import asdict
from typing import List, Optional

try:
    from .simulator_config import SimulatorConfig
except ImportError:
    from simulator_config import SimulatorConfig


# Boolean fields that can be toggled
BOOLEAN_FIELDS = [
    "expose_usage_count",
    "expose_access_count",
    "expose_last_access_time",
    "expose_dirty_time",
    "expose_is_dirty",
    "expose_block_group",
    "expose_scan_ids",
    "enable_estimator",
    "enable_confidence",
    "enable_block_group_estimate",
    "policy_sees_scan_context",
    "simulate_hint_bit_dirtying",
    "simulate_ring_buffers",
    "artifact_suggestions",
    "per_workload_breakdown",
    "wal_cost_enabled",
]

# Numeric fields with valid ranges
NUMERIC_FIELDS = {
    "block_group_size": (16, 512, [16, 32, 64, 128, 256, 512]),
    "bgwriter_coverage": (0.0, 0.60, None),
    "max_usage_count": (1, 10, [1, 2, 3, 5, 8, 10]),
    "dirty_penalty_weight": (0.0, 0.30, None),
    "reeviction_penalty_weight": (0.0, 0.30, None),
}

# Scoring modes
SCORING_MODES = ["hit_rate", "latency", "combined"]

# Available workloads and reasonable weight ranges
AVAILABLE_WORKLOADS = [
    "tpch_full", "tpch_fast", "tpch_hard",
    "tpcc_10w", "tpcc_10w_stress", "tpcc_100w_calibrated",
    "ycsb_a", "ycsb_b",
    "chbenchmark_3to1",
]


def mutate_toggle(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """Toggle a random boolean feature."""
    rng = rng or random.Random()
    field_name = rng.choice(BOOLEAN_FIELDS)
    current_val = getattr(config, field_name)
    new_config = config.clone(**{field_name: not current_val})

    # Direction matters for dependency enforcement:
    # - Toggling OFF: backward deps (disable dependents of the disabled feature)
    # - Toggling ON: forward deps (enable prerequisites of the enabled feature)
    if current_val:
        # Toggled OFF: disable dependents
        new_config = _enforce_backward_deps(new_config)
    else:
        # Toggled ON: enable prerequisites
        new_config = _enforce_forward_deps(new_config)

    return new_config


def _enforce_forward_deps(config: SimulatorConfig) -> SimulatorConfig:
    """Enable prerequisites for enabled features."""
    changes = {}
    if config.enable_estimator and not config.expose_block_group:
        changes["expose_block_group"] = True
    if config.enable_confidence and not config.enable_estimator:
        changes["enable_estimator"] = True
        changes["expose_block_group"] = True
    if config.enable_block_group_estimate and not config.enable_estimator:
        changes["enable_estimator"] = True
        changes["expose_block_group"] = True
    if config.expose_scan_ids and not config.expose_block_group:
        changes["expose_block_group"] = True
    return config.clone(**changes) if changes else config


def _enforce_backward_deps(config: SimulatorConfig) -> SimulatorConfig:
    """Disable dependents of disabled features."""
    changes = {}
    if not config.expose_block_group:
        changes.update(expose_scan_ids=False, enable_estimator=False,
                       enable_confidence=False, enable_block_group_estimate=False)
    if not config.enable_estimator or not changes.get("enable_estimator", config.enable_estimator):
        changes.update(enable_confidence=False, enable_block_group_estimate=False)
    return config.clone(**changes) if changes else config

    return new_config


def mutate_numeric(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """Adjust a random numeric parameter."""
    rng = rng or random.Random()
    field_name = rng.choice(list(NUMERIC_FIELDS.keys()))
    lo, hi, discrete_values = NUMERIC_FIELDS[field_name]

    if discrete_values:
        new_val = rng.choice(discrete_values)
    else:
        current_val = getattr(config, field_name)
        # Gaussian perturbation within bounds
        delta = (hi - lo) * 0.2 * rng.gauss(0, 1)
        new_val = max(lo, min(hi, current_val + delta))
        new_val = round(new_val, 4)

    return config.clone(**{field_name: new_val})


def mutate_scoring(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """Switch scoring mode."""
    rng = rng or random.Random()
    current = config.scoring_mode
    choices = [m for m in SCORING_MODES if m != current]
    return config.clone(scoring_mode=rng.choice(choices))


def mutate_workload_weights(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """Modify workload mix weights."""
    rng = rng or random.Random()
    weights = dict(config.workload_weights)

    action = rng.choice(["shift", "add", "remove", "rebalance"])

    if action == "shift" and weights:
        # Shift weight between two workloads
        wl = rng.choice(list(weights.keys()))
        delta = rng.uniform(-0.15, 0.15)
        weights[wl] = max(0.05, min(0.9, weights[wl] + delta))

    elif action == "add":
        # Add a new workload
        candidates = [w for w in AVAILABLE_WORKLOADS if w not in weights]
        if candidates:
            new_wl = rng.choice(candidates)
            weights[new_wl] = rng.uniform(0.1, 0.3)

    elif action == "remove" and len(weights) > 1:
        # Remove a workload (keep at least one)
        wl = rng.choice(list(weights.keys()))
        del weights[wl]

    elif action == "rebalance":
        # Normalize to sum to 1.0
        pass  # normalization happens below

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}

    return config.clone(workload_weights=weights)


def mutate_composite(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """Apply 2-3 mutations simultaneously."""
    rng = rng or random.Random()
    mutations = [mutate_toggle, mutate_numeric, mutate_scoring, mutate_workload_weights]
    n = rng.randint(2, 3)
    selected = rng.sample(mutations, min(n, len(mutations)))

    result = config
    for mut_fn in selected:
        result = mut_fn(result, rng)
    return result


# All mutation strategies with weights
MUTATION_STRATEGIES = {
    "toggle": (mutate_toggle, 0.30),
    "numeric": (mutate_numeric, 0.20),
    "scoring": (mutate_scoring, 0.10),
    "workload_weights": (mutate_workload_weights, 0.20),
    "composite": (mutate_composite, 0.20),
}


def mutate(config: SimulatorConfig, rng: Optional[random.Random] = None) -> SimulatorConfig:
    """
    Apply a random mutation to a SimulatorConfig.
    Mutation type selected by weighted probability.

    Args:
        config: Parent config to mutate
        rng: Optional random.Random instance for reproducibility

    Returns:
        New SimulatorConfig (parent is not modified)
    """
    rng = rng or random.Random()

    # Weighted selection
    strategies = list(MUTATION_STRATEGIES.values())
    fns, weights = zip(*strategies)
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for fn, w in zip(fns, weights):
        cumulative += w
        if r <= cumulative:
            new_config = fn(config, rng)
            # Generate a descriptive name
            diff = config.diff(new_config)
            diff_keys = [k for k in diff if k != "name"]
            if diff_keys:
                new_config = new_config.clone(name=f"{config.name}_mut_{'_'.join(diff_keys[:2])}")
            return new_config

    # Fallback (shouldn't reach here)
    return mutate_toggle(config, rng)


def generate_ablation_variants(base: SimulatorConfig) -> List[SimulatorConfig]:
    """
    Generate single-feature ablation variants from a base config.
    Each variant disables exactly one feature. Used for the ablation study.
    """
    variants = []
    for field_name in BOOLEAN_FIELDS:
        current_val = getattr(base, field_name)
        if current_val:  # Only ablate features that are currently enabled
            variant = base.clone(**{field_name: False})

            # Enforce dependencies
            if field_name == "expose_block_group":
                variant = variant.clone(
                    expose_scan_ids=False,
                    enable_estimator=False,
                    enable_confidence=False,
                    enable_block_group_estimate=False,
                    name=f"{base.name}_no_scan_tracking",
                )
            elif field_name == "enable_estimator":
                variant = variant.clone(
                    enable_confidence=False,
                    enable_block_group_estimate=False,
                    name=f"{base.name}_no_estimator",
                )
            else:
                variant = variant.clone(name=f"{base.name}_no_{field_name.replace('expose_', '').replace('enable_', '')}")

            variants.append(variant)

    return variants


# ===========================================================================
# LLM-guided mutation
# ===========================================================================

# Description of each config field for the LLM prompt
FIELD_DESCRIPTIONS = {
    "expose_usage_count": "PostgreSQL usage_count (clock-sweep counter, 0-5). Policies can use this for LRU-like eviction.",
    "expose_access_count": "Total access count per buffer. Enables frequency-based eviction (LFU-style).",
    "expose_last_access_time": "Timestamp of last access. Enables recency-based eviction (LRU-style).",
    "expose_dirty_time": "When the page became dirty. Enables WAL-flush-cost-aware eviction.",
    "expose_is_dirty": "Whether the page has been modified. Enables clean-over-dirty preference.",
    "expose_block_group": "PBM block group (128 blocks = 1MiB). PREREQUISITE for scan tracking and estimator. Enables scan-aware eviction.",
    "expose_scan_ids": "Which active scans will access this block group. Requires expose_block_group. Enables multi-scan protection.",
    "enable_estimator": "NextAccessEstimator: predicts when each buffer will next be accessed based on scan tracking. Requires expose_block_group. THIS IS THE KEY V5 INNOVATION — enables Belady-like optimal eviction.",
    "enable_confidence": "Adds confidence scores to estimator predictions. Requires enable_estimator.",
    "enable_block_group_estimate": "Block-group-level estimation (coarser but faster). Requires enable_estimator.",
    "policy_sees_scan_context": "Pass current scan info (relation, position, type) to the policy at eviction time.",
    "simulate_hint_bit_dirtying": "Simulate PostgreSQL's hint-bit writes (85% of reads dirty the page). Affects dirty-page statistics.",
    "simulate_ring_buffers": "Simulate PostgreSQL's ring buffer strategy for bulk reads. Limits buffer usage for large scans.",
    "wal_cost_enabled": "Track WAL flush costs for recently-dirtied pages in scoring.",
    "scoring_mode": "How policies are scored: 'hit_rate' (buffer cache hits), 'latency' (I/O latency including dirty write cost), 'combined' (both).",
    "block_group_size": "Blocks per group (default 128 = 1MiB). Smaller = finer tracking but more overhead.",
    "bgwriter_coverage": "Background writer coverage (0.0-0.6). Higher = fewer synchronous dirty writes.",
    "workload_weights": "Weighted mix of workloads for scoring. Keys: tpch_full (analytical scans), tpcc_10w (OLTP point lookups), ycsb_a (Zipfian hot/cold).",
    "dirty_penalty_weight": "Penalty weight for synchronous dirty evictions in scoring (0.0-0.3).",
    "artifact_suggestions": "Include targeted improvement suggestions in LLM feedback.",
    "per_workload_breakdown": "Include per-workload score breakdown in LLM feedback.",
}

_LLM_MUTATION_PROMPT = """\
You are optimizing the configuration of a PostgreSQL buffer management simulator.
The simulator is used to evolve buffer eviction policies via LLM-guided evolution (inner loop).
Your job is to improve the SIMULATOR CONFIGURATION so that the inner loop can discover better policies.

## Current best configuration and its score
{current_config}
Score: {current_score:.4f} (higher is better, theoretical max ~0.82)

## Score breakdown by workload
{workload_breakdown}

## History of configurations tried and their scores
{history}

## Available configuration fields
{field_descriptions}

## Dependencies
- enable_estimator requires expose_block_group
- enable_confidence requires enable_estimator
- enable_block_group_estimate requires enable_estimator
- expose_scan_ids requires expose_block_group

## Your task
Based on the scores and history, propose changes to the configuration that will help the inner loop discover BETTER eviction policies. Think about:
1. What information would help the policy make better eviction decisions?
2. Are there workloads where the current config underperforms? Why?
3. Would changing the scoring mode reveal policy quality differences that hit_rate misses?
4. Are there features currently disabled that could unlock new policy strategies?

Respond with a JSON object containing ONLY the fields you want to change. For example:
```json
{{"enable_estimator": true, "expose_block_group": true, "scoring_mode": "latency"}}
```

Keep changes focused — change 1-4 fields. Explain your reasoning in 1-2 sentences before the JSON."""


def mutate_llm(
    config: SimulatorConfig,
    generation_results: Optional[List] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> SimulatorConfig:
    """
    LLM-guided mutation: uses an LLM to reason about what config changes
    would help the inner loop discover better policies.

    Args:
        config: Current best config to mutate
        generation_results: List of (config_name, score, workload_breakdown) tuples
            from recent generations, for history context
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: Model to use (default gpt-4o-mini — cheap and fast enough for config proposals)

    Returns:
        New SimulatorConfig with LLM-proposed changes
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Build the prompt
    config_dict = asdict(config)
    # Show only the interesting fields (not inner loop params)
    display_config = {k: v for k, v in config_dict.items()
                      if k not in ("name", "inner_iterations", "inner_population_size",
                                   "inner_num_islands", "latency_model")}

    current_score = 0.0
    workload_breakdown = "No workload breakdown available."
    history_lines = []

    if generation_results:
        # Extract current best score
        best = max(generation_results, key=lambda r: r[1])
        current_score = best[1]

        # Workload breakdown from best
        if len(best) > 2 and best[2]:
            workload_breakdown = best[2]

        # Build history
        for name, score, breakdown in generation_results:
            history_lines.append("  %s: %.4f" % (name, score))

    history = "\n".join(history_lines) if history_lines else "No previous configurations tried."

    field_desc_lines = []
    for field, desc in FIELD_DESCRIPTIONS.items():
        current_val = config_dict.get(field, "N/A")
        field_desc_lines.append("- %s (current: %s): %s" % (field, current_val, desc))

    prompt = _LLM_MUTATION_PROMPT.format(
        current_config=json.dumps(display_config, indent=2),
        current_score=current_score,
        workload_breakdown=workload_breakdown,
        history=history,
        field_descriptions="\n".join(field_desc_lines),
    )

    # Call the LLM
    proposed_changes = _call_llm_for_mutation(prompt, api_key, model)

    if not proposed_changes:
        # Fallback to random mutation
        return mutate(config)

    # Apply proposed changes
    return _apply_llm_changes(config, proposed_changes)


def _call_llm_for_mutation(prompt: str, api_key: Optional[str], model: str) -> Optional[dict]:
    """Call LLM API and parse the JSON response."""
    if not api_key:
        print("WARNING: No API key for LLM mutation. Falling back to random mutation.")
        return None

    try:
        # Try OpenAI-compatible API
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in database buffer management and simulator design. Respond concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        text = response.choices[0].message.content

        # Extract JSON from response (may have reasoning text before it)
        return _extract_json(text)

    except ImportError:
        # Try Anthropic
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return _extract_json(text)
        except ImportError:
            print("WARNING: Neither openai nor anthropic package installed. Falling back to random.")
            return None
        except Exception as e:
            print("WARNING: Anthropic API call failed: %s. Falling back to random." % e)
            return None

    except Exception as e:
        print("WARNING: OpenAI API call failed: %s. Falling back to random." % e)
        return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM response text."""
    # Try to find JSON in code fences
    import re
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    pass
                break

    print("WARNING: Could not parse JSON from LLM response: %s" % text[:200])
    return None


def _apply_llm_changes(config: SimulatorConfig, changes: dict) -> SimulatorConfig:
    """Apply LLM-proposed changes to a config, with validation."""
    valid_fields = set(asdict(config).keys())
    validated = {}

    for field, value in changes.items():
        if field not in valid_fields:
            print("WARNING: LLM proposed unknown field '%s', skipping" % field)
            continue

        # Type validation
        current_val = getattr(config, field)
        if isinstance(current_val, bool) and isinstance(value, bool):
            validated[field] = value
        elif isinstance(current_val, (int, float)) and isinstance(value, (int, float)):
            validated[field] = type(current_val)(value)
        elif isinstance(current_val, str) and isinstance(value, str):
            validated[field] = value
        elif isinstance(current_val, dict) and isinstance(value, dict):
            validated[field] = value
        else:
            print("WARNING: Type mismatch for '%s': expected %s, got %s" %
                  (field, type(current_val).__name__, type(value).__name__))

    if not validated:
        return mutate(config)  # Fallback

    new_config = config.clone(**validated)

    # Dependency enforcement: check what the LLM actually changed.
    # For fields turned OFF: backward deps (disable dependents).
    # For fields turned ON: forward deps (enable prerequisites).
    # Process disables first, then enables, so enables can override.
    turned_off = {k for k, v in validated.items() if isinstance(v, bool) and not v and getattr(config, k)}
    turned_on = {k for k, v in validated.items() if isinstance(v, bool) and v and not getattr(config, k)}

    if turned_off:
        new_config = _enforce_backward_deps(new_config)
    if turned_on:
        new_config = _enforce_forward_deps(new_config)

    # Normalize workload weights if changed
    if "workload_weights" in validated:
        weights = new_config.workload_weights
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
            new_config = new_config.clone(workload_weights=weights)

    # Name it
    changed_fields = list(validated.keys())[:3]
    new_config = new_config.clone(name="%s_llm_%s" % (config.name[:15], "_".join(changed_fields)))

    return new_config
