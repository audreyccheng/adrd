"""
SimulatorConfig — Declarative configuration for a simulator variant.

Defines the search space for the outer loop of two-level simulator evolution.
Each config specifies what information policies can see, how they're scored,
what workloads they're tested against, and simulator fidelity parameters.

Based on the V1→V5 design decisions documented in SIMULATOR_EVOLUTION_ANALYSIS.md.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any

import yaml


@dataclass
class SimulatorConfig:
    """
    A declarative configuration defining a simulator variant.

    The outer loop evolves these configs. The inner loop (OpenEvolve)
    evolves policies within each config's simulator.
    """

    # --- Unique identifier ---
    name: str = "unnamed"

    # --- Information Exposure: BufferDescriptor fields ---
    # Which fields on BufferDescriptor are visible to the evolved policy.
    # If False, the field is masked (set to a default/zero value) in the
    # wrapper passed to the policy.
    expose_usage_count: bool = True         # PostgreSQL baseline (all versions)
    expose_access_count: bool = True        # Frequency tracking (V5)
    expose_last_access_time: bool = True    # Recency tracking (V5)
    expose_dirty_time: bool = True          # WAL flush cost modeling (V5)
    expose_is_dirty: bool = True            # Dirty flag (all versions)
    expose_block_group: bool = True         # PBM scan tracking (V5 innovation)
    expose_scan_ids: bool = True            # Which scans want this block (V5)

    # --- Information Exposure: Estimator methods ---
    # Which NextAccessEstimator methods are available to the policy.
    enable_estimator: bool = True                   # V5 core innovation
    enable_confidence: bool = False                 # estimate_with_confidence()
    enable_block_group_estimate: bool = False        # estimate_for_block_group()

    # --- Policy Interface ---
    policy_sees_scan_context: bool = True    # V5: scan_context arg passed to policy
    # Note: policy_sees_workload is always False — V1-V4's cheating bug is not reproduced

    # --- Simulator Parameters ---
    block_group_size: int = 128             # Blocks per group (V5: 128 = 1 MiB)
    simulate_hint_bit_dirtying: bool = True # 85% of reads dirty the page
    simulate_ring_buffers: bool = True      # BAS_BULKREAD ring buffers
    bgwriter_coverage: float = 0.30         # Background writer async coverage
    max_usage_count: int = 5                # PostgreSQL's BM_MAX_USAGE_COUNT

    # --- Workload Configuration ---
    workload_weights: Dict[str, float] = field(default_factory=lambda: {
        "tpch_full": 0.5,
        "tpcc_10w": 0.3,
        "ycsb_a": 0.2,
    })
    workload_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Scoring ---
    scoring_mode: str = "hit_rate"          # "hit_rate", "latency", "combined"
    dirty_penalty_weight: float = 0.0       # Penalty for sync dirty evictions
    reeviction_penalty_weight: float = 0.0  # Penalty for re-eviction (bad decisions)
    wal_cost_enabled: bool = False          # WAL flush cost in latency model
    # SSD I/O cost model (microseconds), used when scoring_mode includes latency
    latency_model: Dict[str, float] = field(default_factory=lambda: {
        "ssd_random_read_us": 100.0,
        "ssd_random_write_us": 200.0,
        "ssd_seq_read_us": 20.0,
        "memory_hit_us": 0.1,
    })

    # --- Feedback to LLM (artifacts) ---
    artifact_suggestions: bool = True       # Targeted improvement suggestions
    per_workload_breakdown: bool = True     # Per-workload score breakdown

    # --- Inner loop (OpenEvolve) parameters ---
    inner_iterations: int = 50              # OpenEvolve iterations per outer eval
    inner_population_size: int = 40         # MAP-Elites population
    inner_num_islands: int = 3              # Island count

    def config_id(self) -> str:
        """Deterministic hash of the config for deduplication."""
        # Exclude name and inner loop params from hash (they don't affect the simulator)
        d = asdict(self)
        for key in ("name", "inner_iterations", "inner_population_size", "inner_num_islands"):
            d.pop(key, None)
        raw = json.dumps(d, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_yaml(self, path: str) -> None:
        """Serialize to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulatorConfig":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def clone(self, **overrides) -> "SimulatorConfig":
        """Create a copy with optional field overrides."""
        d = asdict(self)
        d.update(overrides)
        return SimulatorConfig(**d)

    def diff(self, other: "SimulatorConfig") -> Dict[str, tuple]:
        """Return fields that differ between two configs."""
        d1, d2 = asdict(self), asdict(other)
        return {k: (d1[k], d2[k]) for k in d1 if d1[k] != d2[k]}

    def summary(self) -> str:
        """Human-readable one-line summary of key settings."""
        features = []
        if self.expose_block_group and self.expose_scan_ids:
            features.append("scan_tracking")
        if self.enable_estimator:
            features.append("estimator")
        if self.policy_sees_scan_context:
            features.append("scan_ctx")
        if self.expose_access_count:
            features.append("ac")
        if self.expose_last_access_time:
            features.append("lat")
        if self.expose_dirty_time:
            features.append("dt")
        wl = "+".join(f"{k}:{v}" for k, v in self.workload_weights.items())
        return f"[{self.name}] features={','.join(features)} scoring={self.scoring_mode} workloads={wl}"


# ---- Predefined configurations matching historical simulator versions ----

def v1_config() -> SimulatorConfig:
    """Recreates V1 (postgres_buffer) design decisions.
    No scan tracking, no estimator, hit-rate-only scoring.
    Policies could cheat via full workload visibility (we set this to False
    since we don't reproduce that bug)."""
    return SimulatorConfig(
        name="v1_baseline",
        expose_usage_count=True,
        expose_access_count=False,
        expose_last_access_time=False,
        expose_dirty_time=False,
        expose_is_dirty=True,
        expose_block_group=False,
        expose_scan_ids=False,
        enable_estimator=False,
        enable_confidence=False,
        enable_block_group_estimate=False,
        policy_sees_scan_context=False,
        simulate_hint_bit_dirtying=False,
        simulate_ring_buffers=False,
        bgwriter_coverage=0.0,
        scoring_mode="hit_rate",
        artifact_suggestions=False,
        per_workload_breakdown=False,
        workload_weights={"tpch_full": 1.0},
    )


def v3_config() -> SimulatorConfig:
    """Recreates V3 (simple_postgres_simulator) design decisions.
    Pluggable policy interface, configurable workloads, but no scan tracking."""
    return SimulatorConfig(
        name="v3_baseline",
        expose_usage_count=True,
        expose_access_count=False,
        expose_last_access_time=False,
        expose_dirty_time=False,
        expose_is_dirty=True,
        expose_block_group=False,
        expose_scan_ids=False,
        enable_estimator=False,
        enable_confidence=False,
        enable_block_group_estimate=False,
        policy_sees_scan_context=False,
        simulate_hint_bit_dirtying=False,
        simulate_ring_buffers=True,
        bgwriter_coverage=0.0,
        scoring_mode="hit_rate",
        artifact_suggestions=False,
        per_workload_breakdown=False,
        workload_weights={"tpch_full": 0.5, "tpcc_10w": 0.5},
    )


def v5_config() -> SimulatorConfig:
    """Recreates V5 (minimal_postgres_simulator) — the current best.
    Full scan tracking, estimator, block groups, multi-workload scoring."""
    return SimulatorConfig(
        name="v5_baseline",
        expose_usage_count=True,
        expose_access_count=True,
        expose_last_access_time=True,
        expose_dirty_time=True,
        expose_is_dirty=True,
        expose_block_group=True,
        expose_scan_ids=True,
        enable_estimator=True,
        enable_confidence=False,
        enable_block_group_estimate=False,
        policy_sees_scan_context=True,
        block_group_size=128,
        simulate_hint_bit_dirtying=True,
        simulate_ring_buffers=True,
        bgwriter_coverage=0.30,
        max_usage_count=5,
        scoring_mode="hit_rate",
        artifact_suggestions=True,
        per_workload_breakdown=True,
        workload_weights={"tpch_full": 0.5, "tpcc_10w": 0.3, "ycsb_a": 0.2},
    )


# ---- Ablation configs (V5 minus one feature) ----

def v5_no_scan_tracking() -> SimulatorConfig:
    """V5 without scan tracking (block groups + scan IDs disabled)."""
    return v5_config().clone(
        name="v5_no_scan_tracking",
        expose_block_group=False,
        expose_scan_ids=False,
        enable_estimator=False,
        enable_confidence=False,
        enable_block_group_estimate=False,
    )


def v5_no_estimator() -> SimulatorConfig:
    """V5 with block groups visible but estimator disabled."""
    return v5_config().clone(
        name="v5_no_estimator",
        enable_estimator=False,
    )


def v5_no_scan_context() -> SimulatorConfig:
    """V5 without scan context passed to policy."""
    return v5_config().clone(
        name="v5_no_scan_context",
        policy_sees_scan_context=False,
    )


def v5_no_frequency() -> SimulatorConfig:
    """V5 without frequency/recency tracking fields."""
    return v5_config().clone(
        name="v5_no_frequency",
        expose_access_count=False,
        expose_last_access_time=False,
    )


def v5_latency_scoring() -> SimulatorConfig:
    """V5 with latency-based scoring instead of hit-rate."""
    return v5_config().clone(
        name="v5_latency_scoring",
        scoring_mode="latency",
    )


PRESET_CONFIGS = {
    "v1": v1_config,
    "v3": v3_config,
    "v5": v5_config,
    "v5_no_scan_tracking": v5_no_scan_tracking,
    "v5_no_estimator": v5_no_estimator,
    "v5_no_scan_context": v5_no_scan_context,
    "v5_no_frequency": v5_no_frequency,
    "v5_latency": v5_latency_scoring,
}
