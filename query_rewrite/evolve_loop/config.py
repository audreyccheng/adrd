"""
Configuration for the Calcite rule evolution loop.
"""

import os
import getpass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Repo root (relative to this file)
_REPO_ROOT = Path(__file__).parent.parent

# R-Bot inner loop (TODO: link as git submodule)
# git submodule add <R-Bot-repo-URL> rbot/
_RBOT_ROOT = _REPO_ROOT / "rbot"


@dataclass
class EvolutionConfig:
    """Configuration for the evolution loop."""

    # Queries
    query_dirs: Dict[str, str] = field(default_factory=lambda: {
        "tpch": str(_REPO_ROOT / "benchmarks" / "tpch"),
        "dsb": str(_REPO_ROOT / "benchmarks" / "dsb"),
    })
    benchmarks: List[str] = field(default_factory=lambda: ["tpch", "dsb"])

    # Optional query subset for faster evaluation.
    # When non-empty, only these queries are used in search/validation.
    # Set via config YAML or EVAL_SUBSET env var.
    # Format: {"tpch": ["query17_0", ...], "dsb": ["query081_0", ...]}
    query_subset: Dict[str, List[str]] = field(default_factory=dict)

    # Search
    main_rules: List[str] = field(default_factory=lambda: [
        "FILTER_SUB_QUERY_TO_CORRELATE",   # FSQ
        "JOIN_TO_CORRELATE",                # JTC
        "FILTER_INTO_JOIN",                 # FIJ
        "SORT_REMOVE_CONSTANT_KEYS",        # SRCK
        "PROJECT_REMOVE",                   # PR
        "JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE",  # JDNF
    ])
    pre_finishers: List[str] = field(default_factory=lambda: [
        "AGGREGATE_REDUCE_FUNCTIONS",       # ARF
        "AGGREGATE_PROJECT_MERGE",          # APM
        "FILTER_REDUCE_EXPRESSIONS",        # FRE
        "AGGREGATE_CASE_TO_FILTER",         # ACTF
        "FILTER_AGGREGATE_TRANSPOSE",       # FAT
        "PROJECT_MERGE",                    # PM
        "FILTER_MERGE",                     # FM
    ])
    post_finishers: List[str] = field(default_factory=lambda: [
        "SORT_PROJECT_TRANSPOSE",           # SPT
        "PROJECT_FILTER_TRANSPOSE",         # PFT
        "PROJECT_REDUCE_EXPRESSIONS",       # PRE
        "SORT_REMOVE_CONSTANT_KEYS",        # SRCK
        "PROJECT_REMOVE",                   # PR
        "JOIN_DERIVE_IS_NOT_NULL_FILTER_RULE",  # JDNF
        "FILTER_MERGE",                     # FM
        "SORT_REMOVE",                      # SR
    ])
    search_timeout_sec: int = 120
    search_parallelism: int = 4

    # Adaptive search
    adaptive_max_queries_per_iter: int = 60   # max queries per adaptive iteration
    adaptive_max_combos_per_query: int = 30   # max combos per query in adaptive mode
    bootstrap_iterations: int = 1             # how many iterations use broad sweep

    # Safety: queries known to be dangerous with certain rules
    fsq_dangerous: List[str] = field(default_factory=lambda: [
        "query2_0", "query2_1", "query21_0", "query21_1",
    ])
    fij_dangerous: List[str] = field(default_factory=lambda: [
        "query91_0", "query91_1", "query102_0", "query102_1",
    ])

    # Validation
    validation_runs: int = 5
    validation_warmup: int = 1
    regression_threshold: float = 0.95   # speedup < this = regression
    win_threshold: float = 1.10          # speedup > this = win

    # Claude API
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 32768

    # Loop
    max_iterations: int = 20
    max_fix_attempts: int = 3
    checkpoint_dir: str = str(_REPO_ROOT / "evolve_output")

    # Train/test split: only use queries with this suffix during search.
    # Validation always uses all variants. Set to "" to disable splitting.
    train_suffix: str = "_0"

    # Java
    canonical_ruleselector: str = str(_REPO_ROOT / "java" / "RuleSelector.java")
    canonical_queryanalyzer: str = str(_REPO_ROOT / "java" / "QueryAnalyzer.java")
    jar_dir: str = str(_REPO_ROOT / "java")
    rebuild_script: str = str(_REPO_ROOT / "java" / "rebuild_jar.sh")
    build_timeout_sec: int = 300          # JAR rebuild timeout (seconds)
    validation_timeout_sec: int = 7200    # Validation subprocess timeout (seconds)

    # PostgreSQL
    pg_configs: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "tpch": {
            "host": "localhost",
            "port": "5432",
            "dbname": "benchbase_tpch_sf10",
            "user": os.environ.get("PGUSER", getpass.getuser()),
            "password": "",
        },
        "dsb": {
            "host": "localhost",
            "port": "5432",
            "dbname": "benchbase_tpcds_sf10",
            "user": os.environ.get("PGUSER", getpass.getuser()),
            "password": "",
        },
    })


def load_config(path: Optional[str] = None) -> EvolutionConfig:
    """Load config from YAML file, falling back to defaults."""
    config = EvolutionConfig()

    if path and Path(path).exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Apply overrides from YAML
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Resolve API key from environment if not set
    if not config.anthropic_api_key:
        config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    return config
