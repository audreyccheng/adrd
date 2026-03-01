# Index Selection Evolution Framework

Co-evolution of index selection policies and evaluation metrics for PostgreSQL, using LLM-driven evolutionary search.

## Overview

This framework has two loops:

- **Inner loop** (OpenEvolve): Evolves index selection algorithms by mutating Python programs and scoring them with a fitness function (evaluator). Discovers novel heuristics that outperform classical baselines (AutoAdmin, DB2Advis, Extend, etc.).

- **Outer loop**: Evolves the evaluation metric itself. Uses an LLM to propose evaluator code, validates it by checking ranking agreement with ground truth (actual query latency), and iteratively discovers that denoised latency — not optimizer cost estimates — is the right fitness signal.

## Setup

### 1. Clone dependencies

This repo requires two git submodules in `deps/`. See [`deps/README.md`](deps/README.md) for instructions.

```bash
# After linking submodules:
cd deps/Index_EAB && pip install -r requirements.txt  # if exists
cd deps/openevolve && pip install -e ".[dev]"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Database setup

PostgreSQL with HypoPG extension required. Configure connection in `deps/Index_EAB/configuration_loader/database/db_con.conf`.

```bash
# Load benchmark data (via Index_EAB scripts)
cd deps/Index_EAB && bash load_tpch.sh
```

Benchmark databases: `benchbase` (TPC-H), `benchbase_tpcds` (TPC-DS), `benchbase_job` (JOB/IMDB).

### 4. API keys

```bash
export GEMINI_API_KEY="..."      # For outer loop (Gemini)
export OPENAI_API_KEY="..."      # For inner loop (OpenEvolve)
```

## Usage

### Inner loop: Evolve index selection algorithms

```bash
# Quick run (subset queries, 10 iterations)
python run_openevolve.py \
  -c configs/config.yaml \
  -i initial_programs/initial_program_extend.py \
  --iterations 10

# Full evaluation (all queries)
python run_openevolve.py \
  -c configs/config_all_workloads.yaml \
  -i initial_programs/best_explore_extend_1215.py \
  --full
```

### Outer loop: Discover the right evaluation metric

```bash
# Auto-discover evaluation metric (default: TPC-H, 10 iterations)
python -m outer_loop.run_outer_loop

# With custom config
python -m outer_loop.run_outer_loop -c outer_loop/config_outer_loop.yaml

# Override benchmark
python -m outer_loop.run_outer_loop --benchmark tpcds --iterations 20

# Use the discovered evaluator with the inner loop
python run_openevolve.py \
  -c configs/config.yaml \
  -i initial_programs/initial_program_extend.py \
  --evaluator outer_loop_outputs/best_evaluator.py
```

The outer loop outputs a validated evaluator `.py` file that can be passed to the inner loop.

### Evaluate programs directly

```bash
# Cost-based evaluation
BENCHMARK=tpch python evaluator.py initial_programs/initial_program_extend.py

# Latency-based evaluation (interleaved warmup, low noise)
BENCHMARK=tpch python evaluator_latency_interleaved.py initial_programs/initial_program_extend.py

# Standalone latency comparison
python latency_evaluator.py --benchmark tpch --algorithm extend
```

### Run the demo (no database needed)

```bash
python outer_loop/test_outer_loop.py
```

This runs the full outer loop pipeline with pre-computed data from our experiments, demonstrating how cost estimation is anti-correlated with actual performance on TPC-H (Spearman = -0.80) and how denoised latency achieves perfect ranking agreement.

## Architecture

```
index_clean/
├── run_openevolve.py                  # Inner loop entry point
├── evaluator.py                       # Cost-based fitness (fast, deterministic)
├── evaluator_full.py                  # Cost + reliability weights + latency validation
├── evaluator_latency.py               # Raw latency fitness (sequential warmup)
├── evaluator_latency_interleaved.py   # Denoised latency fitness (best protocol)
├── latency_evaluator.py               # Standalone benchmarking tool
├── query_reliability_weights.json     # Per-query cost-latency correlation weights
├── configs/                           # Evolution YAML configs (LLM, evaluation, prompts)
├── initial_programs/                  # Baseline algorithms + evolved programs
├── outer_loop/                        # Evaluation metric co-evolution
│   ├── outer_loop.py                  # Main orchestrator
│   ├── strategy_proposer.py           # LLM prompting (Gemini)
│   ├── discrepancy_analyzer.py        # Spearman correlation analysis
│   ├── evaluator_runner.py            # Subprocess-isolated evaluator execution
│   ├── ground_truth.py                # Actual latency measurement (cached)
│   └── ...
└── deps/                              # Git submodules (see deps/README.md)
    ├── Index_EAB/                     # Database utilities, schemas, workloads
    └── openevolve/                    # OpenEvolve evolution framework
```

### Dependencies via `deps/`

| Submodule | Provides | Used by |
|-----------|----------|---------|
| `Index_EAB/` | `index_advisor_selector/` (Workload, Index, PostgresDatabaseConnector), `configuration_loader/` (DB configs, schemas), `workload_generator/` (benchmark queries) | All evaluators, all initial programs |
| `openevolve/` | Evolution engine (MAP-Elites, LLM mutation, cascade evaluation) | `run_openevolve.py` |

## Evaluator Comparison

| Evaluator | Metric | Speed | Noise | Ranking Agreement |
|-----------|--------|-------|-------|-------------------|
| `evaluator.py` | Cost reduction | ~10s | 0% (deterministic) | Poor (rho ~ -0.80 on TPC-H) |
| `evaluator_full.py` | Reliability-weighted cost | ~30s | 0% | Moderate (rho ~ 0.38) |
| `evaluator_latency.py` | Raw latency | ~5min | 17-32% | Good when measurable |
| `evaluator_latency_interleaved.py` | Denoised latency | ~5min | 0.1-0.3% | Excellent (rho ~ 1.0) |

## Key Insight

Cost estimation (PostgreSQL optimizer estimates) is **anti-correlated** with actual query latency because the optimizer is blind to buffer cache contents, uses crude I/O models, and makes cardinality estimation errors. The outer loop discovers this automatically and converges on denoised latency as the right fitness signal.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK` | `tpch` | Benchmark: `tpch`, `tpcds`, `job`, `all` |
| `GEMINI_API_KEY` | — | API key for outer loop LLM (Gemini) |
| `OPENAI_API_KEY` | — | API key for inner loop LLM (OpenEvolve) |
| `LATENCY_NUM_RUNS` | `1` | Runs per query for latency measurement |
| `EVAL_PREWARM` | `1` | Use pg_prewarm to pin data in shared_buffers |
| `EVAL_SUPPRESS_BG` | `1` | Force CHECKPOINT + disable autovacuum |
| `INDEX_PROJECT_ROOT` | `deps/Index_EAB` | Override path to Index_EAB testbed |

## Constraints

All evaluators enforce hard constraints (score = 0 if violated):
- **Storage budget**: 500 MB (SF=1), 5 GB (SF=10)
- **Max indexes**: 15

## Scoring

- Cost evaluator: `combined_score = 0.80 * cost_reduction + 0.20`
- Latency evaluator: `combined_score = -latency_seconds` (OpenEvolve maximizes)
