# pg_clean — Two-Level Simulator Evolution for PostgreSQL Buffer Management

Automated framework that evolves **simulator configurations** (outer loop) to discover what information exposure enables the best **eviction policies** (inner loop, via OpenEvolve).

The key insight: the simulator's design — what information it exposes to policies, how it scores them, what workloads it tests — determines the ceiling of discoverable policies. This framework automates the search over simulator designs.

## Architecture

```
OUTER LOOP (SimEvolver — lightweight, LLM-guided)
│
│  Population of SimulatorConfigs
│  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ Variant A│  │ Variant B│  │ Variant C│  ...
│  └────┬─────┘  └────┬─────┘  └────┬─────┘
│       │              │              │
│       ▼              ▼              ▼
│  INNER LOOP (OpenEvolve, per variant)
│  - Generate evaluator from config
│  - Evolve eviction policies via LLM
│  - Return best_policy + score
│       │
│       ▼
│  TRANSLATION (LLM-assisted, Python → C)
│  - Translate best policy into postgres-pbm C code
│  - Compile postgres-pbm
│       │
│       ▼
│  BENCHMARK (real PostgreSQL)
│  - TPC-H via BenchBase
│  - Return throughput, hit_rate, disk_reads
│       │
│       ▼
│  Outer fitness = real PostgreSQL performance
│  LLM reasons about results → proposes next config
```

## Prerequisites

- Python 3.10+
- `pip install pyyaml` (required)
- `pip install openai` and/or `pip install anthropic` (optional, for LLM-guided mutation)

## Setup

Use the **`buffer-cache`** conda environment (recorded in `.conda-env`):

```bash
cd adrd/buffer_cache

# Create env (first time only)
conda env create -f environment.yml   # creates env: buffer-cache
conda activate buffer-cache
pip install openai anthropic flask
pip install -e ./openevolve

# Or if env already exists:
conda activate buffer-cache
```

Env path: `/opt/anaconda3/envs/buffer-cache` (Python 3.11)

The PBM buffer simulator is bundled in `simulator/`. OpenEvolve is bundled in
`openevolve/`. No symlinks to the postgres repo are required.

### postgres-pbm (optional — C translation + real benchmarks)

**Git submodule placeholder** — not yet linked. This is where a submodule to the `postgres-pbm` repo should be added:

```bash
# Future: git submodule add <postgres-pbm-repo-url> pg_clean/postgres-pbm
```

Not needed for simulator-only runs (`--skip-benchmark --skip-translation`).

## Quick Start

### Smoke test (standalone, no API key, ~1 min)

```bash
conda activate buffer-cache

python sim_evolver.py --quick --random-mutation --generations 1 --population 2 \
    --skip-benchmark --skip-translation --output smoke_output
```

### Ablation study (simulator-only, no API key, ~15–20 min)

```bash
# Run ablation: V5 baseline vs single-feature removals
python sim_evolver.py --ablation --skip-benchmark --skip-translation
```

### Full two-level evolution (requires API key)

```bash
export OPENAI_API_KEY=sk-...
pip install -e ./openevolve

python sim_evolver.py --config v5 --generations 3 --population 3 \
    --inner-iterations 10 --skip-benchmark --skip-translation \
    --output evolution_output
```

Outer loop mutates `SimulatorConfig` (what info/scoring/workloads policies see).
Inner loop (OpenEvolve) evolves eviction policies within each config.

### Full outer loop evolution (LLM-guided config mutation)

```bash
# Set up LLM API key for outer loop mutations
export OPENAI_API_KEY=sk-...

# Run 5 generations, population of 5, 50 inner iterations each
python sim_evolver.py \
    --config v1 \
    --generations 5 \
    --population 5 \
    --inner-iterations 50 \
    --skip-benchmark --skip-translation \
    --output sim_evolver_output
```

### Full pipeline (with C translation + PostgreSQL benchmarks)

```bash
export OPENAI_API_KEY=sk-...
export PBM_ROOT=~/pbm-exp  # PostgreSQL install root

python sim_evolver.py \
    --config v1 \
    --generations 10 \
    --population 5 \
    --output full_evolution_output
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIMEVOLVER_SIMULATOR_DIR` | `buffer_cache/simulator/` | Path to minimal_postgres_simulator |
| `SIMEVOLVER_OPENEVOLVE_RUN` | `buffer_cache/openevolve/openevolve-run.py` | Path to OpenEvolve entry point |
| `SIMEVOLVER_PBM_DIR` | `buffer_cache/postgres-pbm/` | Path to postgres-pbm source tree |
| `PBM_ROOT` | `~/pbm-exp` | PostgreSQL install root (for benchmarks) |
| `OPENAI_API_KEY` | — | OpenAI API key for LLM-guided mutation |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (alternative) |

## Files

| File / Directory | Purpose |
|------|---------|
| `openevolve/` | Bundled OpenEvolve for inner-loop policy evolution |
| `simulator_config.py` | `SimulatorConfig` dataclass — defines the search space (20+ evolvable parameters). V1/V3/V5 presets. |
| `mutations.py` | LLM-guided mutation (default) + random mutation (fallback). Dependency enforcement. |
| `evaluator_generator.py` | Generates `evaluator.py` + `initial_program.py` from a config. Information masking layer. |
| `sim_evolver.py` | Outer loop orchestrator. Population management, selection, CLI entry point. |
| `policy_translator.py` | LLM-assisted Python→C translation for postgres-pbm integration. |
| `pg_benchmarker.py` | PostgreSQL lifecycle + BenchBase TPC-H benchmark automation. |
| `results.py` | Result tracking with JSONL persistence. Fidelity correlation, ranking tables. |
| `configs/` | Preset YAML configs for V1, V3, V5 baselines and ablation variants. |

## Git Submodule Note

`postgres-pbm/` should eventually be added as a git submodule pointing to the postgres-pbm repository. This is not yet linked. To add it:

```bash
cd pg_clean
git submodule add <postgres-pbm-repo-url> postgres-pbm
```

This is only needed for the full pipeline (C translation + real PostgreSQL benchmarks). All simulator-only functionality works without it.
