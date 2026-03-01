# Evolved RuleSelector for SQL Query Rewriting

An LLM-driven evolutionary system that discovers optimized Apache Calcite rewrite rule combinations for SQL query optimization. Achieves **5.9x workload speedup** over unoptimized queries and **3.1x faster** than the state-of-the-art R-Bot system (VLDB 2025).

## Architecture

The system consists of two loops:

- **Outer Loop** (`evolve_loop/`): Evolutionary discovery of RuleSelector patterns using Claude as the analyst/implementer. Runs a 5-phase iteration: SEARCH → ANALYZE → IMPLEMENT → VALIDATE → FIX.
- **Inner Loop** (`rbot/`): R-Bot query rewrite pipeline (LLM-based rule selection + Calcite rule application). Used as the baseline comparison. *(Git submodule — see setup below.)*

```
                 ┌─────────────────────────────────────────┐
                 │           Outer Loop (evolve_loop/)      │
                 │                                          │
                 │  1. SEARCH: brute-force rule combos      │
                 │  2. ANALYZE: Claude discovers patterns   │
                 │  3. IMPLEMENT: Claude generates Java     │
                 │  4. VALIDATE: rebuild JAR, test queries  │
                 │  5. FIX: guards for regressions          │
                 │                                          │
                 │         ┌──────────────┐                 │
                 │         │ RuleSelector │ ← evolved Java  │
                 │         │   .java      │   code          │
                 │         └──────┬───────┘                 │
                 └────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Apache Calcite HepPlanner │
                    │   (LearnedRewrite.jar)      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   PostgreSQL (SF10)         │
                    │   TPC-H + DSB benchmarks    │
                    └─────────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **Java 17+** (for Calcite compilation)
- **PostgreSQL** with TPC-H SF10 and DSB SF10 databases loaded
- **Anthropic API key** (for Claude, used by the outer loop)

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. (Optional) Link R-Bot as git submodule for inner loop comparison
# git submodule add <R-Bot-repo-URL> rbot/
# cd rbot && pip install -r requirements.txt

# 4. Verify Java artifacts
ls java/LearnedRewrite.jar    # pre-built JAR
ls java/RuleSelector.java     # canonical evolved RuleSelector (46-win policy)

# 5. Configure PostgreSQL connection (edit evolve_loop/default_config.yaml or set env vars)
export PGUSER="your_user"
export PGPASSWORD="your_password"
```

## Usage

### Run the Evolution Loop

```bash
# Full run (default: 20 iterations)
bash scripts/run_evolve.sh

# Quick test (3 iterations)
MAX_ITERATIONS=3 bash scripts/run_evolve.sh

# With custom config
PYTHONPATH=. python -m evolve_loop.evolve_loop \
    --config evolve_loop/default_config.yaml \
    --max-iterations 5
```

### Rebuild JAR After Editing RuleSelector.java

```bash
bash java/rebuild_jar.sh
```

## Directory Structure

```
qr_clean/
├── evolve_loop/           Outer loop: evolutionary RuleSelector discovery
│   ├── evolve_loop.py     Main orchestrator (5-phase iteration)
│   ├── config.py          Configuration (paths, rules, thresholds)
│   ├── state.py           Checkpointing and convergence detection
│   ├── searcher.py        Phase 1: brute-force rule combination search
│   ├── analyst.py         Phase 2: Claude analyzes search results
│   ├── implementer.py     Phase 3: Claude generates Java RuleSelector
│   ├── validator.py       Phase 4: rebuild JAR and validate on all queries
│   ├── fixer.py           Phase 5: fix regressions with guards
│   ├── guard_generator.py Deterministic early-exit guard injection
│   ├── features.py        Query feature extraction via Java bridge
│   ├── validate_worker.py Subprocess for fresh JVM validation
│   ├── utils/             Java bridge, PG runner, JAR builder, code extraction
│   └── prompts/           System prompts for Claude (analyst, implementer, fixer)
├── rbot/                  Inner loop: R-Bot (TODO: git submodule)
├── benchmarks/            SQL benchmark queries
│   ├── tpch/              TPC-H (22 templates, 44 instances)
│   └── dsb/               DSB (37 templates, 76 instances)
├── java/                  Java source and pre-built JAR
│   ├── RuleSelector.java  Evolved rule selector (canonical, 46-win policy)
│   ├── QueryAnalyzer.java Feature extraction for query classification
│   ├── Rewriter.java      Calcite HepPlanner wrapper
│   ├── LearnedRewrite.jar Pre-built JAR with all Calcite dependencies
│   └── rebuild_jar.sh     Recompile Java and update JAR
├── scripts/
│   └── run_evolve.sh      Quick-start runner
└── requirements.txt       Python dependencies
```

## R-Bot Submodule

The R-Bot inner loop is not included directly. To set it up:

```bash
git submodule add <R-Bot-repo-URL> rbot/
git submodule update --init
cd rbot && pip install -r requirements.txt
```

The R-Bot repo provides the LLM-based query rewrite pipeline used as the baseline in our evaluation. The outer loop (`evolve_loop/`) does **not** depend on R-Bot Python code — it only shares the benchmark data (`benchmarks/`) and Java artifacts (`java/`).

## Configuration

Edit `evolve_loop/default_config.yaml` or pass a custom config:

```yaml
# Key parameters
search_timeout_sec: 120        # Per-query timeout for search phase
validation_runs: 5             # Measurement runs per query
win_threshold: 1.10            # Speedup > 1.1x counts as a win
regression_threshold: 0.95     # Speedup < 0.95x counts as a regression
model: "claude-sonnet-4-20250514"  # Claude model for analysis/implementation
max_iterations: 20             # Evolution iterations
```

PostgreSQL databases are configured in `evolve_loop/config.py`:
- TPC-H: `benchbase_tpch_sf10`
- DSB: `benchbase_tpcds_sf10`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for analysis/implementation |
| `PGUSER` | No | PostgreSQL user (default: current OS user) |
| `PGPASSWORD` | No | PostgreSQL password (default: empty) |
| `PGHOST` | No | PostgreSQL host (default: localhost) |
| `PGPORT` | No | PostgreSQL port (default: 5432) |
| `JAVA_HOME` | No | Java installation path (auto-detected) |
