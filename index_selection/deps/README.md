# Dependencies (Git Submodules)

This directory contains git submodule mount points. Link them before running.

## Setup

```bash
# From the index_clean/ directory:

# Option 1: Git submodules (if this is its own repo)
git submodule add <Index_EAB_repo_url> deps/Index_EAB
git submodule add <openevolve_repo_url> deps/openevolve

# Option 2: Symlinks (for development within the parent repo)
ln -s /path/to/Index_EAB deps/Index_EAB
ln -s /path/to/openevolve deps/openevolve

# Option 3: Environment variables (override paths without linking)
export INDEX_PROJECT_ROOT=/path/to/Index_EAB
export OPENEVOLVE_DIR=/path/to/openevolve
```

## What each dependency provides

### `Index_EAB/`

The main index advisor testbed repository. Provides:

- **`index_advisor_selector/`** — Shared utilities:
  - `Workload` — Query workload representation
  - `Index` — Index object with columns, table, estimated_size
  - `PostgresDatabaseConnector` — Database connection wrapper
  - `CostEvaluation` — HypoPG-based cost estimation
  - `heu_com` — Schema/query parsing utilities
- **`configuration_loader/database/`** — Database configs and schemas:
  - `db_con.conf` — PostgreSQL connection settings
  - `schema_tpch.json`, `schema_tpcds.json`, `schema_job.json`, etc.
- **`workload_generator/template_based/`** — Benchmark query workloads:
  - `tpch_work_temp_multi_freq.json`, `tpcds_work_temp_multi_freq.json`, etc.

### `openevolve/`

The OpenEvolve evolution framework (AlphaEvolve open-source). Provides:

- **`openevolve-run.py`** — CLI entry point for running evolution
- **`openevolve/`** — Core library:
  - MAP-Elites island-based evolution
  - LLM ensemble for code mutation
  - Cascade evaluation pipeline
  - Checkpoint/resume support

## Verification

After linking, verify the dependencies are accessible:

```bash
# Check Index_EAB
ls deps/Index_EAB/index_advisor_selector/
ls deps/Index_EAB/configuration_loader/database/db_con.conf

# Check openevolve
ls deps/openevolve/openevolve-run.py
python -c "import sys; sys.path.insert(0, 'deps/openevolve'); import openevolve; print('OK')"
```
