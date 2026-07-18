# Dependencies

Mount points for external checkouts used by evaluators and the inner loop.
Place a clone or symlink at each path (or override with env vars).

## Setup

```bash
# From index_selection/

# Option A: symlinks
ln -s /path/to/index_selection_evaluation deps/Index_EAB
ln -s /path/to/openevolve deps/openevolve

# Option B: env vars (no local link required)
export INDEX_PROJECT_ROOT=/path/to/index_selection_evaluation
export OPENEVOLVE_DIR=/path/to/openevolve
```

Index_EAB upstream: https://github.com/hyrise/index_selection_evaluation  
OpenEvolve upstream: https://github.com/algorithmicsuperintelligence/openevolve

## What each dependency provides

### `Index_EAB/`

Index advisor testbed. Provides:

- **`index_advisor_selector/`** — `Workload`, `Index`, `PostgresDatabaseConnector`, HypoPG cost evaluation
- **`configuration_loader/database/`** — `db_con.conf`, schema JSON files
- **`workload_generator/template_based/`** — benchmark query workloads

### `openevolve/`

OpenEvolve evolution framework. Provides:

- **`openevolve-run.py`** — CLI entry point
- **`openevolve/`** — MAP-Elites, LLM mutation, cascade evaluation, checkpoints

## Verification

```bash
ls deps/Index_EAB/index_advisor_selector/
ls deps/Index_EAB/configuration_loader/database/db_con.conf
ls deps/openevolve/openevolve-run.py
```
