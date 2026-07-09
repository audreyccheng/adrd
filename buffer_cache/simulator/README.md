# minimal_postgres_simulator (bundled)

Self-contained copy of the V5 PostgreSQL buffer management simulator used by
`buffer_cache` for policy evaluation and inner-loop evolution.

Source of truth in the monorepo: `postgres/benchmarks/minimal_postgres_simulator/`.

## Layout

- `core/` — BufferManager, BufferDescriptor, block groups
- `scan_tracker/` — PBM scan registration and next-access estimation
- `workload/` — TPC-H, TPC-C, YCSB, CH-benchmark generators
- `policies/` — Reference replacement policies (clock sweep, PBM sampling)
- `evaluator.py` — Workload configs and standalone OpenEvolve evaluator
- `tests/test_basic.py` — Smoke tests

## Quick check

```bash
cd adrd/buffer_cache
python simulator/tests/test_basic.py
```

The outer loop (`sim_evolver.py`) imports this directory via `SIMEVOLVER_SIMULATOR_DIR`
(default: `./simulator/`).
