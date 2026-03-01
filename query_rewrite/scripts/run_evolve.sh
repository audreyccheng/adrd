#!/bin/bash
# Run the evolved RuleSelector discovery loop
#
# Prerequisites:
#   - PostgreSQL with TPC-H and DSB databases loaded (SF10)
#   - Java 17+ installed
#   - ANTHROPIC_API_KEY set
#   - pip install -r requirements.txt
#
# Usage:
#   bash scripts/run_evolve.sh                     # default: 20 iterations
#   MAX_ITERATIONS=3 bash scripts/run_evolve.sh    # quick test
#
set -e

cd "$(dirname "$0")/.."

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "[ERROR] ANTHROPIC_API_KEY not set"
    exit 1
fi

MAX_ITERATIONS="${MAX_ITERATIONS:-20}"

echo "============================================"
echo " Evolved RuleSelector Discovery Loop"
echo "============================================"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Config: evolve_loop/default_config.yaml"
echo ""

PYTHONPATH=. python3 -m evolve_loop.evolve_loop \
    --config evolve_loop/default_config.yaml \
    --max-iterations "$MAX_ITERATIONS" \
    2>&1 | tee evolve_run.log
