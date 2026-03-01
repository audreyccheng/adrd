"""
LLM-driven strategy proposal: analyze discrepancies and generate improved evaluators.

Uses Gemini to propose evaluation strategies by:
1. Analyzing why the current metric disagrees with ground truth
2. Drawing on domain knowledge (PostgreSQL, cost estimation, latency measurement)
3. Generating complete evaluator .py files that follow the OpenEvolve contract
"""

import re
from typing import Optional

from .config import OuterLoopConfig
from .llm_client import GeminiClient
from .strategy import EvaluationStrategy, StrategyHistory


# Read the existing evaluator.py to use as seed/reference
def _load_seed_evaluator() -> str:
    """Load the existing cost-based evaluator as the initial template."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "evaluator.py")
    with open(path) as f:
        return f.read()


SYSTEM_PROMPT = """\
You are an expert in database systems, PostgreSQL internals, and evolutionary \
optimization. You are designing evaluation functions for an automated system \
(OpenEvolve) that evolves index selection algorithms for PostgreSQL.

## Your Task
Generate a Python evaluator module that scores index selection programs. \
The evaluator will be used as the fitness function in an evolutionary search \
over index selection strategies.

## Evaluator Contract
The module MUST define:
```python
def evaluate(program_path: str, benchmark: str = "tpch") -> Dict[str, float]:
    # Must return a dict containing at minimum 'combined_score'
    # Higher combined_score = better program
    ...
```

The evolved programs have a `run_index_selection()` function that returns:
  `(selected_indexes, selection_time, baseline_cost, optimized_cost)`

Each index has: `.columns` (list of Column), `.table()` (table name), \
`.estimated_size` (bytes).

## Available Infrastructure

### Cost Estimation (fast, deterministic, but inaccurate)
Programs use HypoPG hypothetical indexes + EXPLAIN to estimate cost.
- `baseline_cost`: total frequency-weighted cost without indexes
- `optimized_cost`: total frequency-weighted cost with hypothetical indexes
- `cost_reduction = (baseline_cost - optimized_cost) / baseline_cost`
- Problem: cost estimates often ANTI-correlate with actual latency

### Latency Measurement (slow, noisy, but ground truth)
Uses real `CREATE INDEX` + `EXPLAIN (ANALYZE, FORMAT JSON)` for actual execution.
- Requires database connection (PostgreSQL on localhost)
- Noisy without precautions — measurement variance can be 17-32%

### Noise Reduction Techniques Available
1. **Interleaved warmup**: warmup Q1 → measure Q1 → warmup Q2 → measure Q2 ...
   (prevents cache eviction between warmup and measurement)
2. **pg_prewarm**: `SELECT pg_prewarm('table_name')` pins pages in shared_buffers
3. **Background suppression**: `CHECKPOINT` + disable autovacuum during measurement
4. **Multiple runs**: Execute each query N times, take median
5. **Query ordering**: Maintain consistent execution order across evaluations

### PostgreSQL Utilities Available (import paths)
```python
import sys, os
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps", "Index_EAB")
sys.path.insert(0, PROJECT_ROOT)

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index
```

Database connection:
```python
import configparser
db_conf = configparser.ConfigParser()
db_conf.read(f"{PROJECT_ROOT}/configuration_loader/database/db_con.conf")
connector = PostgresDatabaseConnector(db_conf, autocommit=True, db_name="benchbase")
```

### Benchmark Configurations
- tpch: db=benchbase, 18 queries, budget=500MB, max=15 indexes
- tpcds: db=benchbase_tpcds, 79 queries, budget=500MB, max=15 indexes
- job: db=benchbase_job, 33 queries, budget=2000MB, max=15 indexes

### Constraints (HARD — violation = score 0)
- Storage: sum of index sizes must be <= budget_mb
- Count: number of indexes must be <= max_indexes

## Key Domain Insight
Cost estimation (PostgreSQL optimizer estimates) often POORLY predicts actual \
query latency because:
1. Buffer cache blindness: optimizer doesn't know what's in shared_buffers
2. Cardinality estimation errors on multi-join queries
3. Crude I/O model (random_page_cost=4.0 assumes spinning disks, not SSDs)
4. Plan flips: adding an index can cause unpredictable plan changes

The goal is to find a metric that CORRELATES with actual query latency \
(ground truth) so that evolution optimizes for real performance.

## Output Format
Return ONLY a complete Python module inside a ```python code fence. \
The module must be self-contained, importable, and define `evaluate()`.
Include ALL necessary imports. Do NOT use placeholder comments — write complete code.
"""


class StrategyProposer:
    """Uses an LLM to propose evaluation strategies based on discrepancy analysis."""

    def __init__(self, llm: GeminiClient, config: OuterLoopConfig):
        self.llm = llm
        self.config = config
        self._seed_evaluator = _load_seed_evaluator()

    def create_initial_strategy(self) -> EvaluationStrategy:
        """Create the initial strategy: the existing cost-based evaluator."""
        return EvaluationStrategy(
            version=0,
            evaluator_code=self._seed_evaluator,
            rationale="Initial strategy: cost-based evaluation using PostgreSQL optimizer estimates (evaluator.py)",
        )

    def propose(self, history: StrategyHistory) -> EvaluationStrategy:
        """Propose a new evaluation strategy based on history and discrepancy analysis.

        Args:
            history: All previously tried strategies and their results.

        Returns:
            New EvaluationStrategy with LLM-generated evaluator code.
        """
        version = len(history.strategies)

        # Build user prompt
        user_prompt = self._build_user_prompt(history)

        # Call LLM
        print(f"\nProposing strategy v{version}...")
        response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Parse evaluator code from response
        code = self._extract_code(response)
        rationale = self._extract_rationale(response)

        if not code:
            return EvaluationStrategy(
                version=version,
                evaluator_code="",
                rationale=rationale or "LLM failed to generate code",
                error="No Python code block found in LLM response",
            )

        # Basic validation
        validation_error = self._validate_code(code)
        if validation_error:
            return EvaluationStrategy(
                version=version,
                evaluator_code=code,
                rationale=rationale,
                error=f"Code validation failed: {validation_error}",
            )

        return EvaluationStrategy(
            version=version,
            evaluator_code=code,
            rationale=rationale,
        )

    def _build_user_prompt(self, history: StrategyHistory) -> str:
        """Build the user prompt with context from previous iterations."""
        parts = []

        if not history.strategies:
            # First proposal (after seed fails)
            parts.append(
                "The initial cost-based evaluator has poor correlation with actual "
                "query latency. Design a better evaluator.\n"
                "\nHere is the current cost-based evaluator code:\n"
                f"```python\n{self._seed_evaluator}\n```\n"
            )
        else:
            # Show latest strategy + discrepancy
            latest = history.strategies[-1]
            parts.append(f"## Current evaluator (v{latest.version})\n")
            parts.append(f"Rationale: {latest.rationale}\n")

            if latest.discrepancy_report:
                parts.append(f"\n## Discrepancy Analysis\n{latest.discrepancy_report}\n")

            if latest.error:
                parts.append(f"\n## Error\n{latest.error}\n")

            parts.append(f"\n## Current evaluator code:\n```python\n{latest.evaluator_code}\n```\n")

            # Show history
            parts.append(f"\n{history.history_summary()}\n")

        parts.append(
            "\n## Instructions\n"
            "Based on the discrepancy analysis above, generate an IMPROVED evaluator "
            "that better predicts actual query latency (ground truth). "
            "Focus on addressing the specific issues identified.\n"
            "\nReturn a complete, self-contained Python module with an "
            "`evaluate(program_path, benchmark='tpch')` function."
        )

        return "\n".join(parts)

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find ```python ... ``` blocks
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the longest code block (likely the full evaluator)
            return max(matches, key=len).strip()

        # Try generic code blocks
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        return None

    def _extract_rationale(self, response: str) -> str:
        """Extract rationale/explanation from LLM response (text before code)."""
        # Get text before the first code block
        code_start = response.find("```")
        if code_start > 0:
            rationale = response[:code_start].strip()
            # Take last paragraph as the key insight
            paragraphs = [p.strip() for p in rationale.split("\n\n") if p.strip()]
            if paragraphs:
                return paragraphs[-1][:200]
        return "LLM-proposed evaluation strategy"

    def _validate_code(self, code: str) -> Optional[str]:
        """Basic validation of generated evaluator code.

        Returns None if valid, error message if invalid.
        """
        # Check syntax
        try:
            compile(code, "<generated_evaluator>", "exec")
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"

        # Check for evaluate function
        if "def evaluate(" not in code:
            return "Missing 'def evaluate(' function"

        # Check for combined_score in return
        if "combined_score" not in code:
            return "Code does not reference 'combined_score'"

        return None
