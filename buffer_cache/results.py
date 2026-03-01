"""
Result tracking for the two-level simulator evolution framework.

Records (config, simulator_score, real_pg_score) across generations
and provides analysis methods for simulator fidelity, ranking, and progress.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class InnerResult:
    """Result from one inner loop (OpenEvolve) run."""
    config_id: str                      # SimulatorConfig hash
    config_name: str                    # Human-readable name
    simulator_score: float              # Best policy's combined_score
    best_policy_path: str               # Path to best_program.py
    best_policy_code: str = ""          # Source code of best policy
    per_workload_scores: Dict[str, float] = field(default_factory=dict)
    iterations_run: int = 0
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class BenchmarkResult:
    """Result from a real PostgreSQL benchmark."""
    throughput: float = 0.0             # Transactions per second
    hit_rate: float = 0.0               # Buffer cache hit rate (0-1)
    disk_reads: int = 0                 # Total disk reads
    runtime_seconds: float = 0.0        # Benchmark wall-clock time
    benchmark_type: str = "tpch"        # "tpch", "tpcc", "chbench"
    config_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OuterResult:
    """Combined result for one outer loop iteration."""
    generation: int
    config_id: str
    config_name: str
    config_dict: Dict[str, Any]         # Full SimulatorConfig as dict

    # Inner loop results
    inner_result: Optional[InnerResult] = None

    # Translation results
    translation_success: bool = False
    translated_c_code: str = ""
    compile_success: bool = False

    # Real PostgreSQL benchmark results
    benchmark_result: Optional[BenchmarkResult] = None

    # Derived metrics
    simulator_score: float = 0.0        # From inner loop
    real_pg_score: float = 0.0          # From benchmark (throughput or hit_rate)
    fidelity_gap: float = 0.0           # |simulator_score - real_pg_score|

    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class ResultTracker:
    """
    Tracks results across all outer loop generations.
    Persists to JSONL for crash recovery and analysis.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        self.results: List[OuterResult] = []
        self._load_existing()

    def _load_existing(self):
        """Load previously recorded results for resume support."""
        if self.results_file.exists():
            with open(self.results_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Reconstruct nested dataclasses
                        if data.get("inner_result"):
                            data["inner_result"] = InnerResult(**data["inner_result"])
                        if data.get("benchmark_result"):
                            data["benchmark_result"] = BenchmarkResult(**data["benchmark_result"])
                        self.results.append(OuterResult(**data))

    def record(self, result: OuterResult):
        """Record an outer loop result and persist immediately."""
        self.results.append(result)
        with open(self.results_file, "a") as f:
            d = asdict(result)
            f.write(json.dumps(d) + "\n")

    def best_config(self) -> Optional[OuterResult]:
        """Return the config with the highest real PostgreSQL score."""
        benchmarked = [r for r in self.results if r.benchmark_result is not None]
        if not benchmarked:
            return None
        return max(benchmarked, key=lambda r: r.real_pg_score)

    def best_by_generation(self) -> Dict[int, OuterResult]:
        """Return the best result per generation."""
        by_gen: Dict[int, OuterResult] = {}
        for r in self.results:
            gen = r.generation
            if gen not in by_gen or r.real_pg_score > by_gen[gen].real_pg_score:
                by_gen[gen] = r
        return by_gen

    def fidelity_correlation(self) -> Optional[float]:
        """
        Pearson correlation between simulator_score and real_pg_score.
        Higher correlation = more faithful simulator.
        Returns None if fewer than 3 benchmarked results.
        """
        benchmarked = [r for r in self.results if r.benchmark_result is not None]
        if len(benchmarked) < 3:
            return None

        sim_scores = [r.simulator_score for r in benchmarked]
        pg_scores = [r.real_pg_score for r in benchmarked]

        n = len(sim_scores)
        mean_s = sum(sim_scores) / n
        mean_p = sum(pg_scores) / n

        cov = sum((s - mean_s) * (p - mean_p) for s, p in zip(sim_scores, pg_scores)) / n
        std_s = (sum((s - mean_s) ** 2 for s in sim_scores) / n) ** 0.5
        std_p = (sum((p - mean_p) ** 2 for p in pg_scores) / n) ** 0.5

        if std_s == 0 or std_p == 0:
            return 0.0
        return cov / (std_s * std_p)

    def ranking_table(self) -> str:
        """Human-readable ranking table of all evaluated configs."""
        benchmarked = sorted(
            [r for r in self.results if r.benchmark_result is not None],
            key=lambda r: r.real_pg_score,
            reverse=True,
        )
        if not benchmarked:
            return "No benchmarked results yet."

        lines = [
            f"{'Rank':<5} {'Config':<25} {'Gen':<5} {'Sim Score':<12} {'PG Score':<12} {'Throughput':<12} {'Hit Rate':<10} {'Fidelity Gap':<12}",
            "-" * 93,
        ]
        for i, r in enumerate(benchmarked, 1):
            br = r.benchmark_result
            lines.append(
                f"{i:<5} {r.config_name:<25} {r.generation:<5} "
                f"{r.simulator_score:<12.4f} {r.real_pg_score:<12.4f} "
                f"{br.throughput:<12.4f} {br.hit_rate:<10.2%} "
                f"{r.fidelity_gap:<12.4f}"
            )
        return "\n".join(lines)

    def generation_progress(self) -> str:
        """Show best real_pg_score per generation."""
        by_gen = self.best_by_generation()
        if not by_gen:
            return "No results yet."

        lines = [f"{'Gen':<5} {'Config':<25} {'PG Score':<12} {'Sim Score':<12}"]
        for gen in sorted(by_gen.keys()):
            r = by_gen[gen]
            lines.append(f"{gen:<5} {r.config_name:<25} {r.real_pg_score:<12.4f} {r.simulator_score:<12.4f}")
        return "\n".join(lines)

    def save_summary(self):
        """Save human-readable summary to output directory."""
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SIM EVOLVER RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("RANKING:\n")
            f.write(self.ranking_table() + "\n\n")
            f.write("GENERATION PROGRESS:\n")
            f.write(self.generation_progress() + "\n\n")
            corr = self.fidelity_correlation()
            if corr is not None:
                f.write(f"SIMULATOR FIDELITY (Pearson r): {corr:.4f}\n")
            f.write(f"\nTotal evaluations: {len(self.results)}\n")
            best = self.best_config()
            if best:
                f.write(f"Best config: {best.config_name} (PG score: {best.real_pg_score:.4f})\n")
