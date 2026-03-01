"""
Ground truth measurement using the reliable interleaved latency protocol.

Measures actual query latency for each program in the corpus, caches results,
and serves as the "true north" for validating proxy metrics.
"""

import json
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, Optional

from .config import OuterLoopConfig
from .program_corpus import ProgramCorpus


class GroundTruth:
    """Measure and cache ground truth (actual latency) for the program corpus."""

    def __init__(self, config: OuterLoopConfig):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.join(self.base_dir, "deps", "Index_EAB")
        self._cache: Optional[Dict[str, float]] = None

    def measure(self, corpus: ProgramCorpus, force: bool = False) -> Dict[str, float]:
        """Measure ground truth latency for all programs.

        Uses the interleaved latency evaluator with full protocol
        (pg_prewarm + bg suppression + multiple runs).

        Results are cached to disk. Pass force=True to re-measure.

        Args:
            corpus: The program corpus to measure.
            force: If True, ignore cache and re-measure.

        Returns:
            Dict mapping program name → total latency in seconds.
            Lower latency = better program.
        """
        # Check cache
        cache_path = self.config.ground_truth.cache_file
        if not force and os.path.exists(cache_path):
            cached = self._load_cache(cache_path)
            # Verify cache covers all corpus programs
            if all(p.name in cached for p in corpus.programs):
                print(f"Using cached ground truth from {cache_path}")
                self._cache = cached
                return cached

        print("=" * 60)
        print("MEASURING GROUND TRUTH (actual latency)")
        print(f"Protocol: interleaved warmup, {self.config.ground_truth.num_runs} runs/query, "
              f"prewarm={self.config.ground_truth.prewarm}, bg_suppress={self.config.ground_truth.suppress_bg}")
        print(f"Benchmark: {self.config.benchmark}")
        print(f"Programs: {len(corpus)} total")
        print("=" * 60)

        results = {}
        for program in corpus.programs:
            print(f"\nMeasuring {program.name}...")
            start = time.time()
            latency = self._measure_single(program.path)
            elapsed = time.time() - start

            if latency is not None:
                results[program.name] = latency
                print(f"  {program.name}: {latency:.2f}s latency (measured in {elapsed:.0f}s)")
            else:
                print(f"  {program.name}: FAILED (measurement error)")

        # Save cache
        self._save_cache(cache_path, results)
        self._cache = results

        print(f"\nGround truth measured for {len(results)}/{len(corpus)} programs")
        print(f"Results cached to {cache_path}")
        return results

    def _measure_single(self, program_path: str) -> Optional[float]:
        """Measure ground truth latency for a single program.

        Calls evaluator_latency_interleaved.evaluate() via subprocess.
        """
        evaluator_path = self.config.ground_truth.evaluator

        runner_fd, runner_path = None, None
        results_fd, results_path = None, None
        try:
            import tempfile
            results_fd, results_path = tempfile.mkstemp(suffix=".gt_results", prefix="outer_")
            os.close(results_fd)
            results_fd = None

            runner_script = f"""
import sys
import os
import pickle
import traceback

os.environ['BENCHMARK'] = '{self.config.benchmark}'
os.environ['LATENCY_NUM_RUNS'] = '{self.config.ground_truth.num_runs}'
os.environ['EVAL_PREWARM'] = '{"1" if self.config.ground_truth.prewarm else "0"}'
os.environ['EVAL_SUPPRESS_BG'] = '{"1" if self.config.ground_truth.suppress_bg else "0"}'
os.environ['LATENCY_WARMUP_INDEXES'] = '1'

os.chdir('{self.project_root}')
sys.path.insert(0, '{self.project_root}')
sys.path.insert(0, '{self.base_dir}')

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("gt_eval", '{evaluator_path}')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    result = evaluator.evaluate('{program_path}')

    with open('{results_path}', 'wb') as f:
        pickle.dump(result, f)

except Exception as e:
    traceback.print_exc()
    with open('{results_path}', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""

            runner_fd, runner_path = tempfile.mkstemp(suffix="_gt.py", prefix="outer_")
            with os.fdopen(runner_fd, "w") as f:
                f.write(runner_script)
            runner_fd = None  # fd is closed

            # Long timeout for latency measurement
            timeout = 3600  # 1 hour

            process = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"  Ground truth measurement timed out after {timeout}s")
                return None

            if not os.path.exists(results_path):
                print(f"  Results file not created. stderr: {stderr.decode()[-300:]}")
                return None

            with open(results_path, "rb") as f:
                result = pickle.load(f)

            if "error" in result:
                print(f"  Measurement error: {result['error']}")
                return None

            # The interleaved evaluator returns combined_score = -latency_seconds
            score = result.get("combined_score", 0.0)
            latency = -score  # Convert back to positive latency

            return latency

        finally:
            if runner_fd is not None:
                os.close(runner_fd)
            for path in [runner_path, results_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def _load_cache(self, path: str) -> Dict[str, float]:
        with open(path) as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}

    def _save_cache(self, path: str, results: Dict[str, float]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def get_cached(self) -> Optional[Dict[str, float]]:
        """Return cached ground truth if available."""
        return self._cache
