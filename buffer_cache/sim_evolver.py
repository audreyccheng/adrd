"""
SimEvolver — Outer loop orchestrator for two-level simulator evolution.

Maintains a population of SimulatorConfigs, evolves them through:
1. Mutate top configs → new variants
2. Run inner OpenEvolve for each variant (policy evolution)
3. Translate best policy to C via LLM
4. Benchmark on real PostgreSQL
5. Rank by real PG performance → select for next generation

The outer loop is lightweight and custom (not OpenEvolve).
The inner loop uses standard OpenEvolve.
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .evaluator_generator import generate_evaluator, generate_openevolve_config
    from .mutations import mutate, mutate_llm, generate_ablation_variants
    from .pg_benchmarker import PGBenchmarker
    from .policy_translator import PolicyTranslator
    from .results import BenchmarkResult, InnerResult, OuterResult, ResultTracker
    from .simulator_config import SimulatorConfig, PRESET_CONFIGS
except ImportError:
    from evaluator_generator import generate_evaluator, generate_openevolve_config
    from mutations import mutate, mutate_llm, generate_ablation_variants
    from pg_benchmarker import PGBenchmarker
    from policy_translator import PolicyTranslator
    from results import BenchmarkResult, InnerResult, OuterResult, ResultTracker
    from simulator_config import SimulatorConfig, PRESET_CONFIGS

logger = logging.getLogger(__name__)


class SimEvolver:
    """
    Outer loop: evolves simulator configurations to maximize
    the quality of policies discovered by the inner loop.

    Quality is measured by real PostgreSQL benchmark performance.
    """

    def __init__(
        self,
        output_dir: str,
        population_size: int = 5,
        seed_configs: Optional[List[SimulatorConfig]] = None,
        inner_iterations: int = 50,
        translation_threshold: float = 0.01,
        benchmark_top_k: int = 3,
        random_seed: int = 42,
        skip_benchmark: bool = False,
        skip_translation: bool = False,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Args:
            output_dir: Directory for all outputs
            population_size: Number of configs in population
            seed_configs: Initial population (defaults to V5 + random mutations)
            inner_iterations: OpenEvolve iterations per inner loop
            translation_threshold: Only translate if score improves by this fraction
            benchmark_top_k: Benchmark top K configs per generation
            random_seed: For reproducibility
            skip_benchmark: Skip real PG benchmarks (use sim score only)
            skip_translation: Skip Python-to-C translation
            llm_model: Model for outer loop LLM-guided mutation (default: gpt-4o-mini)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.population_size = population_size
        self.inner_iterations = inner_iterations
        self.translation_threshold = translation_threshold
        self.benchmark_top_k = benchmark_top_k
        self.skip_benchmark = skip_benchmark
        self.skip_translation = skip_translation
        self.llm_model = llm_model

        self.rng = random.Random(random_seed)
        self.tracker = ResultTracker(str(self.output_dir))
        self.translator = PolicyTranslator()
        self.benchmarker = PGBenchmarker()

        # Initialize population
        if seed_configs:
            self.population = list(seed_configs)
        else:
            self.population = self._default_population()

        # Track best score for translation threshold
        self.best_sim_score = 0.0
        self.best_pg_score = 0.0
        self.generation = 0

        logger.info(f"SimEvolver initialized: pop={population_size}, "
                     f"inner_iter={inner_iterations}, output={output_dir}")

    def _default_population(self) -> List[SimulatorConfig]:
        """Create default population: V5 baseline + random mutations."""
        from .simulator_config import v5_config
        base = v5_config()
        base = base.clone(inner_iterations=self.inner_iterations)

        population = [base]
        for i in range(self.population_size - 1):
            variant = mutate(base, self.rng)
            variant = variant.clone(
                name=f"seed_variant_{i}",
                inner_iterations=self.inner_iterations,
            )
            population.append(variant)

        return population

    def run(self, num_generations: int = 10) -> OuterResult:
        """
        Run the outer evolution loop.

        Args:
            num_generations: Number of outer loop generations

        Returns:
            The best OuterResult found
        """
        logger.info(f"Starting outer loop: {num_generations} generations, "
                     f"{self.population_size} configs per generation")

        for gen in range(num_generations):
            self.generation = gen
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {gen}")
            logger.info(f"{'='*60}")

            # 1. Evaluate each config in the population
            gen_results: List[OuterResult] = []

            for i, config in enumerate(self.population):
                logger.info(f"\n--- Config {i+1}/{len(self.population)}: {config.name} ---")
                logger.info(f"    {config.summary()}")

                result = self._evaluate_config(config, gen)
                gen_results.append(result)
                self.tracker.record(result)

                logger.info(f"    Sim score: {result.simulator_score:.4f}")
                if result.benchmark_result:
                    logger.info(f"    PG score:  {result.real_pg_score:.4f} "
                                 f"(hit_rate={result.benchmark_result.hit_rate:.2%}, "
                                 f"tps={result.benchmark_result.throughput:.4f})")

            # 2. Select top configs
            ranked = sorted(gen_results, key=lambda r: r.real_pg_score if r.benchmark_result else r.simulator_score, reverse=True)

            logger.info(f"\n--- Generation {gen} Rankings ---")
            for i, r in enumerate(ranked):
                score = r.real_pg_score if r.benchmark_result else r.simulator_score
                logger.info(f"  {i+1}. {r.config_name}: {score:.4f}")

            # 3. Select survivors and mutate for next generation
            self.population = self._select_and_mutate(ranked)

            # 4. Save progress
            self.tracker.save_summary()
            self._save_population(gen)

            logger.info(f"\nGeneration {gen} complete. Best: {ranked[0].config_name} "
                         f"(score={ranked[0].real_pg_score if ranked[0].benchmark_result else ranked[0].simulator_score:.4f})")

        # Return best overall
        best = self.tracker.best_config()
        if best is None:
            # Fall back to best by simulator score
            all_results = sorted(self.tracker.results, key=lambda r: r.simulator_score, reverse=True)
            best = all_results[0] if all_results else None

        logger.info(f"\n{'='*60}")
        logger.info(f"EVOLUTION COMPLETE")
        logger.info(f"{'='*60}")
        if best:
            logger.info(f"Best config: {best.config_name}")
            logger.info(f"Best sim score: {best.simulator_score:.4f}")
            if best.benchmark_result:
                logger.info(f"Best PG score: {best.real_pg_score:.4f}")
        logger.info(f"\n{self.tracker.ranking_table()}")

        return best

    def _evaluate_config(self, config: SimulatorConfig, generation: int) -> OuterResult:
        """Evaluate a single SimulatorConfig through the full pipeline."""
        config_id = config.config_id()
        config_dir = self.output_dir / f"gen{generation}" / config.name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(str(config_dir / "config.yaml"))

        # Step 1: Run inner loop (OpenEvolve)
        inner_result = self._run_inner_loop(config, str(config_dir / "inner"))

        # Step 2: Translate to C (if improvement threshold met)
        translation_success = False
        c_code = ""
        compile_success = False

        if not self.skip_translation and inner_result.simulator_score > self.best_sim_score * (1 + self.translation_threshold):
            c_code_result = self.translator.translate(inner_result.best_policy_path)
            if c_code_result:
                translation_success = True
                c_code = c_code_result
                self.translator.integrate(c_code, config.name)
                compile_success = self.translator.compile()
                self.translator.save_translation(str(config_dir), config.name)

        # Step 3: Benchmark on real PostgreSQL
        benchmark_result = None
        real_pg_score = 0.0

        if not self.skip_benchmark and compile_success:
            benchmark_result = self.benchmarker.benchmark_tpch()
            real_pg_score = benchmark_result.throughput  # Use throughput as primary score
            if real_pg_score > self.best_pg_score:
                self.best_pg_score = real_pg_score

        # Update best sim score
        if inner_result.simulator_score > self.best_sim_score:
            self.best_sim_score = inner_result.simulator_score

        return OuterResult(
            generation=generation,
            config_id=config_id,
            config_name=config.name,
            config_dict=asdict(config),
            inner_result=inner_result,
            translation_success=translation_success,
            translated_c_code=c_code,
            compile_success=compile_success,
            benchmark_result=benchmark_result,
            simulator_score=inner_result.simulator_score,
            real_pg_score=real_pg_score,
            fidelity_gap=abs(inner_result.simulator_score - real_pg_score),
        )

    def _run_inner_loop(self, config: SimulatorConfig, output_dir: str) -> InnerResult:
        """Run OpenEvolve inner loop for a given simulator config."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate evaluator and initial program
        evaluator_path = generate_evaluator(config, str(output_path))
        oe_config_path = generate_openevolve_config(config, str(output_path))
        initial_program_path = str(output_path / "initial_program.py")

        # Find openevolve-run.py
        openevolve_run = os.environ.get(
            "SIMEVOLVER_OPENEVOLVE_RUN",
            str(Path(__file__).parent / "openevolve" / "openevolve-run.py"),
        )
        if not os.path.exists(openevolve_run):
            logger.warning(f"openevolve-run.py not found at {openevolve_run}")
            # Fall back to direct evaluator test
            return self._run_direct_evaluation(config, evaluator_path, initial_program_path, output_dir)

        logger.info(f"  Running OpenEvolve ({config.inner_iterations} iterations)...")
        start_time = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable, openevolve_run,
                    initial_program_path,
                    evaluator_path,
                    "--config", oe_config_path,
                    "--iterations", str(config.inner_iterations),
                ],
                capture_output=True,
                text=True,
                timeout=config.inner_iterations * 120,  # ~2 min per iteration max
                cwd=str(output_path),
            )

            runtime = time.time() - start_time

            if result.returncode != 0:
                logger.warning(f"  OpenEvolve failed: {result.stderr[-300:]}")
                return self._run_direct_evaluation(config, evaluator_path, initial_program_path, output_dir)

            # Find best program
            best_program_path = self._find_best_program(str(output_path))
            best_score = self._extract_best_score(str(output_path))

            return InnerResult(
                config_id=config.config_id(),
                config_name=config.name,
                simulator_score=best_score,
                best_policy_path=best_program_path,
                best_policy_code=open(best_program_path).read() if best_program_path else "",
                iterations_run=config.inner_iterations,
                runtime_seconds=runtime,
            )

        except subprocess.TimeoutExpired:
            logger.warning("  OpenEvolve timed out")
            return self._run_direct_evaluation(config, evaluator_path, initial_program_path, output_dir)
        except Exception as e:
            logger.warning(f"  OpenEvolve error: {e}")
            return self._run_direct_evaluation(config, evaluator_path, initial_program_path, output_dir)

    def _run_direct_evaluation(
        self, config: SimulatorConfig, evaluator_path: str,
        initial_program_path: str, output_dir: str
    ) -> InnerResult:
        """
        Fallback: directly evaluate the initial program without OpenEvolve.
        Useful for testing the framework without a full evolution run.
        """
        logger.info("  Running direct evaluation (no OpenEvolve)...")

        try:
            result = subprocess.run(
                [sys.executable, evaluator_path, initial_program_path],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Parse JSON from output — may be multiline and mixed with other text
                stdout = result.stdout
                brace_start = stdout.find("{")
                if brace_start >= 0:
                    depth = 0
                    for i in range(brace_start, len(stdout)):
                        if stdout[i] == "{":
                            depth += 1
                        elif stdout[i] == "}":
                            depth -= 1
                        if depth == 0:
                            try:
                                data = json.loads(stdout[brace_start:i + 1])
                                if "metrics" in data:
                                    score = data["metrics"].get("combined_score", 0.0)
                                    return InnerResult(
                                        config_id=config.config_id(),
                                        config_name=config.name,
                                        simulator_score=score,
                                        best_policy_path=initial_program_path,
                                        best_policy_code=open(initial_program_path).read(),
                                        iterations_run=0,
                                        runtime_seconds=0.0,
                                    )
                            except json.JSONDecodeError:
                                pass
                            break

        except Exception as e:
            logger.warning(f"  Direct evaluation failed: {e}")

        return InnerResult(
            config_id=config.config_id(),
            config_name=config.name,
            simulator_score=0.0,
            best_policy_path=initial_program_path,
            iterations_run=0,
        )

    def _find_best_program(self, output_dir: str) -> str:
        """Find the best evolved program from OpenEvolve output."""
        # OpenEvolve saves best program to output_dir/best/best_program.py
        best_path = os.path.join(output_dir, "best", "best_program.py")
        if os.path.exists(best_path):
            return best_path

        # Also check checkpoints
        checkpoint_dirs = sorted(
            Path(output_dir).glob("checkpoints/checkpoint_*/best_program.py"),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        if checkpoint_dirs:
            return str(checkpoint_dirs[0])

        # Fallback to initial program
        initial = os.path.join(output_dir, "initial_program.py")
        return initial if os.path.exists(initial) else ""

    def _extract_best_score(self, output_dir: str) -> float:
        """Extract the best combined_score from OpenEvolve output."""
        # Check best program metadata
        best_meta = os.path.join(output_dir, "best", "best_metrics.json")
        if os.path.exists(best_meta):
            with open(best_meta) as f:
                data = json.load(f)
                return float(data.get("combined_score", 0.0))

        # Check checkpoints
        checkpoint_dirs = sorted(
            Path(output_dir).glob("checkpoints/checkpoint_*/"),
            key=lambda p: p.name,
            reverse=True,
        )
        for cp_dir in checkpoint_dirs:
            meta_file = cp_dir / "best_metrics.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    data = json.load(f)
                    return float(data.get("combined_score", 0.0))

        return 0.0

    def _select_and_mutate(self, ranked: List[OuterResult]) -> List[SimulatorConfig]:
        """Select top configs and mutate to fill the next generation.

        Uses LLM-guided mutation by default: the LLM sees the generation's
        scores and config history, and proposes targeted config changes.
        Falls back to random mutation if no API key is available.
        """
        # Keep top 2 (elitism)
        survivors = min(2, len(ranked))
        new_population = []

        for i in range(survivors):
            config = SimulatorConfig(**ranked[i].config_dict)
            config = config.clone(name=f"elite_{i}_{config.name}")
            new_population.append(config)

        # Build generation history for LLM context
        gen_history = []
        for r in self.tracker.results:
            # Build per-workload breakdown string
            breakdown = ""
            if r.inner_result and r.inner_result.per_workload_scores:
                parts = ["%s=%.4f" % (k, v) for k, v in r.inner_result.per_workload_scores.items()]
                breakdown = ", ".join(parts)
            gen_history.append((r.config_name, r.simulator_score, breakdown))

        # Fill rest with LLM-guided mutations (default) or random (fallback)
        while len(new_population) < self.population_size:
            # Tournament selection from top half
            parent_idx = self.rng.randint(0, max(0, len(ranked) // 2 - 1))
            parent_config = SimulatorConfig(**ranked[parent_idx].config_dict)

            child = mutate_llm(
                parent_config,
                generation_results=gen_history,
                model=self.llm_model,
            )
            child = child.clone(
                name=f"gen{self.generation + 1}_child_{len(new_population)}",
                inner_iterations=self.inner_iterations,
            )
            new_population.append(child)

        return new_population

    def _save_population(self, generation: int):
        """Save current population configs to disk."""
        pop_dir = self.output_dir / f"gen{generation}" / "population"
        pop_dir.mkdir(parents=True, exist_ok=True)

        for i, config in enumerate(self.population):
            config.to_yaml(str(pop_dir / f"{config.name}.yaml"))

    def run_ablation(self, base_config: Optional[SimulatorConfig] = None) -> List[OuterResult]:
        """
        Run ablation study: evaluate V5 baseline and all single-feature ablations.

        Args:
            base_config: Base config to ablate (defaults to V5)

        Returns:
            List of OuterResults for base + all ablations
        """
        if base_config is None:
            from .simulator_config import v5_config
            base_config = v5_config()

        variants = [base_config] + generate_ablation_variants(base_config)
        results = []

        logger.info(f"Running ablation study: {len(variants)} variants")

        for i, config in enumerate(variants):
            logger.info(f"\n--- Ablation {i+1}/{len(variants)}: {config.name} ---")
            result = self._evaluate_config(config, generation=0)
            results.append(result)
            self.tracker.record(result)
            logger.info(f"    Score: {result.simulator_score:.4f}")

        # Print ablation table
        logger.info("\n" + "="*60)
        logger.info("ABLATION RESULTS")
        logger.info("="*60)
        base_score = results[0].simulator_score if results else 0.0
        for r in results:
            delta = r.simulator_score - base_score
            sign = "+" if delta >= 0 else ""
            logger.info(f"  {r.config_name:<35} score={r.simulator_score:.4f} ({sign}{delta:.4f})")

        self.tracker.save_summary()
        return results


def main():
    """CLI entry point for SimEvolver."""
    import argparse

    parser = argparse.ArgumentParser(description="Two-level simulator evolution framework")
    parser.add_argument("--output", default="sim_evolver_output",
                        help="Output directory")
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of outer loop generations")
    parser.add_argument("--population", type=int, default=5,
                        help="Population size")
    parser.add_argument("--inner-iterations", type=int, default=50,
                        help="OpenEvolve iterations per inner loop")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip real PostgreSQL benchmarks")
    parser.add_argument("--skip-translation", action="store_true",
                        help="Skip Python-to-C translation")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study instead of evolution")
    parser.add_argument("--config", type=str, default=None,
                        help="Seed config preset name (e.g., v1, v3, v5)")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="Model for outer loop LLM-guided mutation")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Build seed configs
    seed_configs = None
    if args.config:
        if args.config in PRESET_CONFIGS:
            seed_configs = [PRESET_CONFIGS[args.config]()]
        else:
            seed_configs = [SimulatorConfig.from_yaml(args.config)]

    evolver = SimEvolver(
        output_dir=args.output,
        population_size=args.population,
        seed_configs=seed_configs,
        inner_iterations=args.inner_iterations,
        random_seed=args.seed,
        skip_benchmark=args.skip_benchmark,
        skip_translation=args.skip_translation,
        llm_model=args.llm_model,
    )

    if args.ablation:
        evolver.run_ablation()
    else:
        evolver.run(num_generations=args.generations)


if __name__ == "__main__":
    main()
