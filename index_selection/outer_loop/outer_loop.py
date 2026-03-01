"""
Main orchestrator for the evaluator outer loop.

Implements the iterative cycle:
1. Propose evaluation strategy (LLM generates evaluator code)
2. Evaluate corpus with proposed evaluator
3. Measure noise (repeated evaluations)
4. Analyze discrepancies with ground truth
5. Feed analysis back to LLM
6. Repeat until convergence or max iterations
"""

import json
import os
import time
from typing import Optional

from .config import OuterLoopConfig
from .discrepancy_analyzer import DiscrepancyAnalyzer
from .evaluator_runner import EvaluatorRunner
from .ground_truth import GroundTruth
from .llm_client import GeminiClient
from .program_corpus import ProgramCorpus
from .strategy import EvaluationStrategy, StrategyHistory
from .strategy_proposer import StrategyProposer


class OuterLoop:
    """LLM-driven evaluation metric co-evolution framework."""

    def __init__(self, config: OuterLoopConfig):
        self.config = config
        self.llm = GeminiClient(config.llm)
        self.proposer = StrategyProposer(self.llm, config)
        self.runner = EvaluatorRunner(config)
        self.analyzer = DiscrepancyAnalyzer()
        self.corpus = ProgramCorpus(config.corpus_programs)
        self.ground_truth = GroundTruth(config)
        self.history = StrategyHistory()

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

    def run(
        self,
        max_iterations: Optional[int] = None,
    ) -> EvaluationStrategy:
        """Run the outer loop to discover the best evaluation metric.

        Args:
            max_iterations: Override max iterations from config.

        Returns:
            The best evaluation strategy found.
        """
        max_iter = max_iterations or self.config.max_iterations

        print("=" * 60)
        print("EVALUATOR OUTER LOOP")
        print("=" * 60)
        print(f"Benchmark: {self.config.benchmark}")
        print(f"Max iterations: {max_iter}")
        print(f"Convergence: spearman > {self.config.convergence_spearman}, "
              f"noise < {self.config.convergence_noise:.0%}")
        print(f"Corpus: {self.corpus}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 60)

        # Step 0: Measure ground truth (cached by default)
        print("\n--- Step 0: Ground Truth ---")
        gt_scores = self.ground_truth.measure(self.corpus)
        self.history.ground_truth_scores = gt_scores

        if len(gt_scores) < 2:
            print("ERROR: Need ground truth for at least 2 programs. Aborting.")
            return self.proposer.create_initial_strategy()

        print("\nGround truth ranking (lower latency = better):")
        for i, (name, lat) in enumerate(
            sorted(gt_scores.items(), key=lambda x: x[1]), 1
        ):
            print(f"  {i}. {name}: {lat:.2f}s")

        # Step 1: Start with cost-only evaluator
        strategy = self.proposer.create_initial_strategy()

        for iteration in range(max_iter):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration} — {strategy.rationale[:80]}")
            print(f"{'=' * 60}")

            # Step 2: Evaluate corpus
            if strategy.error:
                print(f"Strategy has error: {strategy.error}")
                print("Asking LLM for a new strategy...")
                self.history.add(strategy)
                strategy = self.proposer.propose(self.history)
                continue

            print("\n--- Evaluating corpus ---")
            start = time.time()
            proxy_scores = self.runner.evaluate_corpus(
                strategy.evaluator_code, self.corpus
            )
            eval_time = time.time() - start
            strategy.proxy_scores = proxy_scores
            print(f"Evaluation took {eval_time:.1f}s")

            # Check if enough evaluations succeeded (not errored)
            num_errors = len(self.runner.last_corpus_errors)
            num_ok = len(proxy_scores) - num_errors
            if num_ok < 2:
                strategy.error = f"Only {num_ok} programs evaluated successfully ({num_errors} errors)"
                print(f"ERROR: {strategy.error}")
                self.history.add(strategy)
                strategy = self.proposer.propose(self.history)
                continue

            # Step 3: Noise check
            print(f"\n--- Noise check ({self.config.noise_check_runs} runs) ---")
            noise_program = self.corpus.programs[0]
            noise = self.runner.measure_noise(
                strategy.evaluator_code, noise_program.path
            )
            strategy.noise_level = noise.std_pct
            print(f"Noise (std/mean): {noise.std_pct:.1%} "
                  f"(scores: {[f'{s:.4f}' for s in noise.scores]})")

            # Step 4: Analyze discrepancies
            print("\n--- Discrepancy analysis ---")
            report = self.analyzer.analyze(
                proxy_scores, gt_scores, noise.std_pct
            )
            strategy.ranking_agreement = report.spearman
            strategy.discrepancy_report = report.text
            print(report.text)

            # Save strategy
            self.history.add(strategy)
            self._save_iteration(iteration, strategy)

            # Step 5: Check convergence
            if (report.spearman >= self.config.convergence_spearman
                    and noise.std_pct <= self.config.convergence_noise):
                print(f"\nCONVERGED at iteration {iteration}!")
                print(f"  Spearman: {report.spearman:.4f} (threshold: {self.config.convergence_spearman})")
                print(f"  Noise: {noise.std_pct:.1%} (threshold: {self.config.convergence_noise:.0%})")
                break

            # Step 6: LLM proposes next strategy
            print("\n--- Proposing next strategy ---")
            strategy = self.proposer.propose(self.history)

        # Save final results
        best = self.history.best_strategy()
        if best:
            self._save_best(best)
            print(f"\n{'=' * 60}")
            print(f"BEST STRATEGY: v{best.version}")
            print(f"  Spearman: {best.ranking_agreement:.4f}")
            print(f"  Noise: {best.noise_level:.1%}")
            print(f"  Rationale: {best.rationale}")
            print(f"  Saved to: {self.config.output_dir}/best_evaluator.py")
            print(f"{'=' * 60}")
            return best

        return self.proposer.create_initial_strategy()

    def _save_iteration(self, iteration: int, strategy: EvaluationStrategy):
        """Save iteration results to disk."""
        iter_dir = os.path.join(self.config.output_dir, f"iteration_{iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        # Save evaluator code
        if strategy.evaluator_code:
            with open(os.path.join(iter_dir, "evaluator.py"), "w") as f:
                f.write(strategy.evaluator_code)

        # Save metadata
        meta = strategy.to_dict()
        with open(os.path.join(iter_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save full history
        self.history.save(os.path.join(self.config.output_dir, "history.json"))

    def _save_best(self, strategy: EvaluationStrategy):
        """Save the best evaluator for use with run_openevolve.py."""
        best_path = os.path.join(self.config.output_dir, "best_evaluator.py")
        with open(best_path, "w") as f:
            f.write(strategy.evaluator_code)

        # Also save metadata
        meta_path = os.path.join(self.config.output_dir, "best_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(strategy.to_dict(), f, indent=2)
