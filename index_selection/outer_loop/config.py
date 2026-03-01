"""
Configuration for the evaluator outer loop.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMConfig:
    api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-2.5-pro"
    temperature: float = 0.7
    max_tokens: int = 8192


@dataclass
class GroundTruthConfig:
    evaluator: str = "evaluator_latency_interleaved.py"
    num_runs: int = 3
    prewarm: bool = True
    suppress_bg: bool = True
    cache_file: str = "ground_truth_cache.json"


@dataclass
class OuterLoopConfig:
    max_iterations: int = 10
    benchmark: str = "tpch"
    convergence_spearman: float = 0.9
    convergence_noise: float = 0.05
    noise_check_runs: int = 3
    output_dir: str = "outer_loop_outputs"
    eval_timeout: int = 600  # Per-program evaluation timeout (seconds)

    llm: LLMConfig = field(default_factory=LLMConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    corpus_programs: List[str] = field(default_factory=lambda: [
        "initial_programs/initial_program_autoadmin.py",
        "initial_programs/initial_program_extend.py",
        "initial_programs/initial_program_db2advis.py",
        "initial_programs/initial_program_anytime.py",
        "initial_programs/best_explore_extend_1215.py",
        "initial_programs/best_tpch_v3_extend_evolved.py",
    ])

    @classmethod
    def from_yaml(cls, path: str) -> "OuterLoopConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Top-level outer_loop settings
        ol = data.get("outer_loop", {})
        for key in ["max_iterations", "benchmark", "convergence_spearman",
                     "convergence_noise", "noise_check_runs", "output_dir",
                     "eval_timeout"]:
            if key in ol:
                setattr(config, key, ol[key])

        # LLM settings
        llm_data = data.get("llm", {})
        for key in ["api_base", "api_key_env", "model", "temperature", "max_tokens"]:
            if key in llm_data:
                setattr(config.llm, key, llm_data[key])

        # Ground truth settings
        gt_data = data.get("ground_truth", {})
        for key in ["evaluator", "num_runs", "prewarm", "suppress_bg", "cache_file"]:
            if key in gt_data:
                setattr(config.ground_truth, key, gt_data[key])

        # Corpus programs
        corpus = data.get("corpus", {})
        if "programs" in corpus:
            config.corpus_programs = corpus["programs"]

        return config

    def resolve_paths(self, base_dir: str):
        """Resolve relative paths against base directory."""
        self.output_dir = os.path.join(base_dir, self.output_dir)
        self.corpus_programs = [
            os.path.join(base_dir, p) if not os.path.isabs(p) else p
            for p in self.corpus_programs
        ]
        if not os.path.isabs(self.ground_truth.cache_file):
            self.ground_truth.cache_file = os.path.join(
                self.output_dir, self.ground_truth.cache_file
            )
        if not os.path.isabs(self.ground_truth.evaluator):
            self.ground_truth.evaluator = os.path.join(
                base_dir, self.ground_truth.evaluator
            )
