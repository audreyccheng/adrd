"""
Evaluation strategy data models and history tracking.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class NoiseResult:
    """Result of noise measurement (repeated evaluations of the same program)."""
    scores: List[float]
    mean: float = 0.0
    std: float = 0.0
    std_pct: float = 0.0  # std / mean as a fraction

    def __post_init__(self):
        if self.scores:
            self.mean = sum(self.scores) / len(self.scores)
            if len(self.scores) > 1 and self.mean != 0:
                variance = sum((s - self.mean) ** 2 for s in self.scores) / (len(self.scores) - 1)
                self.std = variance ** 0.5
                self.std_pct = abs(self.std / self.mean)


@dataclass
class EvaluationStrategy:
    """A single proposed evaluation strategy with its validation results."""
    version: int
    evaluator_code: str
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Results (filled after validation)
    proxy_scores: Dict[str, float] = field(default_factory=dict)
    ranking_agreement: float = 0.0  # Spearman correlation with ground truth
    noise_level: float = 1.0  # std/mean from repeated evaluations
    discrepancy_report: str = ""
    error: Optional[str] = None  # Set if evaluator failed

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "proxy_scores": self.proxy_scores,
            "ranking_agreement": self.ranking_agreement,
            "noise_level": self.noise_level,
            "discrepancy_report": self.discrepancy_report,
            "error": self.error,
        }

    def summary(self) -> str:
        """One-line summary for history display."""
        status = f"rho={self.ranking_agreement:.3f}, noise={self.noise_level:.3f}"
        if self.error:
            status = f"ERROR: {self.error[:80]}"
        return f"v{self.version} [{status}]: {self.rationale[:100]}"


@dataclass
class StrategyHistory:
    """Tracks all tried strategies and ground truth data."""
    strategies: List[EvaluationStrategy] = field(default_factory=list)
    ground_truth_scores: Dict[str, float] = field(default_factory=dict)

    def add(self, strategy: EvaluationStrategy):
        self.strategies.append(strategy)

    def best_strategy(self) -> Optional[EvaluationStrategy]:
        """Return strategy with highest ranking agreement (excluding errors)."""
        valid = [s for s in self.strategies if s.error is None]
        if not valid:
            return None
        return max(valid, key=lambda s: s.ranking_agreement)

    def history_summary(self) -> str:
        """Format history for LLM consumption."""
        lines = ["=== Strategy History ==="]
        for s in self.strategies:
            lines.append(s.summary())
        if self.strategies:
            best = self.best_strategy()
            if best:
                lines.append(f"\nBest so far: v{best.version} (rho={best.ranking_agreement:.3f})")
        return "\n".join(lines)

    def save(self, path: str):
        """Save history to JSON (evaluator code saved separately)."""
        data = {
            "ground_truth_scores": self.ground_truth_scores,
            "strategies": [s.to_dict() for s in self.strategies],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "StrategyHistory":
        """Load history from JSON (evaluator code NOT restored)."""
        with open(path) as f:
            data = json.load(f)
        history = cls()
        history.ground_truth_scores = data.get("ground_truth_scores", {})
        for sd in data.get("strategies", []):
            strategy = EvaluationStrategy(
                version=sd["version"],
                evaluator_code="",  # Not saved in JSON
                rationale=sd["rationale"],
                timestamp=sd.get("timestamp", ""),
            )
            strategy.proxy_scores = sd.get("proxy_scores", {})
            strategy.ranking_agreement = sd.get("ranking_agreement", 0.0)
            strategy.noise_level = sd.get("noise_level", 1.0)
            strategy.discrepancy_report = sd.get("discrepancy_report", "")
            strategy.error = sd.get("error")
            history.add(strategy)
        return history
