"""
Discrepancy analysis: compare proxy metric rankings with ground truth.

Computes Spearman correlation, per-program ranking errors, pairwise
direction agreement, and generates structured reports for LLM consumption.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DiscrepancyReport:
    """Result of comparing proxy rankings with ground truth."""
    spearman: float  # Spearman rank correlation coefficient
    pairwise_agreement: float  # Fraction of program pairs ranked correctly
    per_program_errors: List[str]  # Human-readable per-program analysis
    text: str  # Full report formatted for LLM consumption


class DiscrepancyAnalyzer:
    """Analyze discrepancies between proxy metric and ground truth rankings."""

    def analyze(
        self,
        proxy_scores: Dict[str, float],
        ground_truth_latencies: Dict[str, float],
        noise_std_pct: float = 0.0,
    ) -> DiscrepancyReport:
        """Compare proxy scores with ground truth latencies.

        Args:
            proxy_scores: program_name → combined_score (higher = better).
            ground_truth_latencies: program_name → latency_seconds (lower = better).
            noise_std_pct: Noise level (std/mean) from repeated evaluations.

        Returns:
            DiscrepancyReport with analysis.
        """
        # Find programs present in both
        common = sorted(set(proxy_scores.keys()) & set(ground_truth_latencies.keys()))
        if len(common) < 2:
            return DiscrepancyReport(
                spearman=0.0,
                pairwise_agreement=0.0,
                per_program_errors=["Insufficient programs for comparison"],
                text="ERROR: Need at least 2 programs in both proxy and ground truth.",
            )

        # Rank programs by proxy (higher score = better = rank 1)
        proxy_ranked = sorted(common, key=lambda p: proxy_scores[p], reverse=True)
        proxy_ranks = {p: i + 1 for i, p in enumerate(proxy_ranked)}

        # Rank programs by ground truth (lower latency = better = rank 1)
        gt_ranked = sorted(common, key=lambda p: ground_truth_latencies[p])
        gt_ranks = {p: i + 1 for i, p in enumerate(gt_ranked)}

        # Spearman rank correlation
        n = len(common)
        spearman = self._spearman(proxy_ranks, gt_ranks, common)

        # Pairwise direction agreement
        agree, total = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = common[i], common[j]
                proxy_better = proxy_scores[pi] > proxy_scores[pj]
                gt_better = ground_truth_latencies[pi] < ground_truth_latencies[pj]
                if proxy_better == gt_better:
                    agree += 1
                total += 1
        pairwise_agreement = agree / total if total > 0 else 0.0

        # Per-program analysis
        per_program = []
        for p in common:
            pr = proxy_ranks[p]
            gr = gt_ranks[p]
            delta = pr - gr
            score = proxy_scores[p]
            latency = ground_truth_latencies[p]
            direction = "OVER-RANKED" if delta < 0 else "UNDER-RANKED" if delta > 0 else "CORRECT"
            per_program.append(
                f"{p}: proxy_rank={pr}, gt_rank={gr}, delta={delta:+d} ({direction}), "
                f"proxy_score={score:.4f}, gt_latency={latency:.2f}s"
            )

        # Generate text report
        text = self._format_report(
            spearman, pairwise_agreement, per_program, noise_std_pct,
            proxy_ranked, gt_ranked, proxy_scores, ground_truth_latencies,
        )

        return DiscrepancyReport(
            spearman=spearman,
            pairwise_agreement=pairwise_agreement,
            per_program_errors=per_program,
            text=text,
        )

    def _spearman(
        self,
        ranks_a: Dict[str, int],
        ranks_b: Dict[str, int],
        keys: List[str],
    ) -> float:
        """Compute Spearman rank correlation coefficient."""
        n = len(keys)
        if n < 2:
            return 0.0

        d_squared = sum((ranks_a[k] - ranks_b[k]) ** 2 for k in keys)
        # Spearman formula: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        denom = n * (n * n - 1)
        if denom == 0:
            return 0.0
        return 1.0 - (6.0 * d_squared) / denom

    def _format_report(
        self,
        spearman: float,
        pairwise_agreement: float,
        per_program: List[str],
        noise_std_pct: float,
        proxy_ranked: List[str],
        gt_ranked: List[str],
        proxy_scores: Dict[str, float],
        gt_latencies: Dict[str, float],
    ) -> str:
        """Format a structured report for LLM consumption."""
        lines = [
            "=== DISCREPANCY ANALYSIS ===",
            "",
            f"Spearman rank correlation: {spearman:.4f}",
            f"  (1.0 = perfect agreement, 0.0 = no correlation, -1.0 = inverse)",
            f"Pairwise direction agreement: {pairwise_agreement:.1%}",
            f"  (fraction of program pairs where proxy agrees which is better)",
            f"Measurement noise (std/mean): {noise_std_pct:.1%}",
            "",
            "--- Proxy ranking (higher score = better) ---",
        ]

        for i, p in enumerate(proxy_ranked, 1):
            lines.append(f"  {i}. {p} (score={proxy_scores[p]:.4f})")

        lines.append("")
        lines.append("--- Ground truth ranking (lower latency = better) ---")

        for i, p in enumerate(gt_ranked, 1):
            lines.append(f"  {i}. {p} (latency={gt_latencies[p]:.2f}s)")

        lines.append("")
        lines.append("--- Per-program analysis ---")
        lines.extend(f"  {pe}" for pe in per_program)

        # Identify biggest disagreements
        lines.append("")
        lines.append("--- Key disagreements ---")
        disagreements = []
        for p in proxy_ranked:
            if p in gt_latencies:
                pr_idx = proxy_ranked.index(p)
                gt_idx = gt_ranked.index(p)
                if abs(pr_idx - gt_idx) >= 2:
                    disagreements.append(
                        f"  {p}: proxy ranks #{pr_idx+1} but ground truth ranks #{gt_idx+1} "
                        f"(off by {abs(pr_idx - gt_idx)} positions)"
                    )
        if disagreements:
            lines.extend(disagreements)
        else:
            lines.append("  No major disagreements (all within 1 rank position)")

        # Diagnosis hints
        lines.append("")
        lines.append("--- Diagnosis ---")
        if noise_std_pct > 0.10:
            lines.append(f"  HIGH NOISE: {noise_std_pct:.1%} variance between repeated runs.")
            lines.append("  This means the evaluator cannot reliably distinguish programs.")
            lines.append("  Consider: multiple runs per query, cache warmup, background suppression.")
        if spearman < 0.3:
            lines.append(f"  LOW CORRELATION: Spearman={spearman:.3f}")
            lines.append("  The proxy metric does not predict actual performance.")
            lines.append("  Consider: using actual query latency instead of cost estimates,")
            lines.append("  or adjusting the metric to account for I/O, cache effects.")
        if spearman < 0 and noise_std_pct < 0.10:
            lines.append(f"  INVERSE CORRELATION: Spearman={spearman:.3f}")
            lines.append("  The proxy metric is ANTI-correlated with actual performance!")
            lines.append("  Programs that score well are actually WORSE. Metric is misleading.")

        return "\n".join(lines)
