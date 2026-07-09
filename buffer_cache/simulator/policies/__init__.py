"""
Buffer replacement policies.

Includes:
- clock_sweep: PostgreSQL baseline (for comparison)
- pbm_pq: PBM Mode 2 (priority queue based)
- pbm_sampling: PBM Mode 3 (sampling based - recommended)
"""
from .clock_sweep import clock_sweep_policy
from .pbm_pq import pbm_pq_policy
from .pbm_sampling import pbm_sampling_policy

__all__ = [
    'clock_sweep_policy',
    'pbm_pq_policy',
    'pbm_sampling_policy',
]

