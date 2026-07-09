"""
PBM Sampling Policy - Mode 3 reference implementation.

This is the main contribution of the PBM thesis:
- Sample N random buffers
- Compute next-access estimates ON-DEMAND (fresh data!)
- Evict buffer with LARGEST next-access time

Key advantages over Mode 2 (Priority Queue):
- Fresh estimates (not stale)
- No global lock
- Simpler implementation
- Better performance (30% more I/O reduction)
"""
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class SampleResult:
    """Result of sampling a buffer."""
    buf_id: int
    next_access_time: float
    is_requested: bool


def pbm_sampling_policy(
    buffers: List,
    buffer_table: Dict,
    estimator: Any,
    scan_context: Any,
    num_samples: int = 7,  # Optimized: N=7 is sweet spot (better than 10)
) -> Optional[int]:
    """
    PBM Mode 3: Sampling-based eviction.
    
    Algorithm:
    1. Sample N random unpinned buffers
    2. For each sample, compute next-access-time estimate
    3. Return buffer with LARGEST next-access time
    
    This mimics Belady's MIN algorithm by evicting the buffer
    that won't be needed for the longest time.
    
    Args:
        buffers: List of BufferDescriptor
        buffer_table: tag -> buf_id mapping
        estimator: NextAccessEstimator for computing estimates
        scan_context: Current scan info (for context)
        num_samples: Number of buffers to sample (default 10)
    
    Returns:
        buf_id to evict, or None if failed
    """
    if estimator is None:
        return None
    
    num_buffers = len(buffers)
    if num_buffers == 0:
        return None
    
    # Optimization: cache random.random for faster calls
    _random = random.random
    
    # Fast path: find empty buffer
    for _ in range(3):
        buf_id = int(_random() * num_buffers)  # Faster than randint
        buf = buffers[buf_id]
        if buf.refcount == 0 and buf.tag is None:
            return buf_id
    
    # Optimization: use list of tuples instead of dataclass (faster)
    samples: List[Tuple[int, float, bool]] = []  # (buf_id, next_access, is_requested)
    seen: set = set()
    attempts = 0
    max_attempts = num_samples * 3
    
    # Optimization: cache estimator method
    _estimate = estimator.estimate_for_buffer
    
    # Step 1: Sample N random unpinned buffers
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Pick random buffer (faster than randint)
        buf_id = int(_random() * num_buffers)
        
        # Skip if already sampled (O(1) check)
        if buf_id in seen:
            continue
        seen.add(buf_id)
        
        buf = buffers[buf_id]
        
        # Optimization: inline attribute access
        if buf.refcount > 0:
            continue
        
        if buf.tag is None:
            return buf_id
        
        # Compute next-access estimate ON-DEMAND
        next_access, is_requested = _estimate(buf)
        
        samples.append((buf_id, next_access, is_requested))
    
    if not samples:
        return None
    
    # Step 2: Find buffer with largest next-access time (O(n) vs O(n log n) sort)
    # samples is list of (buf_id, next_access, is_requested)
    best_buf_id, best_next_access, _ = max(samples, key=lambda s: s[1])
    
    # Double-check still evictable
    if buffers[best_buf_id].refcount == 0:
        return best_buf_id
    
    # Fallback: find any evictable from samples
    for buf_id, _, _ in samples:
        if buffers[buf_id].refcount == 0:
            return buf_id
    
    return None


def create_pbm_sampling_policy(num_samples: int = 10):
    """
    Factory function to create PBM sampling policy with custom sample size.
    
    Args:
        num_samples: Number of buffers to sample per eviction
    
    Returns:
        Policy function with configured sample size
    """
    def policy(buffers, buffer_table, estimator, scan_context):
        return pbm_sampling_policy(
            buffers, buffer_table, estimator, scan_context,
            num_samples=num_samples
        )
    return policy


# Default policy with N=10 (from thesis)
default_pbm_sampling = create_pbm_sampling_policy(10)

# Bulk sampling: sample 100, for use with bulk eviction
bulk_pbm_sampling = create_pbm_sampling_policy(100)

