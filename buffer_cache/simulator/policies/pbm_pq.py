"""
PBM Priority Queue Policy - Mode 2 implementation.

This is the original PBM approach from Świtakowski et al.:
- Approximate priority queue with exponentially-sized buckets
- Block groups placed in buckets by estimated next-access time
- Evict from bucket with furthest-future access

Characteristics:
- Pre-computed estimates (can become stale)
- Global PQ lock (concurrency bottleneck)
- Bulk eviction from buckets

Mode 3 (sampling) is generally better because:
- Fresh on-demand estimates
- No global lock
- Simpler implementation
"""
import math
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# Configuration
NUM_BUCKETS = 32  # Number of time buckets
BASE_TIMESLICE = 0.1  # Base time range (100ms)


@dataclass
class PQBucket:
    """A bucket in the approximate priority queue."""
    bucket_id: int
    block_groups: set = field(default_factory=set)  # Set of (rel_id, group_num)
    
    @property
    def time_range_start(self) -> float:
        """Start of time range for this bucket."""
        if self.bucket_id == 0:
            return 0.0
        return BASE_TIMESLICE * (2 ** (self.bucket_id - 1))
    
    @property
    def time_range_end(self) -> float:
        """End of time range for this bucket."""
        return BASE_TIMESLICE * (2 ** self.bucket_id)


class ApproximatePriorityQueue:
    """
    Approximate priority queue with exponentially-sized buckets.
    
    Bucket time ranges:
    - Bucket 0: [0, Δ)         where Δ = 100ms
    - Bucket 1: [Δ, 2Δ)
    - Bucket 2: [2Δ, 4Δ)
    - Bucket N: [2^(N-1)·Δ, 2^N·Δ)
    
    Also has a "not_requested" bucket for blocks with no scans.
    """
    
    def __init__(self):
        self.buckets = [PQBucket(i) for i in range(NUM_BUCKETS)]
        self.not_requested = PQBucket(-1)  # Special bucket
        
        # Track which bucket each block group is in
        self.group_to_bucket: Dict[tuple, int] = {}  # (rel, group) -> bucket_id
        
        # Current time baseline
        self.current_time = 0.0
    
    def _compute_bucket(self, next_access_time: float) -> int:
        """Compute which bucket a next-access-time belongs to."""
        if next_access_time == float('inf'):
            return -1  # Not requested
        
        delta = next_access_time - self.current_time
        if delta <= 0:
            return 0  # Imminent access
        
        # bucket = floor(log2(delta / BASE_TIMESLICE)) + 1
        bucket = int(math.log2(delta / BASE_TIMESLICE)) + 1
        return min(bucket, NUM_BUCKETS - 1)
    
    def insert_or_update(self, rel_id: int, group_num: int, next_access_time: float):
        """Insert or update a block group's position in the PQ."""
        key = (rel_id, group_num)
        target_bucket = self._compute_bucket(next_access_time)
        
        # Remove from old bucket if present
        old_bucket = self.group_to_bucket.get(key)
        if old_bucket is not None:
            if old_bucket == -1:
                self.not_requested.block_groups.discard(key)
            else:
                self.buckets[old_bucket].block_groups.discard(key)
        
        # Add to new bucket
        if target_bucket == -1:
            self.not_requested.block_groups.add(key)
        else:
            self.buckets[target_bucket].block_groups.add(key)
        
        self.group_to_bucket[key] = target_bucket
    
    def remove(self, rel_id: int, group_num: int):
        """Remove a block group from the PQ."""
        key = (rel_id, group_num)
        bucket_id = self.group_to_bucket.pop(key, None)
        if bucket_id is not None:
            if bucket_id == -1:
                self.not_requested.block_groups.discard(key)
            else:
                self.buckets[bucket_id].block_groups.discard(key)
    
    def get_eviction_candidates(self) -> set:
        """
        Get block groups from furthest-future bucket.
        
        Returns set of (rel_id, group_num) tuples (avoids list() allocation).
        """
        # First try "not requested" bucket
        if self.not_requested.block_groups:
            return self.not_requested.block_groups  # Return set directly
        
        # Then try regular buckets, from furthest to nearest
        for i in range(NUM_BUCKETS - 1, -1, -1):
            if self.buckets[i].block_groups:
                return self.buckets[i].block_groups  # Return set directly
        
        return set()
    
    def update_time(self, new_time: float):
        """Update current time baseline."""
        self.current_time = new_time


# Global PQ instance (simulates shared state)
_pq = ApproximatePriorityQueue()
_last_refresh_time = 0.0
_refresh_interval = 0.01  # Refresh PQ every 10ms of simulated time


def pbm_pq_policy(
    buffers: List,
    buffer_table: Dict,
    estimator: Any,
    scan_context: Any,
) -> Optional[int]:
    """
    PBM Mode 2: Priority Queue based eviction.
    
    Algorithm:
    1. Periodically refresh block group estimates (not every call!)
    2. Find non-empty bucket with furthest-future access
    3. Sample buffers to find one from target bucket
    
    Optimized: Don't iterate all buffers - sample instead.
    """
    global _pq, _last_refresh_time
    
    if estimator is None:
        return None
    
    num_buffers = len(buffers)
    if num_buffers == 0:
        return None
    
    # Optimization: cache random.random
    _random = random.random
    
    current_time = estimator.scans.get_time()
    _pq.update_time(current_time)
    
    # Refresh PQ periodically (do this BEFORE fast path so PQ stays updated)
    if current_time - _last_refresh_time >= _refresh_interval:
        _last_refresh_time = current_time
        # Cache method lookups
        _estimate = estimator.estimate_for_block_group
        _insert = _pq.insert_or_update
        
        # Sample a subset of buffers to update (not all!)
        sample_size = min(50, num_buffers)
        
        for _ in range(sample_size):
            idx = int(_random() * num_buffers)
            buf = buffers[idx]
            if buf.tag is not None and buf.block_group is not None:
                bg = buf.block_group
                next_access, _ = _estimate(bg.relation_id, bg.group_num, current_time)
                _insert(bg.relation_id, bg.group_num, next_access)
    
    # Fast path: find empty buffer
    for _ in range(5):
        buf_id = int(_random() * num_buffers)
        buf = buffers[buf_id]
        if buf.refcount == 0 and buf.tag is None:
            return buf_id
    
    # Get eviction candidates from PQ (already a set, no conversion needed)
    candidate_set = _pq.get_eviction_candidates()
    
    # OPTIMIZATION: Don't iterate all buffers - sample instead!
    # Sample buffers and return first one that matches a candidate block group
    fallback_buf_id = None
    
    for _ in range(30):  # Sample 30 buffers instead of iterating all
        buf_id = int(_random() * num_buffers)
        buf = buffers[buf_id]
        
        if buf.refcount > 0:
            continue
        
        if buf.tag is None:
            return buf_id  # Empty buffer
        
        if buf.block_group is not None:
            bg = buf.block_group
            key = (bg.relation_id, bg.group_num)
            
            # Check if this buffer is in a candidate block group
            if key in candidate_set:
                return buf_id  # Found a match!
            
            # Track any evictable buffer as fallback (don't call estimator!)
            if fallback_buf_id is None:
                fallback_buf_id = buf_id
    
    # Return fallback buffer
    if fallback_buf_id is not None:
        return fallback_buf_id
    
    # Fallback: find any unpinned buffer
    for _ in range(20):
        buf_id = int(_random() * num_buffers)
        if buffers[buf_id].refcount == 0:
            return buf_id
    
    return None


def reset_pq():
    """Reset the priority queue (for testing)."""
    global _pq, _last_refresh_time
    _pq = ApproximatePriorityQueue()
    _last_refresh_time = 0.0

