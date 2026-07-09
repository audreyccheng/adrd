"""
PBM Sampling Realistic Baseline - Matches key postgres-pbm features.

Key features from postgres-pbm that HELP performance:
1. ✅ Sampling-based eviction (N=7 default)
2. ✅ Next-access estimation for scans (core PBM)
3. ✅ LRU tiebreaker when next_access equal
4. ✅ LRU fallback for non-requested pages

From thesis pbm.c bh_bufsamp_cmp():
  if (a->next_access_time > b->next_access_time) return true;
  if (a->next_access_time == b->next_access_time) 
      return (a->last_access < b->last_access);  // LRU tiebreaker
"""
import random
from typing import List, Dict, Optional, Any, Tuple


# Configuration matching postgres-pbm GUCs
PBM_EVICT_NUM_SAMPLES = 7        # pbm_evict_num_samples
PBM_LRU_IF_NOT_REQUESTED = True  # pbm_lru_if_not_requested


def pbm_sampling_realistic(
    buffers: List,
    buffer_table: Dict,
    estimator: Any,
    scan_context: Any,
    num_samples: int = PBM_EVICT_NUM_SAMPLES,
) -> Optional[int]:
    """
    Realistic PBM-sampling that matches postgres-pbm implementation.
    
    Key differences from simplified pbm_sampling_evolvable.py:
    1. LRU tiebreaker: when next_access equal, prefer older last_access
    2. LRU fallback: for non-requested pages, use LRU ordering
    """
    if estimator is None:
        return None
    
    num_buffers = len(buffers)
    if num_buffers == 0:
        return None
    
    _random = random.random
    
    # Fast path: find empty buffer first (same as postgres-pbm)
    for _ in range(3):
        buf_id = int(_random() * num_buffers)
        buf = buffers[buf_id]
        if buf.refcount == 0 and buf.tag is None:
            return buf_id
    
    # Sample buffers and compute estimates
    # Each sample: (buf_id, next_access, last_access, is_requested)
    samples: List[Tuple[int, float, float, bool]] = []
    seen: set = set()
    attempts = 0
    max_attempts = num_samples * 3
    
    _estimate = estimator.estimate_for_buffer
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        buf_id = int(_random() * num_buffers)
        
        if buf_id in seen:
            continue
        seen.add(buf_id)
        
        buf = buffers[buf_id]
        
        if buf.refcount > 0:
            continue
        
        if buf.tag is None:
            return buf_id
        
        # Get scan-based next-access estimate
        next_access, is_requested = _estimate(buf)
        
        # Get last_access for LRU tiebreaker
        last_access = getattr(buf, 'last_access_time', 0.0)
        
        samples.append((buf_id, next_access, last_access, is_requested))
    
    if not samples:
        return None
    
    # Sort by: next_access DESC, then last_access ASC (LRU tiebreaker)
    # This matches bh_bufsamp_cmp() in pbm.c
    samples.sort(key=lambda s: (-s[1], s[2]))
    
    # LRU fallback for non-requested pages (pbm_lru_if_not_requested)
    if PBM_LRU_IF_NOT_REQUESTED:
        # If best candidate is not requested (next_access == inf), use LRU among non-requested
        non_requested = [(bid, na, la, req) for bid, na, la, req in samples if not req]
        if non_requested and not samples[0][3]:  # Best is not requested
            # Sort non-requested by last_access (oldest first = LRU)
            non_requested.sort(key=lambda s: s[2])
            best_buf_id = non_requested[0][0]
            if buffers[best_buf_id].refcount == 0:
                return best_buf_id
    
    # Return buffer with largest next_access (furthest in future)
    best_buf_id = samples[0][0]
    if buffers[best_buf_id].refcount == 0:
        return best_buf_id
    
    # Fallback: find any evictable from samples
    for buf_id, _, _, _ in samples:
        if buffers[buf_id].refcount == 0:
            return buf_id
    
    return None


# Alias for compatibility
evolved_policy = pbm_sampling_realistic
pbm_sampling_policy = pbm_sampling_realistic
