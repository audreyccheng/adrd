"""
Clock Sweep Policy - PostgreSQL baseline.

This is the standard PostgreSQL buffer replacement algorithm.
Used as baseline for comparison with PBM approaches.

Characteristics:
- O(1) amortized per eviction
- Lock-free (uses atomic operations in real PostgreSQL)
- No scan tracking overhead
- Simple and well-tested
"""
from typing import List, Dict, Optional, Any

# Global clock hand (simulates PostgreSQL's global state)
_clock_hand = 0


def clock_sweep_policy(
    buffers: List,
    buffer_table: Dict,
    estimator: Any,  # Not used by clock sweep
    scan_context: Any  # Not used by clock sweep
) -> Optional[int]:
    """
    PostgreSQL's clock sweep algorithm.
    
    Algorithm:
    1. Check buffer at clock hand
    2. If unpinned and usage_count == 0, evict it
    3. Otherwise, decrement usage_count and advance
    4. Repeat until victim found
    
    Args:
        buffers: List of BufferDescriptor
        buffer_table: tag -> buf_id mapping (not used)
        estimator: NextAccessEstimator (not used)
        scan_context: Current scan info (not used)
    
    Returns:
        buf_id to evict, or None if all pinned
    """
    global _clock_hand
    
    num_buffers = len(buffers)
    if num_buffers == 0:
        return None
    
    # Make at most 2 passes through the buffer pool
    for _ in range(num_buffers * 2):
        buf = buffers[_clock_hand]
        _clock_hand = (_clock_hand + 1) % num_buffers
        
        # Skip pinned buffers
        if buf.refcount > 0:
            continue
        
        # If usage_count is 0, this is our victim
        if buf.usage_count == 0:
            return buf.buf_id
        
        # Decrement usage count (give it another chance)
        buf.usage_count -= 1
    
    # All buffers are either pinned or have usage_count > 0
    # Return None to trigger fallback
    return None


def reset_clock_hand():
    """Reset clock hand (for testing)."""
    global _clock_hand
    _clock_hand = 0

