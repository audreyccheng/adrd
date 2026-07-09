"""
Next Access Estimator - Computes on-demand next-access estimates.

This is the core of PBM Mode 3: compute estimates ONLY when needed,
at eviction time, using the freshest scan data available.
"""
from typing import Tuple, Optional, TYPE_CHECKING

from core.block_group import BLOCKS_PER_GROUP

if TYPE_CHECKING:
    from .scan_registry import ScanRegistry
    from core.block_group import BlockGroupRegistry
    from core.buffer_descriptor import BufferDescriptor


class NextAccessEstimator:
    """
    Computes next-access-time estimates for buffers.
    
    Key insight from PBM Mode 3:
    - Don't pre-compute and cache estimates (they go stale)
    - Compute on-demand at eviction time (fresh data)
    - This is why Mode 3 beats Mode 2 (priority queue)
    """
    
    def __init__(self, scan_registry: 'ScanRegistry', 
                 block_groups: 'BlockGroupRegistry'):
        self.scans = scan_registry
        self.block_groups = block_groups
    
    def estimate_next_access(self, relation_id: int, block_num: int,
                             current_time: Optional[float] = None) -> Tuple[float, bool]:
        """
        Estimate when this block will next be accessed.
        
        Args:
            relation_id: Which relation
            block_num: Which block
            current_time: Current time (uses scan registry time if None)
        
        Returns:
            (estimated_time, is_requested_by_scan)
            - estimated_time: When next access is expected (float('inf') if never)
            - is_requested_by_scan: True if at least one scan wants this block
        """
        est_time, is_requested, _ = self.estimate_with_confidence(
            relation_id, block_num, current_time
        )
        return (est_time, is_requested)
    
    def estimate_with_confidence(self, relation_id: int, block_num: int,
                                  current_time: Optional[float] = None) -> Tuple[float, bool, float]:
        """
        Estimate when this block will next be accessed, with confidence.
        
        Args:
            relation_id: Which relation
            block_num: Which block
            current_time: Current time (uses scan registry time if None)
        
        Returns:
            (estimated_time, is_requested_by_scan, confidence)
            - estimated_time: When next access is expected (float('inf') if never)
            - is_requested_by_scan: True if at least one scan wants this block
            - confidence: 0.0 to 1.0, based on scan speed variance
        """
        if current_time is None:
            current_time = self.scans.get_time()
        
        # Get block group
        bg = self.block_groups.get(relation_id, block_num)
        
        # If no block group or no scans, not requested
        if bg is None or not bg.scan_ids:
            return (float('inf'), False, 0.0)
        
        # Find minimum time across all scans that want this block
        min_time = float('inf')
        any_requested = False
        min_confidence = 1.0  # Track minimum confidence
        num_scans = 0
        
        for scan_id in bg.scan_ids:
            scan = self.scans.get_scan(scan_id)
            if scan is None or not scan.is_active:
                continue
            
            est_time = scan.estimate_time_to_block(block_num, current_time)
            if est_time < float('inf'):
                any_requested = True
                num_scans += 1
                # Track confidence of the scan providing minimum time
                if est_time < min_time:
                    min_time = est_time
                    min_confidence = scan.speed_confidence
        
        # Multiple scans wanting same block → lower confidence
        # (harder to predict which arrives first)
        if num_scans > 1:
            min_confidence *= 0.7  # Penalty for multi-scan uncertainty
        
        return (min_time, any_requested, min_confidence)
    
    def estimate_for_buffer(self, buf: 'BufferDescriptor',
                            current_time: Optional[float] = None) -> Tuple[float, bool]:
        """
        Estimate next access for a buffer.
        
        Convenience method that extracts relation_id and block_num from buffer tag.
        """
        if buf.tag is None:
            return (float('inf'), False)
        
        relation_id, block_num = buf.tag[0], buf.tag[1]
        return self.estimate_next_access(relation_id, block_num, current_time)
    
    def estimate_for_buffer_with_confidence(self, buf: 'BufferDescriptor',
                                             current_time: Optional[float] = None) -> Tuple[float, bool, float]:
        """
        Estimate next access for a buffer, with confidence.
        
        Returns: (estimated_time, is_requested, confidence)
        """
        if buf.tag is None:
            return (float('inf'), False, 0.0)
        
        relation_id, block_num = buf.tag[0], buf.tag[1]
        return self.estimate_with_confidence(relation_id, block_num, current_time)
    
    def estimate_for_block_group(self, relation_id: int, group_num: int,
                                  current_time: Optional[float] = None) -> Tuple[float, bool]:
        """
        Estimate next access for an entire block group.
        
        Returns the minimum time any scan will access this group.
        """
        if current_time is None:
            current_time = self.scans.get_time()
        
        bg = self.block_groups.get(relation_id, group_num * BLOCKS_PER_GROUP)
        if bg is None or not bg.scan_ids:
            return (float('inf'), False)
        
        min_time = float('inf')
        any_requested = False
        
        for scan_id in bg.scan_ids:
            scan = self.scans.get_scan(scan_id)
            if scan is None or not scan.is_active:
                continue
            
            est_time = scan.estimate_time_to_group(
                bg.start_block, bg.end_block, current_time
            )
            if est_time < float('inf'):
                any_requested = True
                if est_time < min_time:
                    min_time = est_time
        
        return (min_time, any_requested)
    
    def get_buffer_priority(self, buf: 'BufferDescriptor',
                            current_time: Optional[float] = None) -> float:
        """
        Get eviction priority for a buffer.
        
        Higher value = better candidate for eviction.
        Uses next-access-time: larger time = better to evict.
        
        This is the key metric for PBM-style eviction:
        Evict buffer with LARGEST next-access time (furthest future).
        """
        next_access, requested = self.estimate_for_buffer(buf, current_time)
        
        if not requested:
            # Not requested by any scan - high priority for eviction
            # But consider frequency stats if available
            if buf.access_count > 1 and buf.last_access_time > 0:
                # Use time since last access as proxy
                time_since_access = (current_time or self.scans.get_time()) - buf.last_access_time
                return time_since_access
            else:
                # No history - very high priority
                return float('inf')
        
        return next_access

