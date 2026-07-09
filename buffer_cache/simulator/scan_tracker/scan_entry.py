"""
Scan Entry - Tracks an active scan's progress.

Matches PBM's ScanHashEntry: stores position, speed, and block range.
Extended with variance tracking for confidence-aware eviction.
"""
import bisect
import math
from dataclasses import dataclass, field
from typing import Optional, Set
from enum import Enum


class ScanType(Enum):
    """Types of scans we track."""
    SEQUENTIAL = "sequential"
    BITMAP = "bitmap"
    INDEX_CORRELATED = "index_correlated"  # B-tree on correlated column
    INDEX_TRAILING = "index_trailing"  # Following another scan


@dataclass
class ScanEntry:
    """
    Tracks an active scan's progress and speed.
    
    This is the core data structure for predicting next-access times.
    Position updates happen every ~32 blocks (amortized overhead).
    """
    scan_id: int
    relation_id: int
    scan_type: ScanType
    
    # Block range this scan will access
    start_block: int
    end_block: int
    
    # Current progress
    current_position: int = 0
    blocks_processed: int = 0
    
    # Timing for speed estimation
    start_time: float = 0.0
    last_update_time: float = 0.0
    
    # Speed estimation (EWMA like PBM)
    est_blocks_per_sec: float = 0.0
    
    # NEW: Variance tracking for confidence-aware eviction
    speed_variance: float = 0.0
    speed_samples: int = 0
    
    # For bitmap scans: which blocks are in the bitmap
    bitmap: Optional[Set[int]] = None
    _sorted_bitmap: Optional[list] = field(default=None, repr=False)  # Cached sorted list
    
    # Tracking state
    is_active: bool = True
    
    @property
    def total_blocks(self) -> int:
        """Total blocks this scan will access."""
        if self.bitmap is not None:
            return len(self.bitmap)
        return self.end_block - self.start_block
    
    @property
    def blocks_remaining(self) -> int:
        """Blocks remaining to be scanned."""
        return max(0, self.total_blocks - self.blocks_processed)
    
    @property
    def progress_fraction(self) -> float:
        """Fraction of scan completed (0.0 to 1.0)."""
        total = self.total_blocks
        if total == 0:
            return 1.0
        return min(1.0, self.blocks_processed / total)
    
    @property
    def speed_confidence(self) -> float:
        """
        Confidence in speed prediction (0.0 = no confidence, 1.0 = high confidence).
        Based on coefficient of variation (CV = stddev/mean).
        """
        if self.speed_samples < 3:
            return 0.3  # Not enough data - low but not zero confidence
        if self.est_blocks_per_sec <= 0:
            return 0.0
        
        stddev = math.sqrt(max(0, self.speed_variance))
        cv = stddev / self.est_blocks_per_sec
        
        # CV of 0 → confidence 1.0
        # CV of 0.5+ → confidence ~0.0
        return max(0.0, min(1.0, 1.0 - cv * 2))
    
    def update_position(self, new_position: int, current_time: float):
        """
        Update scan position and recompute speed estimate.
        
        Uses EWMA (Exponentially Weighted Moving Average) like PBM:
        70% old estimate + 30% new measurement.
        Also tracks variance for confidence estimation.
        """
        # Compute speed if we have a previous timestamp and time has passed
        elapsed = current_time - self.last_update_time
        if elapsed > 0 and self.last_update_time >= 0:
                # For both sequential and bitmap, just use position difference
                # (bitmap positions are indices into sorted blocks, not block numbers)
                blocks = new_position - self.current_position
                
                if blocks > 0:
                    new_speed = blocks / elapsed
                    
                    # NEW: Track variance BEFORE updating mean
                    if self.speed_samples > 0 and self.est_blocks_per_sec > 0:
                        deviation = new_speed - self.est_blocks_per_sec
                        self.speed_variance = (
                            0.7 * self.speed_variance + 
                            0.3 * (deviation ** 2)
                        )
                    
                    # EWMA update (existing)
                    if self.est_blocks_per_sec == 0:
                        self.est_blocks_per_sec = new_speed
                    else:
                        self.est_blocks_per_sec = (
                            0.7 * self.est_blocks_per_sec + 
                            0.3 * new_speed
                        )
                    
                    self.speed_samples += 1
                    self.blocks_processed += blocks
        
        self.current_position = new_position
        self.last_update_time = current_time
    
    def estimate_time_to_block(self, block_num: int, current_time: float) -> float:
        """
        Estimate when this scan will reach a specific block.
        
        Returns:
            Estimated timestamp, or float('inf') if unreachable.
        """
        # For bitmap scans, check if block is in bitmap
        if self.bitmap is not None and block_num not in self.bitmap:
            return float('inf')
        
        # If already passed this block
        if block_num < self.current_position:
            return float('inf')  # Won't access again
        
        # If speed unknown, assume fast
        if self.est_blocks_per_sec <= 0:
            # Default: assume ~10000 blocks/sec
            return current_time + (block_num - self.current_position) / 10000.0
        
        distance = block_num - self.current_position
        return current_time + (distance / self.est_blocks_per_sec)
    
    def estimate_time_to_group(self, group_start: int, group_end: int, 
                               current_time: float) -> float:
        """
        Estimate when this scan will reach a block group.
        
        Uses the first block in the group that the scan will access.
        """
        if self.bitmap is not None:
            # Use cached sorted bitmap (avoid sorting on every call!)
            if self._sorted_bitmap is None:
                self._sorted_bitmap = sorted(self.bitmap)
            
            # Binary search for efficiency O(log n) instead of O(n)
            # Find first block >= max(group_start, current_position)
            search_start = max(group_start, self.current_position)
            idx = bisect.bisect_left(self._sorted_bitmap, search_start)
            
            if idx < len(self._sorted_bitmap):
                block = self._sorted_bitmap[idx]
                if block < group_end:
                    return self.estimate_time_to_block(block, current_time)
            return float('inf')
        else:
            # Sequential: use start of group
            target = max(group_start, self.current_position)
            if target >= group_end:
                return float('inf')
            return self.estimate_time_to_block(target, current_time)
    
    def finish(self):
        """Mark scan as complete."""
        self.is_active = False
    
    def __repr__(self):
        return (f"ScanEntry(id={self.scan_id}, rel={self.relation_id}, "
                f"type={self.scan_type.value}, pos={self.current_position}, "
                f"speed={self.est_blocks_per_sec:.0f} blk/s)")

