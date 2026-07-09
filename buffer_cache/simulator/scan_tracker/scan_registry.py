"""
Scan Registry - Manages all active scans.

Provides scan registration, position updates, and cleanup.
Links scans to block groups for efficient next-access estimation.
"""
from typing import Dict, Optional, Set, List
import time

from .scan_entry import ScanEntry, ScanType
from core.block_group import BlockGroupRegistry, BLOCKS_PER_GROUP


class ScanRegistry:
    """
    Registry of all active scans in the system.
    
    Key operations:
    - register_scan(): Create scan entry, link to block groups
    - update_position(): Update scan progress, unlink passed groups  
    - unregister_scan(): Clean up when scan completes
    """
    
    def __init__(self, block_groups: BlockGroupRegistry):
        self._scans: Dict[int, ScanEntry] = {}
        self._block_groups = block_groups
        self._next_scan_id = 1
        self._current_time = 0.0  # Simulated time
    
    def set_time(self, t: float):
        """Set current simulated time."""
        self._current_time = t
    
    def get_time(self) -> float:
        """Get current simulated time."""
        return self._current_time
    
    def register_sequential_scan(self, relation_id: int, 
                                  start_block: int, end_block: int) -> int:
        """
        Register a new sequential scan.
        
        Returns scan_id for tracking.
        """
        scan_id = self._next_scan_id
        self._next_scan_id += 1
        
        entry = ScanEntry(
            scan_id=scan_id,
            relation_id=relation_id,
            scan_type=ScanType.SEQUENTIAL,
            start_block=start_block,
            end_block=end_block,
            current_position=start_block,
            start_time=self._current_time,
            last_update_time=self._current_time,
        )
        
        self._scans[scan_id] = entry
        
        # Link to all block groups this scan will access
        for bg in self._block_groups.get_groups_for_range(
            relation_id, start_block, end_block
        ):
            bg.add_scan(scan_id)
        
        return scan_id
    
    def register_bitmap_scan(self, relation_id: int, bitmap: Set[int]) -> int:
        """
        Register a bitmap scan with specific blocks.
        
        The bitmap contains all block numbers that will be accessed.
        """
        if not bitmap:
            return -1  # Empty bitmap
        
        scan_id = self._next_scan_id
        self._next_scan_id += 1
        
        sorted_blocks = sorted(bitmap)
        entry = ScanEntry(
            scan_id=scan_id,
            relation_id=relation_id,
            scan_type=ScanType.BITMAP,
            start_block=sorted_blocks[0],
            end_block=sorted_blocks[-1] + 1,
            current_position=sorted_blocks[0],
            start_time=self._current_time,
            last_update_time=self._current_time,
            bitmap=bitmap,
        )
        
        self._scans[scan_id] = entry
        
        # Link to block groups containing bitmap blocks
        linked_groups: Set[int] = set()
        for block in bitmap:
            group_num = block // BLOCKS_PER_GROUP
            if group_num not in linked_groups:
                bg = self._block_groups.get_or_create(relation_id, block)
                bg.add_scan(scan_id)
                linked_groups.add(group_num)
        
        return scan_id
    
    def update_scan_position(self, scan_id: int, new_position: int):
        """
        Update scan position and speed estimate.
        
        Also removes scan from block groups it has passed.
        Called every ~32 blocks to amortize overhead.
        """
        entry = self._scans.get(scan_id)
        if entry is None or not entry.is_active:
            return
        
        old_group = entry.current_position // BLOCKS_PER_GROUP
        new_group = new_position // BLOCKS_PER_GROUP
        
        # Update position and speed
        entry.update_position(new_position, self._current_time)
        
        # Remove from passed block groups
        for g in range(old_group, new_group):
            bg = self._block_groups.get(entry.relation_id, g * BLOCKS_PER_GROUP)
            if bg is not None:
                bg.remove_scan(scan_id)
    
    def unregister_scan(self, scan_id: int):
        """
        Unregister scan when complete.
        
        Removes scan from all remaining block groups.
        """
        entry = self._scans.get(scan_id)
        if entry is None:
            return
        
        entry.finish()
        
        # Remove from all remaining block groups
        if entry.bitmap is not None:
            # Bitmap scan: check all bitmap blocks
            for block in entry.bitmap:
                if block >= entry.current_position:
                    bg = self._block_groups.get(entry.relation_id, block)
                    if bg is not None:
                        bg.remove_scan(scan_id)
        else:
            # Sequential scan: check range
            for bg in self._block_groups.get_groups_for_range(
                entry.relation_id, entry.current_position, entry.end_block
            ):
                bg.remove_scan(scan_id)
        
        del self._scans[scan_id]
    
    def get_scan(self, scan_id: int) -> Optional[ScanEntry]:
        """Get scan entry by ID."""
        return self._scans.get(scan_id)
    
    def get_active_scans(self) -> List[ScanEntry]:
        """Get all active scans."""
        return [s for s in self._scans.values() if s.is_active]
    
    def get_scans_for_relation(self, relation_id: int) -> List[ScanEntry]:
        """Get all active scans for a specific relation."""
        return [s for s in self._scans.values() 
                if s.is_active and s.relation_id == relation_id]
    
    def clear(self):
        """Clear all scans."""
        self._scans.clear()
        self._next_scan_id = 1
    
    def __len__(self):
        return len(self._scans)
    
    def __repr__(self):
        active = sum(1 for s in self._scans.values() if s.is_active)
        return f"ScanRegistry({active} active scans)"

