"""
Block Group - Groups consecutive blocks for efficient scan tracking.

Matches PBM's BlockGroupData: 128 blocks = 1 MiB per group.
This is the key innovation that makes scan tracking practical.
"""
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

# 128 blocks × 8KB = 1 MiB per group (matches PostgreSQL BRIN default)
BLOCKS_PER_GROUP = 128


@dataclass
class BlockGroup:
    """
    Groups 128 consecutive blocks for efficient scan tracking.
    
    Key insight: Instead of tracking scans per-block (expensive),
    we track per 1 MiB chunk. This reduces metadata by 128x.
    """
    relation_id: int
    group_num: int
    
    # Scans that will access this group
    scan_ids: Set[int] = field(default_factory=set)
    
    # Cached next-access estimate (invalidated when scans change)
    est_next_access: Optional[float] = None
    est_computed_at: float = 0.0
    
    # Buffers currently cached from this group
    buffer_ids: List[int] = field(default_factory=list)
    
    def invalidate_estimate(self):
        """Invalidate cached estimate when scan list changes."""
        self.est_next_access = None
    
    @property
    def start_block(self) -> int:
        """First block number in this group."""
        return self.group_num * BLOCKS_PER_GROUP
    
    @property
    def end_block(self) -> int:
        """First block number AFTER this group."""
        return (self.group_num + 1) * BLOCKS_PER_GROUP
    
    def contains_block(self, block_num: int) -> bool:
        """Check if a block belongs to this group."""
        return self.start_block <= block_num < self.end_block
    
    def add_buffer(self, buf_id: int):
        """Track a buffer that contains a page from this group."""
        if buf_id not in self.buffer_ids:
            self.buffer_ids.append(buf_id)
    
    def remove_buffer(self, buf_id: int):
        """Remove a buffer from tracking."""
        if buf_id in self.buffer_ids:
            self.buffer_ids.remove(buf_id)
    
    def add_scan(self, scan_id: int):
        """Register a scan that will access this group."""
        self.scan_ids.add(scan_id)
        self.invalidate_estimate()
    
    def remove_scan(self, scan_id: int):
        """Unregister a scan from this group."""
        self.scan_ids.discard(scan_id)
        self.invalidate_estimate()
    
    def __repr__(self):
        return (f"BlockGroup(rel={self.relation_id}, group={self.group_num}, "
                f"blocks=[{self.start_block}-{self.end_block}), "
                f"scans={len(self.scan_ids)}, buffers={len(self.buffer_ids)})")


class BlockGroupRegistry:
    """
    Hash table of block groups, keyed by (relation_id, group_num).
    
    Provides O(1) lookup for any block -> block group mapping.
    """
    
    def __init__(self):
        self._groups: Dict[tuple, BlockGroup] = {}
    
    def get_or_create(self, relation_id: int, block_num: int) -> BlockGroup:
        """Get or create the block group containing a specific block."""
        group_num = block_num // BLOCKS_PER_GROUP
        key = (relation_id, group_num)
        
        if key not in self._groups:
            self._groups[key] = BlockGroup(relation_id, group_num)
        
        return self._groups[key]
    
    def get(self, relation_id: int, block_num: int) -> Optional[BlockGroup]:
        """Get block group if it exists, otherwise None."""
        group_num = block_num // BLOCKS_PER_GROUP
        return self._groups.get((relation_id, group_num))
    
    def get_groups_for_range(self, relation_id: int, 
                             start_block: int, end_block: int) -> List[BlockGroup]:
        """
        Get all block groups covering a range of blocks.
        
        Used during scan registration to link scan to all relevant groups.
        """
        start_group = start_block // BLOCKS_PER_GROUP
        end_group = (end_block - 1) // BLOCKS_PER_GROUP + 1
        
        return [
            self.get_or_create(relation_id, g * BLOCKS_PER_GROUP)
            for g in range(start_group, end_group)
        ]
    
    def clear(self):
        """Clear all block groups."""
        self._groups.clear()
    
    def __len__(self):
        return len(self._groups)
    
    def __repr__(self):
        return f"BlockGroupRegistry({len(self._groups)} groups)"


def block_to_group_num(block_num: int) -> int:
    """Convert a block number to its group number."""
    return block_num // BLOCKS_PER_GROUP

