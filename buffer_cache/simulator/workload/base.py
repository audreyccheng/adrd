"""
Base classes for workload generation.

Provides the interface between workloads and the buffer manager/scan tracker.
"""
from dataclasses import dataclass
from typing import Optional, Set
from enum import Enum


class AccessType(Enum):
    """Types of buffer access patterns."""
    SEQUENTIAL = "sequential"
    BITMAP = "bitmap"
    INDEX = "index"
    RANDOM = "random"


@dataclass
class ScanContext:
    """
    Context for a scan, exposed to eviction policies.
    
    This is the key information PBM uses for prediction:
    - Which scan is this access part of
    - Where is the scan in its execution
    - How fast is it progressing
    """
    scan_id: int
    relation_id: int
    access_type: AccessType
    
    # Scan progress
    current_position: int
    total_blocks: int
    blocks_remaining: int
    
    # Speed estimate (blocks per second)
    est_blocks_per_sec: float
    
    # For bitmap scans: which blocks are included
    bitmap: Optional[Set[int]] = None
    
    @property
    def progress(self) -> float:
        """Fraction of scan completed (0.0 to 1.0)."""
        if self.total_blocks == 0:
            return 1.0
        return 1.0 - (self.blocks_remaining / self.total_blocks)


@dataclass
class WorkloadAccess:
    """
    A single buffer access in a workload.
    
    Includes:
    - tag: Which page to access
    - access_type: Sequential, bitmap, index, random
    - scan_context: Full scan information for PBM tracking
    """
    tag: tuple  # (relation_id, block_num)
    access_type: AccessType
    is_write: bool = False
    
    # Scan context for PBM-style tracking
    scan_context: Optional[ScanContext] = None
    
    @property
    def relation_id(self) -> int:
        return self.tag[0]
    
    @property
    def block_num(self) -> int:
        return self.tag[1]


class BaseWorkload:
    """
    Base class for workload generators.
    
    Subclasses implement generate() to yield WorkloadAccess objects.
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self._access_count = 0
    
    def generate(self):
        """
        Generate workload accesses.
        
        Yields: WorkloadAccess objects
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset workload state for re-execution."""
        self._access_count = 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

