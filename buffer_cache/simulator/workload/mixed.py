"""
Mixed TPC-H + TPC-C Workload for testing interleaved access patterns.

Probability-based mixing: each access randomly comes from TPC-H or TPC-C.
This tests policy adaptation to mixed workload patterns in a shared buffer pool.

Key characteristics:
- TPC-H: Large sequential/bitmap scans (read-only)
- TPC-C: Point lookups and small updates (read-write, dirty pages)
- Shared buffer pool creates contention
- Tests: Can policy protect TPC-C hot pages from TPC-H scan eviction?
"""
import random
from dataclasses import dataclass
from typing import Generator

from .base import WorkloadAccess, BaseWorkload
from .tpch_full import TPCHWorkload, TPCHConfig
from .tpcc import TPCCWorkload, TPCCConfig


# Offset TPC-C relation IDs to avoid conflict with TPC-H (which uses 0-7)
TPCC_RELATION_OFFSET = 100


@dataclass
class MixedWorkloadConfig:
    """Configuration for mixed TPC-H + TPC-C workload."""
    # TPC-H config (scaled down for mixed testing)
    tpch_num_streams: int = 4
    tpch_table_scale: float = 0.03  # Smaller to fit shared buffer
    
    # TPC-C config (scaled down)
    tpcc_num_warehouses: int = 10
    tpcc_num_terminals: int = 5
    tpcc_transactions_per_terminal: int = 100
    
    # Mixing parameters
    tpch_probability: float = 0.5  # 50% TPC-H, 50% TPC-C
    
    # Shared buffer pool
    buffer_pool_blocks: int = 32768  # 256MB shared


class MixedWorkload(BaseWorkload):
    """
    Interleaved TPC-H + TPC-C workload.
    
    Each access is randomly drawn from either TPC-H or TPC-C based on
    tpch_probability. Both workloads share the same buffer pool, creating
    realistic contention between analytical scans and OLTP point lookups.
    """
    
    def __init__(self, config: MixedWorkloadConfig):
        super().__init__(name="mixed_tpch_tpcc")
        self.config = config
        
        # Create TPC-H workload
        tpch_config = TPCHConfig(
            num_streams=config.tpch_num_streams,
            table_scale=config.tpch_table_scale,
            buffer_pool_blocks=config.buffer_pool_blocks,
            synchronized_seqscans=True,
        )
        self.tpch = TPCHWorkload(tpch_config)
        
        # Create TPC-C workload
        tpcc_config = TPCCConfig(
            num_warehouses=config.tpcc_num_warehouses,
            num_terminals=config.tpcc_num_terminals,
            transactions_per_terminal=config.tpcc_transactions_per_terminal,
            buffer_pool_blocks=config.buffer_pool_blocks,
        )
        self.tpcc = TPCCWorkload(tpcc_config)
        
        self.tpch_prob = config.tpch_probability
    
    def _offset_tpcc_access(self, access: WorkloadAccess) -> WorkloadAccess:
        """Offset TPC-C relation IDs to avoid collision with TPC-H."""
        # Offset the relation ID in the tag
        new_tag = (access.tag[0] + TPCC_RELATION_OFFSET, access.tag[1])
        
        # Offset scan context if present
        new_scan_context = None
        if access.scan_context:
            ctx = access.scan_context
            from .base import ScanContext
            new_scan_context = ScanContext(
                scan_id=ctx.scan_id + 10000,  # Offset scan IDs too
                relation_id=ctx.relation_id + TPCC_RELATION_OFFSET,
                access_type=ctx.access_type,
                current_position=ctx.current_position,
                total_blocks=ctx.total_blocks,
                blocks_remaining=ctx.blocks_remaining,
                est_blocks_per_sec=ctx.est_blocks_per_sec,
                bitmap=ctx.bitmap,
            )
        
        return WorkloadAccess(
            tag=new_tag,
            access_type=access.access_type,
            is_write=access.is_write,
            scan_context=new_scan_context,
        )
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """
        Generate interleaved accesses from TPC-H and TPC-C.
        
        Randomly picks from TPC-H or TPC-C based on tpch_probability.
        Continues until both workloads are exhausted.
        """
        tpch_gen = self.tpch.generate()
        tpcc_gen = self.tpcc.generate()
        
        tpch_done = False
        tpcc_done = False
        
        tpch_buffer = None
        tpcc_buffer = None
        
        while not (tpch_done and tpcc_done):
            # Decide which workload to draw from
            if tpch_done:
                use_tpch = False
            elif tpcc_done:
                use_tpch = True
            else:
                use_tpch = random.random() < self.tpch_prob
            
            if use_tpch:
                # Get next TPC-H access
                if tpch_buffer is not None:
                    yield tpch_buffer
                    tpch_buffer = None
                else:
                    try:
                        access = next(tpch_gen)
                        yield access
                        self._access_count += 1
                    except StopIteration:
                        tpch_done = True
            else:
                # Get next TPC-C access (with offset)
                if tpcc_buffer is not None:
                    yield tpcc_buffer
                    tpcc_buffer = None
                else:
                    try:
                        access = next(tpcc_gen)
                        yield self._offset_tpcc_access(access)
                        self._access_count += 1
                    except StopIteration:
                        tpcc_done = True
    
    def reset(self):
        """Reset both workloads."""
        super().reset()
        self.tpch.reset()
        self.tpcc.reset()


