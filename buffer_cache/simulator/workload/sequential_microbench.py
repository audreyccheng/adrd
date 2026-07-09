"""
Sequential Microbenchmarks from PBM Thesis Section 6.2.

Key parameters (from thesis):
- TPC-H SF10: ~8.6 GiB lineitem table
- BRIN index on l_shipdate
- Each query scans ~30% of table  
- 1-32 parallel query streams
- 16 queries per stream
"""
import random
from dataclasses import dataclass
from typing import Generator, List, Set

from .base import WorkloadAccess, ScanContext, AccessType, BaseWorkload

# TPC-H SF10 parameters
LINEITEM_BLOCKS = 1100000  # ~8.6 GiB at 8KB blocks
BRIN_PAGES_PER_RANGE = 128  # 1 MiB per BRIN range (matches block groups!)


@dataclass
class SequentialMicrobenchConfig:
    """Configuration matching PBM thesis Section 6.2."""
    num_streams: int = 8
    queries_per_stream: int = 16
    table_blocks: int = LINEITEM_BLOCKS
    selectivity: float = 0.30  # 30% of table per query
    buffer_pool_blocks: int = 327680  # 2.5 GiB
    
    # Default scan speed (blocks per simulated second)
    scan_speed: float = 10000.0  # ~80 MB/s
    
    # Time per block (for simulation)
    time_per_block: float = 0.0001  # 0.1ms per block


class SequentialMicrobench(BaseWorkload):
    """
    Generates workload matching PBM thesis Section 6.2.
    
    Simulates concurrent BRIN bitmap scans with 30% selectivity.
    """
    
    def __init__(self, config: SequentialMicrobenchConfig = None):
        super().__init__(name="sequential_microbench")
        self.config = config or SequentialMicrobenchConfig()
        self._scan_id_counter = 0
        self._current_time = 0.0
    
    def _generate_brin_bitmap(self) -> Set[int]:
        """
        Generate a BRIN bitmap for a 30% selectivity query.
        
        BRIN works by checking each 128-block range against the filter.
        With correlated-but-not-sorted data, ~30% of ranges match.
        """
        num_ranges = self.config.table_blocks // BRIN_PAGES_PER_RANGE
        num_selected = int(num_ranges * self.config.selectivity)
        
        # Select random ranges (simulates date filter selectivity)
        selected_ranges = random.sample(range(num_ranges), num_selected)
        
        # Build bitmap from selected ranges
        bitmap = set()
        for range_idx in selected_ranges:
            start = range_idx * BRIN_PAGES_PER_RANGE
            end = min(start + BRIN_PAGES_PER_RANGE, self.config.table_blocks)
            for block in range(start, end):
                bitmap.add(block)
        
        return bitmap
    
    def _generate_query(self, stream_id: int) -> Generator[WorkloadAccess, None, None]:
        """Generate accesses for a single query (bitmap scan)."""
        scan_id = self._scan_id_counter
        self._scan_id_counter += 1
        
        bitmap = self._generate_brin_bitmap()
        blocks = sorted(bitmap)
        total_blocks = len(blocks)
        
        for i, block in enumerate(blocks):
            scan_context = ScanContext(
                scan_id=scan_id,
                relation_id=0,  # lineitem
                access_type=AccessType.BITMAP,
                current_position=block,
                total_blocks=total_blocks,
                blocks_remaining=total_blocks - i - 1,
                est_blocks_per_sec=self.config.scan_speed,
                bitmap=bitmap,
            )
            
            yield WorkloadAccess(
                tag=(0, block),
                access_type=AccessType.BITMAP,
                scan_context=scan_context,
            )
            
            self._current_time += self.config.time_per_block
    
    def _generate_stream(self, stream_id: int) -> Generator[WorkloadAccess, None, None]:
        """Generate a complete query stream."""
        for query_num in range(self.config.queries_per_stream):
            yield from self._generate_query(stream_id)
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """
        Generate interleaved workload from multiple streams.
        
        Simulates concurrent execution by round-robin interleaving.
        """
        self._scan_id_counter = 0
        self._current_time = 0.0
        
        # Create generators for each stream
        streams = [self._generate_stream(i) for i in range(self.config.num_streams)]
        active = list(range(len(streams)))
        
        # Round-robin interleaving
        while active:
            for idx in list(active):
                try:
                    access = next(streams[idx])
                    yield access
                except StopIteration:
                    active.remove(idx)
    
    def generate_single_stream(self) -> Generator[WorkloadAccess, None, None]:
        """Generate a single stream (for simpler testing)."""
        self._scan_id_counter = 0
        self._current_time = 0.0
        yield from self._generate_stream(0)
    
    def get_expected_accesses(self) -> int:
        """Estimate total number of accesses."""
        blocks_per_query = int(self.config.table_blocks * self.config.selectivity)
        return self.config.num_streams * self.config.queries_per_stream * blocks_per_query


def create_sequential_microbench(
    parallelism: int = 8,
    cache_size_gb: float = 2.5,
    queries_per_stream: int = 16,
) -> SequentialMicrobench:
    """Factory function matching thesis experiment parameters."""
    config = SequentialMicrobenchConfig(
        num_streams=parallelism,
        queries_per_stream=queries_per_stream,
        buffer_pool_blocks=int(cache_size_gb * 1024 * 1024 / 8),
    )
    return SequentialMicrobench(config)

