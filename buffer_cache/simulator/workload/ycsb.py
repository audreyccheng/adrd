"""
YCSB Workload Generator for Buffer Simulator.

Implements YCSB-like workload with:
- Zipfian distribution for key access (hot spots)
- Configurable read/write ratio
- Point lookups and range scans

Key characteristics:
- Skewed access patterns (some keys much hotter than others)
- Simple key-value access model
- Configurable theta for Zipfian skew
"""
import random
import math
from dataclasses import dataclass
from typing import Generator, List
from enum import IntEnum

from .base import WorkloadAccess, ScanContext, AccessType, BaseWorkload


class YCSBOperation(IntEnum):
    """YCSB operation types."""
    READ = 0
    UPDATE = 1
    SCAN = 2
    INSERT = 3
    READ_MODIFY_WRITE = 4


@dataclass
class YCSBConfig:
    """Configuration for YCSB workload."""
    num_records: int = 1_000_000  # 1M records
    record_size_bytes: int = 1000  # 1KB per record
    buffer_pool_blocks: int = 131072  # 1 GB
    
    # Operation mix (default: YCSB-A 50/50 read/update)
    read_pct: float = 0.50
    update_pct: float = 0.50
    scan_pct: float = 0.0
    insert_pct: float = 0.0
    rmw_pct: float = 0.0  # Read-modify-write
    
    # Zipfian distribution parameter
    # theta=0.99 is very skewed, theta=0.0 is uniform
    zipfian_theta: float = 0.9
    
    # Scan parameters
    max_scan_length: int = 100
    
    # Workload size
    num_operations: int = 100_000
    num_threads: int = 8
    
    # Timing
    time_per_block: float = 0.0001


class ZipfianGenerator:
    """
    Zipfian distribution generator.
    
    Uses rejection sampling for efficiency.
    """
    
    def __init__(self, n: int, theta: float, seed: int = None):
        self.n = n
        self.theta = theta
        self._rng = random.Random(seed)
        
        # Pre-compute constants
        self._zetan = self._zeta(n, theta)
        self._zeta2 = self._zeta(2, theta)
        self._alpha = 1.0 / (1.0 - theta)
        self._eta = (1.0 - pow(2.0 / n, 1.0 - theta)) / (1.0 - self._zeta2 / self._zetan)
    
    def _zeta(self, n: int, theta: float) -> float:
        """Compute zeta(n, theta) = sum_{i=1}^{n} 1/i^theta."""
        total = 0.0
        for i in range(1, n + 1):
            total += 1.0 / pow(i, theta)
        return total
    
    def next(self) -> int:
        """Generate next Zipfian-distributed value in [0, n-1]."""
        u = self._rng.random()
        uz = u * self._zetan
        
        if uz < 1.0:
            return 0
        if uz < 1.0 + pow(0.5, self.theta):
            return 1
        
        return int(self.n * pow(self._eta * u - self._eta + 1.0, self._alpha))


class YCSBWorkload(BaseWorkload):
    """
    YCSB workload with Zipfian access distribution.
    
    Access patterns:
    - Point lookups with Zipfian key distribution (hot keys)
    - Range scans with Zipfian start key
    - Updates modeled as read + write to same block
    """
    
    def __init__(self, config: YCSBConfig = None):
        super().__init__(name="ycsb")
        self.config = config or YCSBConfig()
        
        # Calculate table size in blocks
        records_per_block = 8192 // self.config.record_size_bytes
        self.table_blocks = max(1, self.config.num_records // records_per_block)
        
        # Initialize Zipfian generator
        self._zipf = ZipfianGenerator(
            self.table_blocks, 
            self.config.zipfian_theta,
            seed=42  # Fixed seed for reproducibility
        )
        
        self._scan_id = 0
        self._current_time = 0.0
        self._insert_cursor = self.config.num_records  # For inserts
    
    def _next_scan_id(self) -> int:
        sid = self._scan_id
        self._scan_id += 1
        return sid
    
    def _get_zipf_block(self) -> int:
        """Get a block number using Zipfian distribution."""
        return self._zipf.next() % self.table_blocks
    
    def _read_op(self) -> Generator[WorkloadAccess, None, None]:
        """Point read operation."""
        block = self._get_zipf_block()
        yield WorkloadAccess(
            tag=(0, block),  # Table 0 = USERTABLE
            access_type=AccessType.INDEX,
            scan_context=None,
        )
        self._current_time += self.config.time_per_block
    
    def _update_op(self) -> Generator[WorkloadAccess, None, None]:
        """Point update operation (read + write same block)."""
        block = self._get_zipf_block()
        # Read
        yield WorkloadAccess(
            tag=(0, block),
            access_type=AccessType.INDEX,
            scan_context=None,
        )
        self._current_time += self.config.time_per_block
        # Write (same block, will be a hit, marks buffer dirty)
        yield WorkloadAccess(
            tag=(0, block),
            access_type=AccessType.INDEX,
            scan_context=None,
            is_write=True,  # Mark as write for dirty page tracking
        )
        self._current_time += self.config.time_per_block
    
    def _scan_op(self) -> Generator[WorkloadAccess, None, None]:
        """Range scan operation."""
        start_block = self._get_zipf_block()
        scan_length = random.randint(1, self.config.max_scan_length)
        scan_id = self._next_scan_id()
        
        for i in range(scan_length):
            block = (start_block + i) % self.table_blocks
            ctx = ScanContext(
                scan_id=scan_id,
                relation_id=0,
                access_type=AccessType.SEQUENTIAL,
                current_position=block,
                total_blocks=scan_length,
                blocks_remaining=scan_length - i - 1,
                est_blocks_per_sec=10000.0,
            )
            yield WorkloadAccess(
                tag=(0, block),
                access_type=AccessType.SEQUENTIAL,
                scan_context=ctx,
            )
            self._current_time += self.config.time_per_block
    
    def _insert_op(self) -> Generator[WorkloadAccess, None, None]:
        """Insert operation (append to end)."""
        # Inserts go to end of table
        block = self._insert_cursor % self.table_blocks
        self._insert_cursor += 1
        
        yield WorkloadAccess(
            tag=(0, block),
            access_type=AccessType.INDEX,
            scan_context=None,
        )
        self._current_time += self.config.time_per_block
    
    def _rmw_op(self) -> Generator[WorkloadAccess, None, None]:
        """Read-modify-write operation."""
        block = self._get_zipf_block()
        # Read
        yield WorkloadAccess(
            tag=(0, block),
            access_type=AccessType.INDEX,
            scan_context=None,
        )
        self._current_time += self.config.time_per_block
        # Modify + Write (same block)
        yield WorkloadAccess(
            tag=(0, block),
            access_type=AccessType.INDEX,
            scan_context=None,
        )
        self._current_time += self.config.time_per_block
    
    def _generate_thread_stream(self, thread_id: int
                                ) -> Generator[WorkloadAccess, None, None]:
        """Generate operations for one thread."""
        ops_per_thread = self.config.num_operations // self.config.num_threads
        
        for _ in range(ops_per_thread):
            r = random.random()
            cumulative = 0.0
            
            cumulative += self.config.read_pct
            if r < cumulative:
                yield from self._read_op()
                continue
            
            cumulative += self.config.update_pct
            if r < cumulative:
                yield from self._update_op()
                continue
            
            cumulative += self.config.scan_pct
            if r < cumulative:
                yield from self._scan_op()
                continue
            
            cumulative += self.config.insert_pct
            if r < cumulative:
                yield from self._insert_op()
                continue
            
            # Default: RMW
            yield from self._rmw_op()
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """Generate interleaved workload from multiple threads."""
        self._scan_id = 0
        self._current_time = 0.0
        
        streams = [self._generate_thread_stream(i) 
                   for i in range(self.config.num_threads)]
        active = list(range(len(streams)))
        
        while active:
            for idx in list(active):
                try:
                    access = next(streams[idx])
                    yield access
                except StopIteration:
                    active.remove(idx)
    
    def generate_single_thread(self) -> Generator[WorkloadAccess, None, None]:
        """Generate single thread stream."""
        self._scan_id = 0
        self._current_time = 0.0
        yield from self._generate_thread_stream(0)


def create_ycsb_workload(
    num_records: int = 1_000_000,
    read_pct: float = 0.50,
    update_pct: float = 0.50,
    zipfian_theta: float = 0.9,
    num_operations: int = 100_000,
    num_threads: int = 8,
    buffer_pool_mb: int = 1024,
) -> YCSBWorkload:
    """Factory function for YCSB workload."""
    config = YCSBConfig(
        num_records=num_records,
        read_pct=read_pct,
        update_pct=update_pct,
        zipfian_theta=zipfian_theta,
        num_operations=num_operations,
        num_threads=num_threads,
        buffer_pool_blocks=buffer_pool_mb * 128,
    )
    return YCSBWorkload(config)


# Preset configurations matching standard YCSB workloads
def create_ycsb_a(num_records: int = 1_000_000, **kwargs) -> YCSBWorkload:
    """YCSB-A: 50% read, 50% update (update heavy)."""
    return create_ycsb_workload(
        num_records=num_records,
        read_pct=0.50,
        update_pct=0.50,
        **kwargs
    )


def create_ycsb_b(num_records: int = 1_000_000, **kwargs) -> YCSBWorkload:
    """YCSB-B: 95% read, 5% update (read heavy)."""
    return create_ycsb_workload(
        num_records=num_records,
        read_pct=0.95,
        update_pct=0.05,
        **kwargs
    )


def create_ycsb_c(num_records: int = 1_000_000, **kwargs) -> YCSBWorkload:
    """YCSB-C: 100% read (read only)."""
    return create_ycsb_workload(
        num_records=num_records,
        read_pct=1.0,
        update_pct=0.0,
        **kwargs
    )


def create_ycsb_e(num_records: int = 1_000_000, **kwargs) -> YCSBWorkload:
    """YCSB-E: 95% scan, 5% insert (scan heavy)."""
    config = YCSBConfig(
        num_records=num_records,
        read_pct=0.0,
        update_pct=0.0,
        scan_pct=0.95,
        insert_pct=0.05,
        **kwargs
    )
    return YCSBWorkload(config)

