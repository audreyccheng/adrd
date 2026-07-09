"""
Buffer Manager - Core buffer pool management with PBM scan tracking.

This is the main interface for buffer operations:
- read_buffer(): Load a page into buffer pool
- Pluggable replacement policy with access to scan tracking
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from .buffer_descriptor import BufferDescriptor
from .buffer_tag import normalize_tag
from .block_group import BlockGroupRegistry


@dataclass
class BufferStats:
    """
    Statistics for buffer manager performance with optional latency tracking.
    
    Latency model grounded on postgres-pbm thesis results:
    - PBM-sampling achieves up to 38% I/O reduction over clock-sweep
    - Sequential workloads: ~60% hit rate with PBM vs ~45% clock-sweep
    - TPC-H mixed: ~82-84% hit rate with freq stats
    
    Latency constants based on real NVMe SSD measurements.
    
    Extended with:
    - WAL flush cost modeling (EVOLVE_WAL_COST=1)
    - Re-eviction tracking (EVOLVE_REEVICT=1)
    """
    buffer_hits: int = 0
    buffer_misses: int = 0
    buffer_evictions: int = 0
    buffer_writes: int = 0  # Dirty pages written at eviction (sync)
    
    # I/O volume tracking (key metric for PBM)
    bytes_read: int = 0
    bytes_written: int = 0
    
    # Overhead tracking (P4: explains why COMBINED is faster than SAMPLING)
    total_inspections: int = 0  # Buffers checked during eviction
    eviction_calls: int = 0
    estimator_calls: int = 0    # Expensive next-access predictions
    fast_path_successes: int = 0  # Evictions handled by fast paths (O(1))
    
    BLOCK_SIZE: int = field(default=8192, repr=False)  # 8KB
    
    # === LATENCY TRACKING (enabled via latency_enabled flag) ===
    latency_enabled: bool = field(default=False, repr=False)
    
    # Latency accumulators (nanoseconds)
    total_access_latency_ns: int = 0
    total_eviction_latency_ns: int = 0
    
    # Dirty page tracking for latency
    sync_dirty_writes: int = 0    # Dirty pages written synchronously at eviction
    async_dirty_writes: int = 0   # Dirty pages pre-written by simulated bgwriter
    
    # Sequential vs random I/O tracking
    sequential_reads: int = 0
    random_reads: int = 0
    
    # === WAL FLUSH COST TRACKING (enabled via wal_cost_enabled flag) ===
    # Models PostgreSQL's XLogNeedsFlush() check before dirty page write
    # Recently dirtied pages likely need WAL flush = extra latency
    wal_cost_enabled: bool = field(default=False, repr=False)
    wal_flush_evictions: int = 0  # Dirty evictions that needed WAL flush
    
    # === RE-EVICTION TRACKING (enabled via reeviction_enabled flag) ===
    # Tracks when we evict a page that gets re-loaded soon (bad decision)
    # This is a direct measure of prediction quality
    reeviction_enabled: bool = field(default=False, repr=False)
    re_evictions: int = 0  # Pages re-loaded within RE_EVICTION_WINDOW after eviction
    # Recently evicted pages: tag -> eviction_time (managed by BufferManager)
    # Note: The dict is managed externally to avoid dataclass complexity
    
    # === LATENCY CONSTANTS (grounded on real NVMe SSD measurements) ===
    # From postgres-pbm thesis context: real experiments on Intel S3700 SSD
    MEMORY_HIT_NS: int = field(default=100, repr=False)         # ~0.1μs - L3 cache miss to DRAM
    SSD_RANDOM_READ_NS: int = field(default=100_000, repr=False)  # ~100μs - NVMe random read
    SSD_RANDOM_WRITE_NS: int = field(default=200_000, repr=False) # ~200μs - NVMe random write
    SSD_SEQ_READ_NS: int = field(default=20_000, repr=False)      # ~20μs - Sequential (prefetch)
    
    # Background writer efficiency
    # Default 0.30 = stress test scenario (bgwriter under load)
    # Real PostgreSQL: 0.60-0.80 when bgwriter keeps up
    # Set lower to make dirty penalty more impactful for evolution
    BGWRITER_COVERAGE: float = field(default=0.30, repr=False)  # 30% = stress test
    
    # === WAL FLUSH CONSTANTS ===
    # WAL flush is expensive: ~200-500μs to sync WAL to disk
    # Pages dirtied recently (< WAL_FLUSH_WINDOW) likely need WAL flush
    WAL_FLUSH_NS: int = field(default=300_000, repr=False)  # ~300μs for WAL sync
    WAL_FLUSH_WINDOW: float = field(default=0.010, repr=False)  # 10ms - pages dirtied within this window likely need WAL flush
    
    # === RE-EVICTION CONSTANTS ===
    # If a page is re-loaded within this window after eviction, it was a bad decision
    RE_EVICTION_WINDOW: float = field(default=0.100, repr=False)  # 100ms
    # Penalty for bad eviction: represents wasted I/O work
    # The reload is already charged as a cache miss, this is EXTRA penalty
    # for the prediction failure (wasted eviction + potential dirty write)
    RE_EVICTION_PENALTY_NS: int = field(default=50_000, repr=False)  # 50μs extra penalty
    
    @property
    def total_accesses(self) -> int:
        return self.buffer_hits + self.buffer_misses
    
    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.buffer_hits / self.total_accesses
    
    @property
    def io_volume_mb(self) -> float:
        """Total I/O volume in megabytes."""
        return (self.bytes_read + self.bytes_written) / (1024 * 1024)
    
    @property
    def avg_inspections_per_eviction(self) -> float:
        if self.eviction_calls == 0:
            return 0.0
        return self.total_inspections / self.eviction_calls
    
    @property
    def avg_estimator_calls_per_eviction(self) -> float:
        """Average expensive estimator calls per eviction - key overhead metric."""
        if self.eviction_calls == 0:
            return 0.0
        return self.estimator_calls / self.eviction_calls
    
    @property
    def fast_path_rate(self) -> float:
        """Fraction of evictions handled by O(1) fast paths."""
        if self.eviction_calls == 0:
            return 0.0
        return self.fast_path_successes / self.eviction_calls
    
    # === LATENCY PROPERTIES ===
    @property
    def avg_access_latency_ns(self) -> float:
        """Average latency per buffer access - KEY METRIC for realistic scoring."""
        if self.total_accesses == 0:
            return 0.0
        return self.total_access_latency_ns / self.total_accesses
    
    @property
    def avg_access_latency_us(self) -> float:
        """Average latency in microseconds."""
        return self.avg_access_latency_ns / 1000.0
    
    @property
    def effective_throughput(self) -> float:
        """Theoretical accesses per second based on latency."""
        if self.avg_access_latency_ns == 0:
            return float('inf')
        return 1e9 / self.avg_access_latency_ns
    
    @property
    def dirty_eviction_rate(self) -> float:
        """Fraction of evictions that required synchronous write (not pre-written by bgwriter)."""
        total_dirty = self.sync_dirty_writes + self.async_dirty_writes
        if total_dirty == 0:
            return 0.0
        return self.sync_dirty_writes / total_dirty
    
    @property
    def sync_dirty_rate(self) -> float:
        """Fraction of all evictions that caused sync dirty writes."""
        if self.buffer_evictions == 0:
            return 0.0
        return self.sync_dirty_writes / self.buffer_evictions
    
    # === WAL FLUSH PROPERTIES ===
    @property
    def wal_flush_rate(self) -> float:
        """Fraction of dirty evictions that required WAL flush."""
        total_dirty = self.sync_dirty_writes + self.async_dirty_writes
        if total_dirty == 0:
            return 0.0
        return self.wal_flush_evictions / total_dirty
    
    # === RE-EVICTION PROPERTIES ===
    @property
    def re_eviction_rate(self) -> float:
        """Fraction of evictions that were re-loaded within window (bad decisions)."""
        if self.buffer_evictions == 0:
            return 0.0
        return self.re_evictions / self.buffer_evictions
    
    @property 
    def latency_breakdown(self) -> dict:
        """Where time is spent (percentage breakdown)."""
        total = self.total_access_latency_ns
        if total == 0:
            return {"hit_pct": 0, "read_pct": 0, "dirty_write_pct": 0}
        
        hit_latency = self.buffer_hits * self.MEMORY_HIT_NS
        read_latency = (self.sequential_reads * self.SSD_SEQ_READ_NS + 
                       self.random_reads * self.SSD_RANDOM_READ_NS)
        dirty_latency = self.sync_dirty_writes * self.SSD_RANDOM_WRITE_NS
        
        return {
            "hit_pct": hit_latency / total * 100,
            "read_pct": read_latency / total * 100, 
            "dirty_write_pct": dirty_latency / total * 100,
        }
    
    @property
    def latency_score(self) -> float:
        """
        Latency score (higher is better, NO CAP).
        
        Score = baseline_latency / actual_latency
        - Score 1.0 = baseline performance (Clock-sweep typical)
        - Score 2.0 = 2x better than baseline
        - Score 0.5 = 2x worse than baseline
        
        Baseline calibrated for simulator workloads:
        - TPC-H with sync scans: ~30% miss rate, mostly sequential
        - TPC-C: ~50% miss rate, random + dirty pages
        - Weighted average baseline: ~30μs
        """
        if not self.latency_enabled or self.total_accesses == 0:
            return self.hit_rate  # Fallback to hit rate
        
        # Baseline: Clock-sweep typical performance in our simulator
        # TPC-H: 30% miss * 50μs seq read = 15μs + 0.1μs hits ≈ 15μs
        # TPC-C: 50% miss * 100μs random + dirty = 50-60μs
        # Combined baseline: ~30μs represents "average" Clock-sweep performance
        baseline_avg_latency_ns = 30_000  # 30μs
        
        # Score = baseline / actual (no cap!)
        # Lower actual latency = higher score
        return baseline_avg_latency_ns / max(self.avg_access_latency_ns, 1)
    
    def record_hit(self):
        """Record a buffer hit."""
        self.buffer_hits += 1
        if self.latency_enabled:
            self.total_access_latency_ns += self.MEMORY_HIT_NS
    
    def record_miss(self, is_sequential: bool = False):
        """Record a buffer miss with optional sequential hint."""
        self.buffer_misses += 1
        self.bytes_read += self.BLOCK_SIZE
        
        if self.latency_enabled:
            if is_sequential:
                self.sequential_reads += 1
                self.total_access_latency_ns += self.SSD_SEQ_READ_NS
            else:
                self.random_reads += 1
                self.total_access_latency_ns += self.SSD_RANDOM_READ_NS
    
    def record_eviction(self, is_dirty: bool, inspections: int = 1, 
                        dirty_time: float = 0.0, current_time: float = 0.0):
        """
        Record a buffer eviction.
        
        For dirty pages, simulates PostgreSQL's background writer:
        - bgwriter pre-writes ~65% of dirty pages (no latency hit)
        - Remaining ~35% require sync write at eviction time
        
        Extended with WAL flush cost modeling:
        - Recently dirtied pages (< WAL_FLUSH_WINDOW) likely need WAL flush
        - WAL flush adds ~300μs latency
        
        Args:
            is_dirty: Whether the evicted page was dirty
            inspections: Number of buffers inspected during eviction
            dirty_time: When the page became dirty (0 = clean or unknown)
            current_time: Current simulated time
        """
        self.buffer_evictions += 1
        self.eviction_calls += 1
        self.total_inspections += inspections
        
        if is_dirty:
            self.buffer_writes += 1
            self.bytes_written += self.BLOCK_SIZE
            
            if self.latency_enabled:
                # Simulate bgwriter: probabilistically pre-writes dirty pages
                import random
                if random.random() < self.BGWRITER_COVERAGE:
                    # bgwriter got to it - no latency impact
                    self.async_dirty_writes += 1
                else:
                    # Sync write required at eviction time
                    self.sync_dirty_writes += 1
                    self.total_access_latency_ns += self.SSD_RANDOM_WRITE_NS
                    self.total_eviction_latency_ns += self.SSD_RANDOM_WRITE_NS
                    
                    # === WAL FLUSH COST ===
                    # If page was dirtied recently, WAL likely not flushed yet
                    # PostgreSQL calls XLogNeedsFlush() and may need XLogFlush()
                    if self.wal_cost_enabled and dirty_time > 0:
                        time_since_dirty = current_time - dirty_time
                        if time_since_dirty < self.WAL_FLUSH_WINDOW:
                            # Recently dirtied - need WAL flush
                            self.wal_flush_evictions += 1
                            self.total_access_latency_ns += self.WAL_FLUSH_NS
                            self.total_eviction_latency_ns += self.WAL_FLUSH_NS
    
    def record_re_eviction(self):
        """
        Record that a recently evicted page was re-loaded (bad eviction decision).
        Called by BufferManager when it detects a re-eviction.
        
        This adds a latency penalty to incentivize evolution to minimize re-evictions.
        The reload itself is already charged as a cache miss - this is EXTRA penalty
        for the wasted work of evicting a page that was needed soon after.
        """
        self.re_evictions += 1
        # KEY FIX: Add latency cost so evolution has incentive to minimize re-evictions
        if self.reeviction_enabled:
            self.total_access_latency_ns += self.RE_EVICTION_PENALTY_NS
    
    def reset(self):
        """Reset all statistics."""
        self.buffer_hits = 0
        self.buffer_misses = 0
        self.buffer_evictions = 0
        self.buffer_writes = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.total_inspections = 0
        self.eviction_calls = 0
        # Latency fields
        self.total_access_latency_ns = 0
        self.total_eviction_latency_ns = 0
        self.sync_dirty_writes = 0
        self.async_dirty_writes = 0
        self.sequential_reads = 0
        self.random_reads = 0
        # WAL flush fields
        self.wal_flush_evictions = 0
        # Re-eviction fields
        self.re_evictions = 0
    
    def get_latency_summary(self) -> dict:
        """Get a summary of latency metrics for reporting."""
        summary = {
            "avg_latency_us": round(self.avg_access_latency_us, 2),
            "hit_rate": round(self.hit_rate, 4),
            "latency_score": round(self.latency_score, 4),
            "sync_dirty_rate": round(self.sync_dirty_rate, 4),
            "dirty_eviction_rate": round(self.dirty_eviction_rate, 4),
            "seq_read_pct": round(self.sequential_reads / max(1, self.buffer_misses) * 100, 1),
            "breakdown": self.latency_breakdown,
        }
        # Add WAL flush metrics if enabled
        if self.wal_cost_enabled:
            summary["wal_flush_rate"] = round(self.wal_flush_rate, 4)
            summary["wal_flush_evictions"] = self.wal_flush_evictions
        # Add re-eviction metrics if enabled
        if self.reeviction_enabled:
            summary["re_eviction_rate"] = round(self.re_eviction_rate, 4)
            summary["re_evictions"] = self.re_evictions
        return summary
    
    def __repr__(self):
        base = (f"BufferStats(hits={self.buffer_hits}, misses={self.buffer_misses}, "
                f"hit_rate={self.hit_rate:.4f}, io_mb={self.io_volume_mb:.1f}")
        if self.latency_enabled:
            base += f", avg_latency={self.avg_access_latency_us:.1f}μs, latency_score={self.latency_score:.4f}"
        return base + ")"


# Type for replacement policy function
# Args: (buffers, buffer_table, estimator, scan_context)
# Returns: buffer_id to evict, or None for fallback
ReplacementPolicy = Callable[[List[BufferDescriptor], Dict, Any, Any], Optional[int]]


class BufferManager:
    """
    Buffer manager with PBM-style scan tracking support.
    
    Key features:
    - Fixed-size buffer pool
    - Hash table for O(1) tag lookup
    - Block group registry for scan tracking
    - Pluggable replacement policy with access to estimator
    - Optional latency metrics (grounded on postgres-pbm thesis results)
    
    Extended with:
    - WAL flush cost modeling (wal_cost_enabled)
    - Re-eviction tracking (reeviction_enabled)
    """
    
    def __init__(self, num_buffers: int, replacement_policy: ReplacementPolicy = None,
                 latency_enabled: bool = False, wal_cost_enabled: bool = False,
                 reeviction_enabled: bool = False):
        self.num_buffers = num_buffers
        
        # Core data structures
        self.buffers = [BufferDescriptor(i) for i in range(num_buffers)]
        self.buffer_table: Dict[tuple, int] = {}  # tag -> buf_id
        
        # Scan tracking infrastructure
        self.block_groups = BlockGroupRegistry()
        
        # These will be set by set_scan_tracker()
        self._scan_registry = None
        self._estimator = None
        
        # Replacement policy
        self._replacement_policy = replacement_policy
        self._clock_hand = 0  # For fallback clock sweep
        
        # Statistics (with latency and extended tracking)
        self.stats = BufferStats(
            latency_enabled=latency_enabled,
            wal_cost_enabled=wal_cost_enabled,
            reeviction_enabled=reeviction_enabled
        )
        self._latency_enabled = latency_enabled
        self._wal_cost_enabled = wal_cost_enabled
        self._reeviction_enabled = reeviction_enabled
        
        # Current time (for frequency tracking)
        self._current_time = 0.0
        
        # Track last access per relation for sequential detection
        self._last_access: Dict[int, int] = {}  # relation_id -> last_block
        
        # Re-eviction tracking: recently evicted pages
        # tag -> eviction_time (for detecting bad eviction decisions)
        self._recently_evicted: Dict[tuple, float] = {}
    
    def set_scan_tracker(self, scan_registry, estimator):
        """
        Set up scan tracking infrastructure.
        
        Called by evaluator after creating ScanRegistry and NextAccessEstimator.
        """
        self._scan_registry = scan_registry
        self._estimator = estimator
    
    def set_time(self, t: float):
        """Set current simulated time."""
        self._current_time = t
        if self._scan_registry:
            self._scan_registry.set_time(t)
    
    def read_buffer(self, tag, scan_context: Any = None, is_write: bool = False) -> int:
        """
        Read a page into buffer pool.
        
        Args:
            tag: Page identifier (relation_id, block_num)
            scan_context: Optional scan context for tracking
            is_write: Whether this is a write access
        
        Returns:
            buffer_id where the page is located
        """
        tag = normalize_tag(tag)
        relation_id, block_num = tag
        
        # Detect sequential access for latency modeling
        # (Sequential reads benefit from OS prefetch: ~20μs vs ~100μs random)
        is_sequential = False
        if self._latency_enabled:
            last_block = self._last_access.get(relation_id, -999)
            # Sequential if within 8 blocks of last access (like OS readahead)
            is_sequential = (0 < block_num - last_block <= 8)
            self._last_access[relation_id] = block_num
        
        # Check if already in buffer pool
        if tag in self.buffer_table:
            buf_id = self.buffer_table[tag]
            buf = self.buffers[buf_id]
            
            # Pin and update stats
            buf.pin()
            buf.increment_usage()
            buf.access_count += 1
            buf.last_access_time = self._current_time
            
            if is_write:
                # Track when page first becomes dirty (for WAL flush cost)
                if not buf.is_dirty:
                    buf.dirty_time = self._current_time
                buf.is_dirty = True
            
            self.stats.record_hit()
            return buf_id
        
        # Cache miss - need to allocate buffer
        self.stats.record_miss(is_sequential=is_sequential)
        
        # === RE-EVICTION DETECTION ===
        # If this page was recently evicted, it was a bad eviction decision
        if self._reeviction_enabled and tag in self._recently_evicted:
            eviction_time = self._recently_evicted[tag]
            if self._current_time - eviction_time < self.stats.RE_EVICTION_WINDOW:
                self.stats.record_re_eviction()
            # Clean up old entry
            del self._recently_evicted[tag]
        
        # Find victim buffer
        buf_id, inspections = self._allocate_buffer(scan_context)
        buf = self.buffers[buf_id]
        
        # Evict old page if necessary
        if buf.tag is not None:
            old_tag = buf.tag
            
            # Remove from buffer table
            del self.buffer_table[old_tag]
            
            # Remove from block group tracking
            old_bg = self.block_groups.get(old_tag[0], old_tag[1])
            if old_bg:
                old_bg.remove_buffer(buf_id)
            
            # Record eviction with dirty_time for WAL flush cost
            self.stats.record_eviction(
                buf.is_dirty, 
                inspections,
                dirty_time=buf.dirty_time if buf.is_dirty else 0.0,
                current_time=self._current_time
            )
            
            # === TRACK RECENTLY EVICTED ===
            # For re-eviction detection
            if self._reeviction_enabled:
                self._recently_evicted[old_tag] = self._current_time
                # Periodically clean up old entries to avoid memory growth
                if len(self._recently_evicted) > self.num_buffers * 2:
                    self._cleanup_recently_evicted()
        
        # Load new page
        buf.reset()
        buf.tag = tag
        buf.pin()
        buf.increment_usage()
        buf.access_count = 1
        buf.last_access_time = self._current_time
        
        if is_write:
            buf.is_dirty = True
            buf.dirty_time = self._current_time  # Track when dirtied
        
        # Add to buffer table
        self.buffer_table[tag] = buf_id
        
        # Link to block group
        new_bg = self.block_groups.get_or_create(tag[0], tag[1])
        new_bg.add_buffer(buf_id)
        buf.block_group = new_bg
        
        return buf_id
    
    def _cleanup_recently_evicted(self):
        """Remove old entries from recently evicted tracking."""
        cutoff = self._current_time - self.stats.RE_EVICTION_WINDOW
        self._recently_evicted = {
            tag: evict_time 
            for tag, evict_time in self._recently_evicted.items()
            if evict_time > cutoff
        }
    
    def unpin_buffer(self, buf_id: int):
        """Unpin a buffer after use."""
        if 0 <= buf_id < self.num_buffers:
            self.buffers[buf_id].unpin()
    
    def _allocate_buffer(self, scan_context: Any) -> tuple:
        """
        Find a buffer to evict.
        
        Returns: (buf_id, num_inspections)
        """
        # Try custom replacement policy first
        if self._replacement_policy and self._estimator:
            try:
                victim = self._replacement_policy(
                    self.buffers,
                    self.buffer_table,
                    self._estimator,
                    scan_context
                )
                if victim is not None and 0 <= victim < self.num_buffers:
                    if self.buffers[victim].can_evict():
                        return (victim, 1)
            except Exception:
                pass  # Fall back to clock sweep
        
        # Fallback: clock sweep
        return self._clock_sweep()
    
    def _clock_sweep(self) -> tuple:
        """
        Standard PostgreSQL clock sweep algorithm.
        
        Returns: (buf_id, num_inspections)
        """
        inspections = 0
        max_inspections = self.num_buffers * 2
        
        while inspections < max_inspections:
            buf = self.buffers[self._clock_hand]
            self._clock_hand = (self._clock_hand + 1) % self.num_buffers
            inspections += 1
            
            # Skip pinned buffers
            if not buf.can_evict():
                continue
            
            # If usage_count is 0, evict
            if buf.usage_count == 0:
                return (buf.buf_id, inspections)
            
            # Decrement usage count
            buf.decrement_usage()
        
        # Emergency: find any unpinned buffer
        for buf in self.buffers:
            if buf.can_evict():
                return (buf.buf_id, inspections)
        
        raise RuntimeError("No unpinned buffers available")
    
    def get_buffer(self, buf_id: int) -> BufferDescriptor:
        """Get buffer descriptor by ID."""
        return self.buffers[buf_id]
    
    def reset(self):
        """Reset buffer manager to initial state."""
        for buf in self.buffers:
            buf.reset()
        self.buffer_table.clear()
        self.block_groups.clear()
        self.stats.reset()
        self._clock_hand = 0
        self._current_time = 0.0
        self._last_access.clear()
        self._recently_evicted.clear()

