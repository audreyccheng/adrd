"""
Full TPC-H Workload from PBM Thesis Section 6.5.

Mix of sequential scans, bitmap scans, and index lookups.
Tests hybrid workloads with multiple access patterns.

Now includes synchronized sequential scan modeling (PostgreSQL's synchronize_seqscans=on).
"""
import random
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Optional
from enum import IntEnum

from .base import WorkloadAccess, ScanContext, AccessType, BaseWorkload


class SynchronizedScanManager:
    """
    Models PostgreSQL's synchronized sequential scans (synchronize_seqscans=on).
    
    When a new sequential scan starts on a table that already has an active scan,
    the new scan joins at the current position instead of starting from block 0.
    This dramatically increases buffer sharing between concurrent scans.
    
    PostgreSQL behavior:
    - Stream 1: blocks 0, 1, 2, 3, ... 100
    - Stream 2 joins at block 50: blocks 50, 51, ... 100, 0, 1, ... 49 (wraps around)
    
    This explains why real PostgreSQL CLOCK achieves ~68% hit rate while simulator
    without this feature only achieves ~35% hit rate.
    """
    
    def __init__(self):
        # Maps table_id -> (current_position, total_blocks, active_count)
        self._active_scans: Dict[int, List[int, int, int]] = {}
    
    def start_scan(self, table_id: int, total_blocks: int) -> int:
        """
        Register a new scan and return the starting position.
        
        If there's already an active scan on this table, join at its position.
        Otherwise, start at block 0.
        
        Returns:
            Starting block position for the new scan
        """
        if table_id in self._active_scans:
            # Join existing scan at current position
            pos, total, count = self._active_scans[table_id]
            self._active_scans[table_id] = [pos, total, count + 1]
            return pos
        else:
            # Start new scan at block 0
            self._active_scans[table_id] = [0, total_blocks, 1]
            return 0
    
    def advance_position(self, table_id: int, new_position: int):
        """Update the shared scan position for a table."""
        if table_id in self._active_scans:
            _, total, count = self._active_scans[table_id]
            self._active_scans[table_id] = [new_position, total, count]
    
    def end_scan(self, table_id: int):
        """Mark a scan as complete on this table."""
        if table_id in self._active_scans:
            pos, total, count = self._active_scans[table_id]
            if count <= 1:
                del self._active_scans[table_id]
            else:
                self._active_scans[table_id] = [pos, total, count - 1]
    
    def reset(self):
        """Clear all active scans."""
        self._active_scans.clear()

# TPC-H table sizes at SF10 (in 8KB blocks)
class TPCHTable(IntEnum):
    LINEITEM = 0
    ORDERS = 1
    CUSTOMER = 2
    PART = 3
    PARTSUPP = 4
    SUPPLIER = 5
    NATION = 6
    REGION = 7

TABLE_SIZES: Dict[int, int] = {
    TPCHTable.LINEITEM: 1100000,   # ~8.6 GiB
    TPCHTable.ORDERS: 200000,      # ~1.6 GiB
    TPCHTable.CUSTOMER: 25000,     # ~200 MB
    TPCHTable.PART: 30000,         # ~240 MB
    TPCHTable.PARTSUPP: 150000,    # ~1.2 GiB
    TPCHTable.SUPPLIER: 1500,      # ~12 MB
    TPCHTable.NATION: 10,          # tiny
    TPCHTable.REGION: 5,           # tiny
}


@dataclass
class TPCHConfig:
    """Configuration for TPC-H workload."""
    num_streams: int = 8
    buffer_pool_blocks: int = 327680  # 2.5 GiB
    table_scale: float = 1.0  # Scale factor for table sizes (1.0 = full SF10)
    scan_speed: float = 10000.0
    time_per_block: float = 0.0001
    # P1: Synchronized sequential scans (PostgreSQL's synchronize_seqscans=on)
    # When True, concurrent scans on same table share position (join in progress)
    # This dramatically increases buffer sharing and hit rates for all policies
    synchronized_seqscans: bool = True  # Match real PostgreSQL default


class TPCHWorkload(BaseWorkload):
    """
    Full TPC-H workload with mixed access patterns.
    
    Implements all 16 TPC-H queries (Q1-Q16) with simplified access patterns
    matching typical PostgreSQL execution plans with BRIN indexes.
    
    Now supports synchronized sequential scans (synchronize_seqscans=on) which
    models how PostgreSQL shares scan positions between concurrent scans.
    """
    
    def __init__(self, config: TPCHConfig = None):
        super().__init__(name="tpch_full")
        self.config = config or TPCHConfig()
        self._scan_id = 0
        self._current_time = 0.0
        
        # Apply table scale
        self._table_sizes = {
            k: max(10, int(v * self.config.table_scale))
            for k, v in TABLE_SIZES.items()
        }
        
        # P1: Synchronized scan manager for modeling synchronize_seqscans=on
        self._sync_scan_mgr = SynchronizedScanManager() if self.config.synchronized_seqscans else None
    
    def _next_scan_id(self) -> int:
        sid = self._scan_id
        self._scan_id += 1
        return sid
    
    def _make_context(self, scan_id: int, rel: int, atype: AccessType,
                      position: int, total: int, remaining: int) -> ScanContext:
        return ScanContext(
            scan_id=scan_id,
            relation_id=rel,
            access_type=atype,
            current_position=position,
            total_blocks=total,
            blocks_remaining=remaining,
            est_blocks_per_sec=self.config.scan_speed,
        )
    
    def _sequential_scan(self, table: int, selectivity: float = 1.0
                         ) -> Generator[WorkloadAccess, None, None]:
        """
        Generate a sequential scan on a table.
        
        With synchronized_seqscans=True (default), new scans join existing scans
        at their current position, wrapping around to cover all blocks.
        This models PostgreSQL's synchronize_seqscans=on behavior.
        """
        scan_id = self._next_scan_id()
        table_size = self._table_sizes.get(table, 1000)
        
        if selectivity < 1.0:
            # Sample blocks - synchronized scans don't apply to sampled scans
            num_blocks = int(table_size * selectivity)
            blocks = sorted(random.sample(range(table_size), num_blocks))
        else:
            # Full sequential scan - apply synchronized scan logic
            if self._sync_scan_mgr is not None:
                # P1: Get starting position (may join existing scan)
                start_pos = self._sync_scan_mgr.start_scan(table, table_size)
                # Generate blocks in wrap-around order: start_pos -> end -> 0 -> start_pos-1
                blocks = list(range(start_pos, table_size)) + list(range(0, start_pos))
            else:
                # No sync: always start at 0
                blocks = list(range(table_size))
        
        total = len(blocks)
        for i, block in enumerate(blocks):
            # Update shared position for synchronized scans
            if self._sync_scan_mgr is not None and selectivity >= 1.0:
                self._sync_scan_mgr.advance_position(table, block)
            
            ctx = self._make_context(scan_id, table, AccessType.SEQUENTIAL,
                                     block, total, total - i - 1)
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.SEQUENTIAL,
                scan_context=ctx,
            )
            self._current_time += self.config.time_per_block
        
        # Mark scan as complete
        if self._sync_scan_mgr is not None and selectivity >= 1.0:
            self._sync_scan_mgr.end_scan(table)
    
    def _bitmap_scan(self, table: int, selectivity: float
                     ) -> Generator[WorkloadAccess, None, None]:
        """Generate a BRIN bitmap scan."""
        scan_id = self._next_scan_id()
        table_size = self._table_sizes.get(table, 1000)
        
        # Select ranges (BRIN style)
        num_ranges = table_size // 128
        num_selected = max(1, int(num_ranges * selectivity))
        selected_ranges = random.sample(range(num_ranges), min(num_selected, num_ranges))
        
        bitmap = set()
        for r in selected_ranges:
            for b in range(r * 128, min((r + 1) * 128, table_size)):
                bitmap.add(b)
        
        blocks = sorted(bitmap)
        total = len(blocks)
        
        for i, block in enumerate(blocks):
            ctx = self._make_context(scan_id, table, AccessType.BITMAP,
                                     block, total, total - i - 1)
            ctx.bitmap = bitmap
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.BITMAP,
                scan_context=ctx,
            )
            self._current_time += self.config.time_per_block
    
    def _index_lookup(self, table: int, selectivity: float
                      ) -> Generator[WorkloadAccess, None, None]:
        """Generate index point lookups."""
        table_size = self._table_sizes.get(table, 1000)
        num_lookups = max(1, int(table_size * selectivity))
        blocks = random.sample(range(table_size), min(num_lookups, table_size))
        
        for block in blocks:
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.INDEX,
                scan_context=None,  # Point lookups don't need full context
            )
            self._current_time += self.config.time_per_block * 2  # Index is slower
    
    # =========================================================================
    # TPC-H Queries Q1-Q16 (simplified access patterns)
    # Based on typical PostgreSQL execution plans with BRIN indexes
    # =========================================================================
    
    def _q1_pricing_summary(self) -> Generator[WorkloadAccess, None, None]:
        """Q1: Pricing summary - FULL seq scan on lineitem (filters rows, not blocks)."""
        # Real PostgreSQL reads ALL blocks sequentially, then filters ~60% of ROWS
        # This enables synchronized_seqscans to work when multiple streams run Q1
        yield from self._sequential_scan(TPCHTable.LINEITEM, 1.0)
    
    def _q2_minimum_cost_supplier(self) -> Generator[WorkloadAccess, None, None]:
        """Q2: Minimum cost supplier - joins part, supplier, partsupp, nation, region."""
        yield from self._sequential_scan(TPCHTable.REGION, 1.0)  # tiny
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)  # tiny
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.2)
        yield from self._bitmap_scan(TPCHTable.PARTSUPP, 0.1)
        yield from self._index_lookup(TPCHTable.PART, 0.05)
    
    def _q3_shipping_priority(self) -> Generator[WorkloadAccess, None, None]:
        """Q3: Shipping priority - joins customer, orders, lineitem."""
        yield from self._index_lookup(TPCHTable.CUSTOMER, 0.1)
        yield from self._sequential_scan(TPCHTable.ORDERS, 0.5)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.3)
    
    def _q4_order_priority(self) -> Generator[WorkloadAccess, None, None]:
        """Q4: Order priority checking - orders with lineitem existence check."""
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.25)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.15)
    
    def _q5_local_supplier_volume(self) -> Generator[WorkloadAccess, None, None]:
        """Q5: Local supplier volume - multi-way join."""
        yield from self._sequential_scan(TPCHTable.REGION, 1.0)
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.2)
        yield from self._bitmap_scan(TPCHTable.CUSTOMER, 0.2)
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.15)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.1)
    
    def _q6_forecasting(self) -> Generator[WorkloadAccess, None, None]:
        """Q6: Forecasting revenue change - FULL seq scan on lineitem."""
        # Real PostgreSQL does full sequential scan with date/discount filters
        # This enables synchronized_seqscans to work when multiple streams run Q6
        yield from self._sequential_scan(TPCHTable.LINEITEM, 1.0)
    
    def _q7_volume_shipping(self) -> Generator[WorkloadAccess, None, None]:
        """Q7: Volume shipping - supplier/customer/lineitem join."""
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.1)
        yield from self._bitmap_scan(TPCHTable.CUSTOMER, 0.1)
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.2)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.15)
    
    def _q8_national_market_share(self) -> Generator[WorkloadAccess, None, None]:
        """Q8: National market share - complex multi-way join."""
        yield from self._sequential_scan(TPCHTable.REGION, 1.0)
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._index_lookup(TPCHTable.PART, 0.02)
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.15)
        yield from self._bitmap_scan(TPCHTable.CUSTOMER, 0.15)
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.2)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.1)
    
    def _q9_product_type_profit(self) -> Generator[WorkloadAccess, None, None]:
        """Q9: Product type profit - all major tables."""
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._bitmap_scan(TPCHTable.PART, 0.1)
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.3)
        yield from self._bitmap_scan(TPCHTable.PARTSUPP, 0.2)
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.3)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.2)
    
    def _q10_returned_item_reporting(self) -> Generator[WorkloadAccess, None, None]:
        """Q10: Returned item reporting - customer/orders/lineitem."""
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._bitmap_scan(TPCHTable.CUSTOMER, 0.3)
        yield from self._bitmap_scan(TPCHTable.ORDERS, 0.25)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.1)
    
    def _q11_important_stock(self) -> Generator[WorkloadAccess, None, None]:
        """Q11: Important stock identification - partsupp/supplier/nation."""
        yield from self._sequential_scan(TPCHTable.NATION, 1.0)
        yield from self._bitmap_scan(TPCHTable.SUPPLIER, 0.1)
        yield from self._sequential_scan(TPCHTable.PARTSUPP, 0.5)
    
    def _q12_shipping_modes(self) -> Generator[WorkloadAccess, None, None]:
        """Q12: Shipping modes and order priority - orders/lineitem."""
        yield from self._sequential_scan(TPCHTable.ORDERS, 0.3)
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.25)
    
    def _q13_customer_distribution(self) -> Generator[WorkloadAccess, None, None]:
        """Q13: Customer distribution - customer/orders outer join."""
        yield from self._sequential_scan(TPCHTable.CUSTOMER, 1.0)
        yield from self._sequential_scan(TPCHTable.ORDERS, 0.5)
    
    def _q14_promotion_effect(self) -> Generator[WorkloadAccess, None, None]:
        """Q14: Promotion effect - lineitem/part."""
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.1)
        yield from self._index_lookup(TPCHTable.PART, 0.05)
    
    def _q15_top_supplier(self) -> Generator[WorkloadAccess, None, None]:
        """Q15: Top supplier query - lineitem/supplier with view."""
        yield from self._bitmap_scan(TPCHTable.LINEITEM, 0.25)
        yield from self._sequential_scan(TPCHTable.SUPPLIER, 1.0)
    
    def _q16_parts_supplier_relationship(self) -> Generator[WorkloadAccess, None, None]:
        """Q16: Parts/supplier relationship - partsupp/part with NOT IN."""
        yield from self._sequential_scan(TPCHTable.PART, 0.3)
        yield from self._sequential_scan(TPCHTable.PARTSUPP, 0.4)
        yield from self._index_lookup(TPCHTable.SUPPLIER, 0.05)
    
    def _generate_query_stream(self, stream_id: int) -> Generator[WorkloadAccess, None, None]:
        """Generate a stream of all TPC-H Q1-Q16 queries."""
        # All 16 TPC-H queries
        queries = [
            self._q1_pricing_summary,
            self._q2_minimum_cost_supplier,
            self._q3_shipping_priority,
            self._q4_order_priority,
            self._q5_local_supplier_volume,
            self._q6_forecasting,
            self._q7_volume_shipping,
            self._q8_national_market_share,
            self._q9_product_type_profit,
            self._q10_returned_item_reporting,
            self._q11_important_stock,
            self._q12_shipping_modes,
            self._q13_customer_distribution,
            self._q14_promotion_effect,
            self._q15_top_supplier,
            self._q16_parts_supplier_relationship,
        ]
        
        # Run each query type 2 times per stream (32 queries total)
        for _ in range(2):
            random.shuffle(queries)
            for query_func in queries:
                yield from query_func()
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """Generate interleaved workload from multiple streams."""
        self._scan_id = 0
        self._current_time = 0.0
        
        # Reset synchronized scan manager for fresh run
        if self._sync_scan_mgr is not None:
            self._sync_scan_mgr.reset()
        
        streams = [self._generate_query_stream(i) for i in range(self.config.num_streams)]
        active = list(range(len(streams)))
        
        while active:
            for idx in list(active):
                try:
                    access = next(streams[idx])
                    yield access
                except StopIteration:
                    active.remove(idx)
    
    def generate_single_stream(self) -> Generator[WorkloadAccess, None, None]:
        """Generate a single query stream."""
        self._scan_id = 0
        self._current_time = 0.0
        
        # Reset synchronized scan manager for fresh run
        if self._sync_scan_mgr is not None:
            self._sync_scan_mgr.reset()
        
        yield from self._generate_query_stream(0)


def create_tpch_workload(
    parallelism: int = 8,
    cache_size_gb: float = 2.5,
) -> TPCHWorkload:
    """Factory function for TPC-H workload."""
    config = TPCHConfig(
        num_streams=parallelism,
        buffer_pool_blocks=int(cache_size_gb * 1024 * 1024 / 8),
    )
    return TPCHWorkload(config)

