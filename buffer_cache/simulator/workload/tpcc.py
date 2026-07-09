"""
TPC-C Workload Generator for Buffer Simulator.

Implements all 5 TPC-C transaction types with realistic access patterns:
- NewOrder (45%): OLTP insert-heavy, touches many tables
- Payment (43%): Update-heavy, hot warehouse/district
- OrderStatus (4%): Read-only customer lookup
- Delivery (4%): Batch update of orders
- StockLevel (4%): Read-heavy, range scan

Key characteristics:
- Random point lookups (not sequential scans like TPC-H)
- Hot/cold data: warehouse/district always hot
- 100 warehouses = ~10GB data
"""
import random
from dataclasses import dataclass
from typing import Generator, Dict, Set
from enum import IntEnum

from .base import WorkloadAccess, ScanContext, AccessType, BaseWorkload


class TPCCTable(IntEnum):
    """TPC-C tables with their relation IDs."""
    WAREHOUSE = 0
    DISTRICT = 1
    CUSTOMER = 2
    ITEM = 3
    STOCK = 4
    ORDERS = 5
    ORDER_LINE = 6
    NEW_ORDER = 7
    HISTORY = 8


# TPC-C table sizes at 100 warehouses (in 8KB blocks)
# Based on standard TPC-C row sizes and PostgreSQL storage
def get_table_sizes(num_warehouses: int) -> Dict[int, int]:
    """Calculate table sizes based on warehouse count."""
    return {
        TPCCTable.WAREHOUSE: max(1, num_warehouses // 100),  # ~1 row per block
        TPCCTable.DISTRICT: max(1, num_warehouses * 10 // 50),  # 10 districts per warehouse
        TPCCTable.CUSTOMER: num_warehouses * 3000 // 10,  # 30K customers per warehouse
        TPCCTable.ITEM: 10000,  # Fixed 100K items, ~10K blocks
        TPCCTable.STOCK: num_warehouses * 10000,  # 100K stock records per warehouse
        TPCCTable.ORDERS: num_warehouses * 3000,  # ~30K orders per warehouse
        TPCCTable.ORDER_LINE: num_warehouses * 30000,  # ~300K order lines per warehouse
        TPCCTable.NEW_ORDER: max(100, num_warehouses * 90),  # ~900 new orders per warehouse
        TPCCTable.HISTORY: num_warehouses * 3000,  # ~30K history per warehouse
    }


@dataclass
class TPCCConfig:
    """Configuration for TPC-C workload."""
    num_warehouses: int = 100
    num_terminals: int = 10  # Concurrent terminals per warehouse
    transactions_per_terminal: int = 100
    buffer_pool_blocks: int = 131072  # 1 GB default
    
    # Transaction mix (standard TPC-C)
    new_order_pct: float = 0.45
    payment_pct: float = 0.43
    order_status_pct: float = 0.04
    delivery_pct: float = 0.04
    stock_level_pct: float = 0.04
    
    # Access timing
    time_per_block: float = 0.0001


class TPCCWorkload(BaseWorkload):
    """
    TPC-C workload with all 5 transaction types.
    
    Access patterns:
    - Point lookups via primary key (modeled as single block access)
    - Index range scans (modeled as small sequential access)
    - Hot data: warehouse/district accessed frequently
    """
    
    def __init__(self, config: TPCCConfig = None):
        super().__init__(name="tpcc")
        self.config = config or TPCCConfig()
        self.table_sizes = get_table_sizes(self.config.num_warehouses)
        self._scan_id = 0
        self._current_time = 0.0
    
    def _next_scan_id(self) -> int:
        sid = self._scan_id
        self._scan_id += 1
        return sid
    
    def _point_lookup(self, table: int, key: int, is_write: bool = False
                     ) -> Generator[WorkloadAccess, None, None]:
        """Single block point lookup (index access + heap access)."""
        table_size = self.table_sizes.get(table, 1000)
        block = key % table_size
        
        yield WorkloadAccess(
            tag=(table, block),
            access_type=AccessType.INDEX,
            scan_context=None,
            is_write=is_write,
        )
        self._current_time += self.config.time_per_block
    
    def _range_scan(self, table: int, start_key: int, count: int,
                   is_write: bool = False) -> Generator[WorkloadAccess, None, None]:
        """Small range scan (e.g., order lines for an order)."""
        table_size = self.table_sizes.get(table, 1000)
        scan_id = self._next_scan_id()
        
        start_block = start_key % table_size
        blocks_to_scan = min(count, table_size - start_block)
        
        for i in range(blocks_to_scan):
            block = start_block + i
            ctx = ScanContext(
                scan_id=scan_id,
                relation_id=table,
                access_type=AccessType.SEQUENTIAL,
                current_position=block,
                total_blocks=blocks_to_scan,
                blocks_remaining=blocks_to_scan - i - 1,
                est_blocks_per_sec=10000.0,
            )
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.SEQUENTIAL,
                scan_context=ctx,
                is_write=is_write,
            )
            self._current_time += self.config.time_per_block
    
    def _new_order_txn(self, w_id: int) -> Generator[WorkloadAccess, None, None]:
        """
        NewOrder transaction (45% of mix):
        - Read warehouse
        - Read district, update next_o_id (WRITE)
        - Read customer
        - Read items (5-15 random)
        - Read/update stock (5-15 random) (WRITE)
        - Insert order, order_line, new_order (WRITE)
        """
        d_id = random.randint(1, 10)
        c_id = random.randint(1, 3000)
        num_items = random.randint(5, 15)
        
        # Warehouse lookup (always hot)
        yield from self._point_lookup(TPCCTable.WAREHOUSE, w_id)
        
        # District lookup (hot) + update next_o_id
        yield from self._point_lookup(TPCCTable.DISTRICT, w_id * 10 + d_id, is_write=True)
        
        # Customer lookup
        yield from self._point_lookup(TPCCTable.CUSTOMER, w_id * 30000 + d_id * 3000 + c_id)
        
        # Item and stock lookups/updates (random)
        for _ in range(num_items):
            item_id = random.randint(1, 100000)
            yield from self._point_lookup(TPCCTable.ITEM, item_id)
            yield from self._point_lookup(TPCCTable.STOCK, w_id * 100000 + item_id, is_write=True)
        
        # Insert order (spread across table for CH-benchmark OLAP overlap)
        order_block = random.randint(0, self.table_sizes[TPCCTable.ORDERS] - 1)
        yield from self._point_lookup(TPCCTable.ORDERS, order_block, is_write=True)
        
        # Insert order_lines (spread across table for CH-benchmark OLAP overlap)
        ol_block = random.randint(0, self.table_sizes[TPCCTable.ORDER_LINE] - num_items)
        yield from self._range_scan(TPCCTable.ORDER_LINE, ol_block, num_items, is_write=True)
        
        # Insert new_order
        yield from self._point_lookup(TPCCTable.NEW_ORDER, 
                                      w_id * 900 + random.randint(1, 100), is_write=True)
    
    def _payment_txn(self, w_id: int) -> Generator[WorkloadAccess, None, None]:
        """
        Payment transaction (43% of mix):
        - Read/update warehouse (WRITE)
        - Read/update district (WRITE)
        - Read/update customer (by name or id) (WRITE)
        - Insert history (WRITE)
        """
        d_id = random.randint(1, 10)
        c_id = random.randint(1, 3000)
        
        # Warehouse (always hot) - update YTD
        yield from self._point_lookup(TPCCTable.WAREHOUSE, w_id, is_write=True)
        
        # District (hot) - update YTD
        yield from self._point_lookup(TPCCTable.DISTRICT, w_id * 10 + d_id, is_write=True)
        
        # Customer lookup/update (may scan by name - model as 2-3 block access)
        if random.random() < 0.6:  # 60% by last name
            yield from self._range_scan(TPCCTable.CUSTOMER, 
                                       w_id * 30000 + d_id * 3000 + random.randint(1, 2900), 3)
            # Update customer after scan
            yield from self._point_lookup(TPCCTable.CUSTOMER, 
                                         w_id * 30000 + d_id * 3000 + c_id, is_write=True)
        else:
            yield from self._point_lookup(TPCCTable.CUSTOMER, 
                                         w_id * 30000 + d_id * 3000 + c_id, is_write=True)
        
        # Insert history (spread across table for CH-benchmark OLAP overlap)
        yield from self._point_lookup(TPCCTable.HISTORY, 
                                      random.randint(0, self.table_sizes[TPCCTable.HISTORY] - 1), is_write=True)
    
    def _order_status_txn(self, w_id: int) -> Generator[WorkloadAccess, None, None]:
        """
        OrderStatus transaction (4% of mix) - Read only:
        - Read customer (by name or id)
        - Read most recent order
        - Read order lines
        """
        d_id = random.randint(1, 10)
        c_id = random.randint(1, 3000)
        
        # Customer lookup
        if random.random() < 0.6:
            yield from self._range_scan(TPCCTable.CUSTOMER,
                                       w_id * 30000 + d_id * 3000 + random.randint(1, 2900), 3)
        else:
            yield from self._point_lookup(TPCCTable.CUSTOMER,
                                         w_id * 30000 + d_id * 3000 + c_id)
        
        # Most recent order (near end of table)
        yield from self._point_lookup(TPCCTable.ORDERS,
                                      self.table_sizes[TPCCTable.ORDERS] - random.randint(1, 1000))
        
        # Order lines (5-15 lines)
        yield from self._range_scan(TPCCTable.ORDER_LINE,
                                   self.table_sizes[TPCCTable.ORDER_LINE] - random.randint(1, 5000),
                                   random.randint(5, 15))
    
    def _delivery_txn(self, w_id: int) -> Generator[WorkloadAccess, None, None]:
        """
        Delivery transaction (4% of mix) - Batch update:
        - For each district: read/delete oldest new_order
        - Update order
        - Update order_lines
        - Update customer balance
        """
        for d_id in range(1, 11):
            # Read/delete new_order (oldest - start of table region)
            yield from self._point_lookup(TPCCTable.NEW_ORDER, w_id * 900 + d_id * 90)
            
            # Update order
            yield from self._point_lookup(TPCCTable.ORDERS,
                                         w_id * 30000 + d_id * 3000 + random.randint(1, 2000))
            
            # Update order_lines (spread across table for CH-benchmark OLAP overlap)
            yield from self._range_scan(TPCCTable.ORDER_LINE,
                                       random.randint(0, self.table_sizes[TPCCTable.ORDER_LINE] - 15),
                                       random.randint(5, 15), is_write=True)
            
            # Update customer
            yield from self._point_lookup(TPCCTable.CUSTOMER,
                                         w_id * 30000 + d_id * 3000 + random.randint(1, 3000))
    
    def _stock_level_txn(self, w_id: int) -> Generator[WorkloadAccess, None, None]:
        """
        StockLevel transaction (4% of mix) - Read heavy:
        - Read district
        - Read recent order_lines (range scan)
        - Read stock for each item (many point lookups)
        """
        d_id = random.randint(1, 10)
        
        # District
        yield from self._point_lookup(TPCCTable.DISTRICT, w_id * 10 + d_id)
        
        # Recent order_lines (last 20 orders worth)
        num_order_lines = random.randint(100, 200)
        yield from self._range_scan(TPCCTable.ORDER_LINE,
                                   self.table_sizes[TPCCTable.ORDER_LINE] - num_order_lines,
                                   num_order_lines)
        
        # Stock lookups (unique items from order_lines)
        num_stock_checks = random.randint(20, 50)
        for _ in range(num_stock_checks):
            item_id = random.randint(1, 100000)
            yield from self._point_lookup(TPCCTable.STOCK, w_id * 100000 + item_id)
    
    def _generate_terminal_stream(self, terminal_id: int) -> Generator[WorkloadAccess, None, None]:
        """Generate transactions for one terminal."""
        w_id = (terminal_id % self.config.num_warehouses) + 1
        
        for _ in range(self.config.transactions_per_terminal):
            # Choose transaction type based on mix
            r = random.random()
            if r < self.config.new_order_pct:
                yield from self._new_order_txn(w_id)
            elif r < self.config.new_order_pct + self.config.payment_pct:
                yield from self._payment_txn(w_id)
            elif r < self.config.new_order_pct + self.config.payment_pct + self.config.order_status_pct:
                yield from self._order_status_txn(w_id)
            elif r < self.config.new_order_pct + self.config.payment_pct + self.config.order_status_pct + self.config.delivery_pct:
                yield from self._delivery_txn(w_id)
            else:
                yield from self._stock_level_txn(w_id)
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """Generate interleaved workload from multiple terminals."""
        self._scan_id = 0
        self._current_time = 0.0
        
        num_terminals = self.config.num_warehouses * self.config.num_terminals
        streams = [self._generate_terminal_stream(i) for i in range(num_terminals)]
        active = list(range(len(streams)))
        
        # Round-robin interleaving
        while active:
            for idx in list(active):
                try:
                    access = next(streams[idx])
                    yield access
                except StopIteration:
                    active.remove(idx)
    
    def generate_single_terminal(self) -> Generator[WorkloadAccess, None, None]:
        """Generate single terminal stream (for simpler testing)."""
        self._scan_id = 0
        self._current_time = 0.0
        yield from self._generate_terminal_stream(0)


def create_tpcc_workload(
    num_warehouses: int = 100,
    num_terminals: int = 10,
    transactions_per_terminal: int = 100,
    buffer_pool_mb: int = 1024,
) -> TPCCWorkload:
    """Factory function for TPC-C workload."""
    config = TPCCConfig(
        num_warehouses=num_warehouses,
        num_terminals=num_terminals,
        transactions_per_terminal=transactions_per_terminal,
        buffer_pool_blocks=buffer_pool_mb * 128,
    )
    return TPCCWorkload(config)

