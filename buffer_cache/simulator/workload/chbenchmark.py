"""
CH-benchmark Workload Generator for Buffer Simulator.

CH-benchmark combines TPC-C (OLTP) with TPC-H-like analytical queries (OLAP)
running on the SAME schema. This creates realistic HTAP buffer contention.

Reference: "From A to E: Analyzing TPC's OLTP Benchmarks" and
"The mixed workload CH-benCHmark" by Florian Funke et al.

Key characteristics:
- TPC-C transactions run continuously (OLTP)
- 22 TPC-H-like analytical queries adapted to TPC-C schema (OLAP)
- Queries scan tables while transactions modify them
- Creates realistic dirty page / scan interactions
"""
import random
from dataclasses import dataclass, field
from typing import Generator, Dict, List, Optional
from enum import IntEnum

from .base import WorkloadAccess, ScanContext, AccessType, BaseWorkload
from .tpcc import TPCCWorkload, TPCCConfig, TPCCTable


class CHTable(IntEnum):
    """CH-benchmark tables (extends TPC-C with 3 new tables)."""
    # TPC-C tables (0-8)
    WAREHOUSE = 0
    DISTRICT = 1
    CUSTOMER = 2
    ITEM = 3
    STOCK = 4
    ORDERS = 5
    ORDER_LINE = 6
    NEW_ORDER = 7
    HISTORY = 8
    # CH-benchmark additions
    NATION = 9
    REGION = 10
    SUPPLIER = 11


# CH-benchmark adds these fixed-size tables
CH_EXTRA_TABLE_SIZES = {
    CHTable.NATION: 25,      # 25 nations (fixed)
    CHTable.REGION: 5,       # 5 regions (fixed)
    CHTable.SUPPLIER: 1000,  # 10K suppliers, ~1K blocks
}

# Query weights from real CH-benchmark config (BenchBase)
# <weights bench="chbenchmark">3, 2, 3, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5</weights>
CH_QUERY_WEIGHTS = {
    1: 3, 2: 2, 3: 3, 4: 2,  # First 4 queries have different weights
    5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5,
    11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5,
    17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5,
}


@dataclass
class CHBenchmarkConfig:
    """Configuration for CH-benchmark workload."""
    # TPC-C parameters
    num_warehouses: int = 10
    num_terminals: int = 5
    tpcc_transactions_per_terminal: int = 200
    
    # CH analytical parameters
    num_olap_streams: int = 2        # Concurrent OLAP query streams
    olap_queries_per_stream: int = 22  # Run all 22 queries per stream
    
    # Mix control - models interleaved OLTP/OLAP execution
    # olap_probability: fraction of "time" spent on OLAP (e.g., 0.25 = 4/16 terminals)
    # olap_batch_size: how many OLAP accesses per injection (models OLAP's larger I/O)
    # 
    # For realistic 12:4 terminal ratio (3:1 OLTP:OLAP):
    #   - olap_probability = 4/16 = 0.25
    #   - olap_batch_size = 40 (OLAP does ~10x more I/O per unit time)
    #   - Result: 4 OLTP accesses : 40 OLAP accesses = 1:10 ratio
    olap_probability: float = 0.05   # 5% chance of OLAP after each OLTP access
    olap_batch_size: int = 1000      # OLAP accesses per injection batch
    
    # Buffer pool
    buffer_pool_blocks: int = 32768  # 256 MB default
    
    # Timing
    time_per_block: float = 0.0001


class CHBenchmarkWorkload(BaseWorkload):
    """
    CH-benchmark: TPC-C + 22 TPC-H-like OLAP queries on same schema.
    
    The 22 OLAP queries are adapted from TPC-H to work on TPC-C schema:
    - LINEITEM -> ORDER_LINE
    - ORDERS -> ORDERS  
    - CUSTOMER -> CUSTOMER
    - PART -> ITEM
    - PARTSUPP -> STOCK
    - SUPPLIER -> SUPPLIER (added table)
    - NATION -> NATION (added table)
    - REGION -> REGION (added table)
    """
    
    def __init__(self, config: CHBenchmarkConfig = None):
        super().__init__(name="chbenchmark")
        self.config = config or CHBenchmarkConfig()
        
        # Initialize TPC-C workload component
        tpcc_config = TPCCConfig(
            num_warehouses=self.config.num_warehouses,
            num_terminals=self.config.num_terminals,
            transactions_per_terminal=self.config.tpcc_transactions_per_terminal,
            buffer_pool_blocks=self.config.buffer_pool_blocks,
            time_per_block=self.config.time_per_block,
        )
        self.tpcc = TPCCWorkload(tpcc_config)
        
        # Extend table sizes with CH additions
        self.table_sizes = dict(self.tpcc.table_sizes)
        self.table_sizes.update(CH_EXTRA_TABLE_SIZES)
        
        self._scan_id = 10000  # Start high to avoid collision with TPC-C
        self._current_time = 0.0
        
        # Track which queries have been run for diversity
        self._query_count = {i: 0 for i in range(1, 23)}
    
    def _next_scan_id(self) -> int:
        sid = self._scan_id
        self._scan_id += 1
        return sid
    
    # =========================================================================
    # Helper methods for OLAP query access patterns
    # =========================================================================
    
    def _seq_scan(self, table: int, start_block: int = 0, 
                  num_blocks: Optional[int] = None,
                  sample_rate: float = 1.0) -> Generator[WorkloadAccess, None, None]:
        """Sequential scan of a table (or portion)."""
        table_size = self.table_sizes.get(table, 1000)
        if num_blocks is None:
            num_blocks = table_size - start_block
        num_blocks = min(num_blocks, table_size - start_block)
        
        scan_id = self._next_scan_id()
        
        for i in range(num_blocks):
            if sample_rate < 1.0 and random.random() > sample_rate:
                continue  # Skip for sampling
                
            block = start_block + i
            ctx = ScanContext(
                scan_id=scan_id,
                relation_id=table,
                access_type=AccessType.SEQUENTIAL,
                current_position=block,
                total_blocks=num_blocks,
                blocks_remaining=num_blocks - i - 1,
                est_blocks_per_sec=10000.0,
            )
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.SEQUENTIAL,
                scan_context=ctx,
                is_write=False,
            )
            self._current_time += self.config.time_per_block
    
    def _index_scan(self, table: int, keys: List[int]) -> Generator[WorkloadAccess, None, None]:
        """Index lookup on specific keys."""
        table_size = self.table_sizes.get(table, 1000)
        scan_id = self._next_scan_id()
        
        for i, key in enumerate(keys):
            block = key % table_size
            ctx = ScanContext(
                scan_id=scan_id,
                relation_id=table,
                access_type=AccessType.INDEX,
                current_position=i,
                total_blocks=len(keys),
                blocks_remaining=len(keys) - i - 1,
                est_blocks_per_sec=5000.0,
            )
            yield WorkloadAccess(
                tag=(table, block),
                access_type=AccessType.INDEX,
                scan_context=ctx,
                is_write=False,
            )
            self._current_time += self.config.time_per_block
    
    def _hash_join_probe(self, build_table: int, probe_table: int,
                         probe_blocks: int) -> Generator[WorkloadAccess, None, None]:
        """
        Simulate hash join: build side scan + probe side scan.
        Build side is typically smaller (fits in memory).
        """
        # Build phase - scan entire build table
        yield from self._seq_scan(build_table)
        
        # Probe phase - scan probe table
        yield from self._seq_scan(probe_table, num_blocks=probe_blocks)
    
    # =========================================================================
    # All 22 CH-benchmark OLAP Queries
    # =========================================================================
    
    def _ch_query_1(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q1: Pricing Summary Report
        SELECT ol_number, sum(ol_quantity), sum(ol_amount), ...
        FROM order_line
        WHERE ol_delivery_d > '2007-01-02'
        GROUP BY ol_number ORDER BY ol_number
        
        Access: Full scan of ORDER_LINE
        """
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_2(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q2: Minimum Cost Supplier
        SELECT su_suppkey, su_name, n_name, i_id, i_name, ...
        FROM item, supplier, stock, nation, region
        WHERE i_id = s_i_id AND ...
        
        Access: Joins across ITEM, STOCK, SUPPLIER, NATION, REGION
        """
        # Scan small dimension tables first
        yield from self._seq_scan(CHTable.REGION)
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        # Then larger tables
        yield from self._seq_scan(CHTable.ITEM)
        # Stock scan (large, may sample)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.1)
    
    def _ch_query_3(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q3: Shipping Priority
        SELECT ol_o_id, ol_w_id, ol_d_id, sum(ol_amount) as revenue, o_entry_d
        FROM customer, new_order, orders, order_line
        WHERE c_state LIKE 'A%' AND ...
        
        Access: Joins CUSTOMER, NEW_ORDER, ORDERS, ORDER_LINE
        """
        # Customer scan with filter
        yield from self._seq_scan(CHTable.CUSTOMER, sample_rate=0.1)
        # New_order scan
        yield from self._seq_scan(CHTable.NEW_ORDER)
        # Orders scan
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.3)
        # Order_line scan
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
    
    def _ch_query_4(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q4: Order Priority Checking
        SELECT o_ol_cnt, count(*) as order_count
        FROM orders
        WHERE o_entry_d >= '2007-01-02' AND EXISTS (SELECT * FROM order_line ...)
        
        Access: ORDERS scan with correlated ORDER_LINE lookup
        """
        yield from self._seq_scan(CHTable.ORDERS)
        # Correlated subquery - sample of ORDER_LINE
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.2)
    
    def _ch_query_5(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q5: Local Supplier Volume
        SELECT n_name, sum(ol_amount) as revenue
        FROM customer, orders, order_line, stock, supplier, nation, region
        WHERE c_id = o_c_id AND ... AND n_name = 'Germany'
        
        Access: Large multi-way join
        """
        yield from self._seq_scan(CHTable.REGION)
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.CUSTOMER, sample_rate=0.2)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.2)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.2)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.1)
    
    def _ch_query_6(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q6: Forecasting Revenue Change
        SELECT sum(ol_amount) as revenue
        FROM order_line
        WHERE ol_delivery_d >= '1999-01-01' 
          AND ol_delivery_d < '2020-01-01'
          AND ol_quantity BETWEEN 1 AND 100000
        
        Access: Full scan of ORDER_LINE with filter
        """
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_7(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q7: Volume Shipping
        SELECT su_nationkey as supp_nation, ... , sum(ol_amount) as revenue
        FROM supplier, stock, order_line, orders, customer, nation n1, nation n2
        WHERE ol_supply_w_id = s_w_id AND ...
        
        Access: Complex multi-table join
        """
        yield from self._seq_scan(CHTable.NATION)  # Scanned twice for n1, n2
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.CUSTOMER, sample_rate=0.3)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.3)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.1)
    
    def _ch_query_8(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q8: National Market Share
        SELECT o_entry_d, sum(case when n2.n_name = 'Germany' then ol_amount else 0 end) / sum(ol_amount)
        FROM region, nation n1, nation n2, customer, orders, order_line, stock, supplier, item
        WHERE i_data LIKE '%b' AND ...
        
        Access: Largest join in CH-benchmark
        """
        yield from self._seq_scan(CHTable.REGION)
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.ITEM, sample_rate=0.5)
        yield from self._seq_scan(CHTable.CUSTOMER, sample_rate=0.2)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.2)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.2)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.1)
    
    def _ch_query_9(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q9: Product Type Profit Measure
        SELECT n_name, o_entry_d, sum(ol_amount) as sum_profit
        FROM item, stock, supplier, order_line, orders, nation
        WHERE ol_i_id = s_i_id AND ...
        
        Access: Multi-table join focused on profit calculation
        """
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.ITEM, sample_rate=0.3)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.3)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.1)
    
    def _ch_query_10(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q10: Returned Item Reporting
        SELECT c_id, c_last, sum(ol_amount) as revenue, c_city, c_phone, n_name
        FROM customer, orders, order_line, nation
        WHERE c_id = o_c_id AND o_entry_d >= ... AND ol_delivery_d IS NOT NULL
        
        Access: CUSTOMER, ORDERS, ORDER_LINE, NATION join
        """
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.CUSTOMER)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.5)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.5)
    
    def _ch_query_11(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q11: Important Stock Identification
        SELECT s_i_id, sum(s_order_cnt) as ordercount
        FROM stock, supplier, nation
        WHERE ... GROUP BY s_i_id 
        HAVING sum(s_order_cnt) > (SELECT sum(s_order_cnt) * .005 
                                   FROM stock, supplier, nation WHERE ...)
        
        Access: STOCK, SUPPLIER, NATION scanned TWICE (main query + HAVING subquery)
        """
        # Main query scan
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.STOCK)
        # HAVING subquery scans same tables again
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.STOCK)
    
    def _ch_query_12(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q12: Shipping Modes and Order Priority
        SELECT o_ol_cnt, sum(case when o_carrier_id = 1 or o_carrier_id = 2 then 1 else 0 end), ...
        FROM orders, order_line
        WHERE ol_w_id = o_w_id AND ol_d_id = o_d_id AND o_id = ol_o_id
          AND ol_delivery_d >= o_entry_d
        
        Access: ORDERS + ORDER_LINE join
        """
        yield from self._seq_scan(CHTable.ORDERS)
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_13(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q13: Customer Distribution
        SELECT c_count, count(*) as custdist
        FROM (SELECT c_id, count(o_id) as c_count FROM customer LEFT OUTER JOIN orders ON ...)
        GROUP BY c_count ORDER BY custdist DESC
        
        Access: CUSTOMER outer join with ORDERS
        """
        yield from self._seq_scan(CHTable.CUSTOMER)
        yield from self._seq_scan(CHTable.ORDERS)
    
    def _ch_query_14(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q14: Promotion Effect
        SELECT 100.00 * sum(case when i_data like 'PR%' then ol_amount else 0 end) / ...
        FROM order_line, item
        WHERE ol_i_id = i_id AND ol_delivery_d >= '2007-01-02'
        
        Access: ORDER_LINE + ITEM join
        """
        yield from self._seq_scan(CHTable.ITEM)
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_15(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q15: Top Supplier
        WITH revenue AS (SELECT ... FROM order_line WHERE ...)
        SELECT su_suppkey, su_name, su_address, su_phone, total_revenue
        FROM supplier, revenue
        WHERE su_suppkey = supplier_no AND total_revenue = (SELECT max(total_revenue) FROM revenue)
        
        Access: ORDER_LINE aggregation + SUPPLIER lookup
        """
        yield from self._seq_scan(CHTable.ORDER_LINE)
        yield from self._seq_scan(CHTable.SUPPLIER)
    
    def _ch_query_16(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q16: Parts/Supplier Relationship
        SELECT i_name, substr(i_data, 1, 3) as brand, i_price, count(DISTINCT s_su_suppkey)
        FROM stock, item
        WHERE i_id = s_i_id AND i_data NOT LIKE 'zz%' 
          AND s_su_suppkey NOT IN (SELECT su_suppkey FROM supplier WHERE su_comment LIKE '%bad%')
        
        Access: STOCK, ITEM, SUPPLIER scan
        """
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.ITEM)
        yield from self._seq_scan(CHTable.STOCK)
    
    def _ch_query_17(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q17: Small-Quantity-Order Revenue
        SELECT sum(ol_amount) / 2.0 as avg_yearly
        FROM order_line, item
        WHERE ol_i_id = i_id AND i_data LIKE '%b'
          AND ol_quantity < (SELECT 0.2 * avg(ol_quantity) FROM order_line WHERE ol_i_id = i_id)
        
        Access: ORDER_LINE (twice for correlated subquery) + ITEM
        """
        yield from self._seq_scan(CHTable.ITEM, sample_rate=0.5)
        yield from self._seq_scan(CHTable.ORDER_LINE)
        # Correlated subquery accesses ORDER_LINE again
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
    
    def _ch_query_18(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q18: Large Volume Customer
        SELECT c_last, c_id, o_id, o_entry_d, o_ol_cnt, sum(ol_amount) as amount
        FROM customer, orders, order_line
        WHERE c_id = o_c_id AND o_id = ol_o_id
        GROUP BY o_id, o_entry_d, o_ol_cnt, c_id, c_last
        HAVING sum(ol_amount) > 200
        
        Access: CUSTOMER, ORDERS, ORDER_LINE join with aggregation
        """
        yield from self._seq_scan(CHTable.CUSTOMER)
        yield from self._seq_scan(CHTable.ORDERS)
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_19(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q19: Discounted Revenue
        SELECT sum(ol_amount) as revenue
        FROM order_line, item
        WHERE (ol_i_id = i_id AND i_data LIKE '%a' AND ol_quantity >= 1 AND ol_quantity <= 10 ...) OR ...
        
        Access: ORDER_LINE + ITEM join with complex predicates
        """
        yield from self._seq_scan(CHTable.ITEM)
        yield from self._seq_scan(CHTable.ORDER_LINE)
    
    def _ch_query_20(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q20: Potential Part Promotion
        SELECT su_name, su_address
        FROM supplier, nation
        WHERE su_suppkey IN (SELECT s_su_suppkey FROM stock, order_line
                             WHERE s_i_id = ol_i_id AND ol_delivery_d > ...)
          AND su_nationkey = n_nationkey AND n_name = 'Germany'
        
        Access: SUPPLIER, NATION, STOCK, ORDER_LINE, (ITEM implied)
        """
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.5)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.2)
    
    def _ch_query_21(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q21: Suppliers Who Kept Orders Waiting
        SELECT su_name, count(*) as numwait
        FROM supplier, order_line l1, orders, stock, nation
        WHERE ol_o_id = o_id AND ol_w_id = o_w_id AND ...
          AND l1.ol_delivery_d > o_entry_d
          AND NOT EXISTS (SELECT * FROM order_line l2 WHERE ...)
          AND EXISTS (SELECT * FROM order_line l3 WHERE ...)
        
        Access: Complex with multiple ORDER_LINE scans (l1, l2, l3)
        """
        yield from self._seq_scan(CHTable.NATION)
        yield from self._seq_scan(CHTable.SUPPLIER)
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.5)
        yield from self._seq_scan(CHTable.STOCK, sample_rate=0.2)
        # l1, l2, l3 scans of ORDER_LINE
        yield from self._seq_scan(CHTable.ORDER_LINE)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
        yield from self._seq_scan(CHTable.ORDER_LINE, sample_rate=0.3)
    
    def _ch_query_22(self) -> Generator[WorkloadAccess, None, None]:
        """
        Q22: Global Sales Opportunity
        SELECT substr(c_state, 1, 1) as country, count(*) as numcust, sum(c_balance) as totacctbal
        FROM customer
        WHERE substr(c_phone, 1, 1) IN ('1','2','3','4','5','6','7')
          AND c_balance > (SELECT avg(c_balance) FROM customer WHERE c_balance > 0.00 AND ...)
          AND NOT EXISTS (SELECT * FROM orders WHERE o_c_id = c_id AND o_w_id = c_w_id AND o_d_id = c_d_id)
        
        Access: CUSTOMER (twice for subquery) + ORDERS anti-join
        """
        yield from self._seq_scan(CHTable.CUSTOMER)
        yield from self._seq_scan(CHTable.CUSTOMER, sample_rate=0.5)  # Subquery
        yield from self._seq_scan(CHTable.ORDERS, sample_rate=0.3)    # Anti-join
    
    # =========================================================================
    # Query dispatch
    # =========================================================================
    
    def _run_olap_query(self, query_num: int = None) -> Generator[WorkloadAccess, None, None]:
        """Run a specific OLAP query or random one."""
        if query_num is None:
            query_num = random.randint(1, 22)
        
        self._query_count[query_num] += 1
        
        query_methods = {
            1: self._ch_query_1,
            2: self._ch_query_2,
            3: self._ch_query_3,
            4: self._ch_query_4,
            5: self._ch_query_5,
            6: self._ch_query_6,
            7: self._ch_query_7,
            8: self._ch_query_8,
            9: self._ch_query_9,
            10: self._ch_query_10,
            11: self._ch_query_11,
            12: self._ch_query_12,
            13: self._ch_query_13,
            14: self._ch_query_14,
            15: self._ch_query_15,
            16: self._ch_query_16,
            17: self._ch_query_17,
            18: self._ch_query_18,
            19: self._ch_query_19,
            20: self._ch_query_20,
            21: self._ch_query_21,
            22: self._ch_query_22,
        }
        
        yield from query_methods[query_num]()
    
    def _select_weighted_query(self) -> int:
        """Select a query number based on CH-benchmark weights."""
        # Build cumulative weights for weighted random selection
        total_weight = sum(CH_QUERY_WEIGHTS.values())
        r = random.random() * total_weight
        cumulative = 0
        for q, weight in CH_QUERY_WEIGHTS.items():
            cumulative += weight
            if r <= cumulative:
                return q
        return 22  # Fallback
    
    def _run_all_olap_queries(self) -> Generator[WorkloadAccess, None, None]:
        """Run OLAP queries with weighted selection (matching real CH-benchmark)."""
        # Run a fixed number of queries based on weights
        num_queries = self.config.olap_queries_per_stream
        for _ in range(num_queries):
            query_num = self._select_weighted_query()
            yield from self._run_olap_query(query_num)
    
    # =========================================================================
    # Main workload generation
    # =========================================================================
    
    def generate(self) -> Generator[WorkloadAccess, None, None]:
        """
        Generate interleaved OLTP + OLAP workload.
        
        TPC-C transactions run continuously, with OLAP queries
        injected in batches to model concurrent execution.
        """
        self._scan_id = 10000
        self._current_time = 0.0
        self._query_count = {i: 0 for i in range(1, 23)}
        
        # Generate TPC-C stream
        tpcc_stream = self.tpcc.generate()
        
        # OLAP query streams
        olap_streams = [iter(self._run_all_olap_queries()) 
                       for _ in range(self.config.num_olap_streams)]
        active_olap = list(range(len(olap_streams)))
        
        access_count = 0
        olap_injection_count = 0  # Track injections separately for proper round-robin
        # How many OLAP accesses to inject per batch (models concurrent execution)
        olap_batch_size = self.config.olap_batch_size
        olap_interval = int(1.0 / self.config.olap_probability) if self.config.olap_probability > 0 else 1000000
        
        for tpcc_access in tpcc_stream:
            # Yield TPC-C access
            yield tpcc_access
            access_count += 1
            
            # Periodically inject BATCH of OLAP accesses
            if access_count % olap_interval == 0 and active_olap:
                # Round-robin through OLAP streams using injection count (not access_count)
                stream_idx = active_olap[olap_injection_count % len(active_olap)]
                olap_injection_count += 1
                try:
                    # Yield a batch of OLAP accesses (models concurrent query execution)
                    for _ in range(olap_batch_size):
                        olap_access = next(olap_streams[stream_idx])
                        yield olap_access
                except StopIteration:
                    active_olap.remove(stream_idx)
    
    def generate_olap_only(self) -> Generator[WorkloadAccess, None, None]:
        """Generate only OLAP queries (for testing)."""
        self._scan_id = 10000
        self._current_time = 0.0
        self._query_count = {i: 0 for i in range(1, 23)}
        
        for stream in range(self.config.num_olap_streams):
            yield from self._run_all_olap_queries()
    
    def get_query_stats(self) -> Dict[int, int]:
        """Return count of each query type executed."""
        return dict(self._query_count)


def create_chbenchmark_workload(
    num_warehouses: int = 10,
    num_terminals: int = 5,
    tpcc_transactions_per_terminal: int = 200,
    num_olap_streams: int = 2,
    olap_probability: float = 0.05,
    olap_batch_size: int = 1000,
    buffer_pool_mb: int = 256,
) -> CHBenchmarkWorkload:
    """Factory function for CH-benchmark workload."""
    config = CHBenchmarkConfig(
        num_warehouses=num_warehouses,
        num_terminals=num_terminals,
        tpcc_transactions_per_terminal=tpcc_transactions_per_terminal,
        num_olap_streams=num_olap_streams,
        olap_probability=olap_probability,
        olap_batch_size=olap_batch_size,
        buffer_pool_blocks=buffer_pool_mb * 128,
    )
    return CHBenchmarkWorkload(config)

