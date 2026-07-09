"""
Basic tests for minimal_postgres_simulator.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BufferManager, BufferDescriptor, BlockGroup, BlockGroupRegistry
from core.block_group import BLOCKS_PER_GROUP
from scan_tracker import ScanRegistry, ScanEntry, NextAccessEstimator, ScanType
from workload import SequentialMicrobench, SequentialMicrobenchConfig
from policies import clock_sweep_policy, pbm_sampling_policy


def test_block_group():
    """Test BlockGroup and BlockGroupRegistry."""
    print("Testing BlockGroup...")
    
    registry = BlockGroupRegistry()
    
    # Create block group for block 0
    bg0 = registry.get_or_create(0, 0)
    assert bg0.relation_id == 0
    assert bg0.group_num == 0
    assert bg0.start_block == 0
    assert bg0.end_block == BLOCKS_PER_GROUP
    
    # Same block group for block 127
    bg127 = registry.get_or_create(0, 127)
    assert bg127 is bg0
    
    # Different block group for block 128
    bg128 = registry.get_or_create(0, 128)
    assert bg128.group_num == 1
    assert bg128 is not bg0
    
    print("  ✓ BlockGroup tests passed")


def test_scan_registry():
    """Test ScanRegistry."""
    print("Testing ScanRegistry...")
    
    block_groups = BlockGroupRegistry()
    registry = ScanRegistry(block_groups)
    
    # Set initial time BEFORE registration
    registry.set_time(0.0)
    
    # Register a sequential scan
    scan_id = registry.register_sequential_scan(0, 0, 1000)
    assert scan_id == 1
    
    scan = registry.get_scan(scan_id)
    assert scan is not None
    assert scan.scan_type == ScanType.SEQUENTIAL
    assert scan.total_blocks == 1000
    
    # Check block groups are linked
    bg0 = block_groups.get(0, 0)
    assert scan_id in bg0.scan_ids
    
    # First update (sets position, but no speed yet because no time passed)
    registry.set_time(0.05)  # 50ms later
    registry.update_scan_position(scan_id, 100)  # Move 100 blocks
    
    scan = registry.get_scan(scan_id)
    # Speed should be computed now: 100 blocks / 0.05 sec = 2000 blocks/sec
    assert scan.est_blocks_per_sec > 0, f"Speed was {scan.est_blocks_per_sec}"
    
    # Second update
    registry.set_time(0.1)  # 50ms more
    registry.update_scan_position(scan_id, 500)  # Move 400 more blocks
    
    scan = registry.get_scan(scan_id)
    assert scan.current_position == 500
    
    # Unregister
    registry.unregister_scan(scan_id)
    assert registry.get_scan(scan_id) is None
    
    print("  ✓ ScanRegistry tests passed")


def test_next_access_estimator():
    """Test NextAccessEstimator."""
    print("Testing NextAccessEstimator...")
    
    block_groups = BlockGroupRegistry()
    scans = ScanRegistry(block_groups)
    estimator = NextAccessEstimator(scans, block_groups)
    
    # No scans - should return infinity
    est, requested = estimator.estimate_next_access(0, 500, current_time=0)
    assert not requested
    assert est == float('inf')
    
    # Register a scan
    scans.set_time(0)
    scan_id = scans.register_sequential_scan(0, 0, 1000)
    
    # Now block 500 should be requested
    est, requested = estimator.estimate_next_access(0, 500, current_time=0)
    assert requested
    assert est < float('inf')
    
    print("  ✓ NextAccessEstimator tests passed")


def test_workload_generation():
    """Test workload generation."""
    print("Testing Workload Generation...")
    
    config = SequentialMicrobenchConfig(
        num_streams=1,
        queries_per_stream=1,
        table_blocks=10000,  # Larger for reliable bitmap
        selectivity=0.3,  # 30% like thesis
    )
    workload = SequentialMicrobench(config)
    
    # Generate and count accesses
    count = 0
    for access in workload.generate():
        count += 1
        assert access.scan_context is not None, f"Access {count} has no scan_context"
        if count > 200:  # Just check first few
            break
    
    assert count > 0, "Workload generated 0 accesses"
    print(f"  ✓ Generated {count}+ accesses")


def test_buffer_manager():
    """Test BufferManager with scan tracking."""
    print("Testing BufferManager...")
    
    # Create buffer manager
    manager = BufferManager(100)  # 100 buffers
    
    # Set up scan tracking
    scans = ScanRegistry(manager.block_groups)
    estimator = NextAccessEstimator(scans, manager.block_groups)
    manager.set_scan_tracker(scans, estimator)
    
    # Access some buffers
    for i in range(50):
        buf_id = manager.read_buffer((0, i))
        manager.unpin_buffer(buf_id)
    
    # Check stats
    assert manager.stats.buffer_misses == 50
    
    # Access same buffers again (should hit)
    for i in range(50):
        buf_id = manager.read_buffer((0, i))
        manager.unpin_buffer(buf_id)
    
    assert manager.stats.buffer_hits == 50
    assert manager.stats.hit_rate == 0.5
    
    print("  ✓ BufferManager tests passed")


def test_policies():
    """Test replacement policies."""
    print("Testing Policies...")
    
    # Create small buffer pool
    manager = BufferManager(10, clock_sweep_policy)
    scans = ScanRegistry(manager.block_groups)
    estimator = NextAccessEstimator(scans, manager.block_groups)
    manager.set_scan_tracker(scans, estimator)
    
    # Fill buffer pool
    for i in range(10):
        buf_id = manager.read_buffer((0, i))
        manager.unpin_buffer(buf_id)
    
    # Access one more (forces eviction)
    buf_id = manager.read_buffer((0, 100))
    manager.unpin_buffer(buf_id)
    
    assert manager.stats.buffer_evictions == 1
    
    print("  ✓ Policy tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running minimal_postgres_simulator tests")
    print("=" * 50)
    
    test_block_group()
    test_scan_registry()
    test_next_access_estimator()
    test_workload_generation()
    test_buffer_manager()
    test_policies()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == '__main__':
    run_all_tests()

