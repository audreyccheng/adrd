"""
Core buffer management simulator components.

Minimal but representative simulation of PostgreSQL's buffer cache,
with PBM-style scan tracking support.
"""
from .buffer_descriptor import BufferDescriptor
from .buffer_manager import BufferManager, BufferStats
from .buffer_tag import normalize_tag, get_relation_id, get_block_num
from .block_group import BlockGroup, BlockGroupRegistry, BLOCKS_PER_GROUP

__all__ = [
    'BufferDescriptor',
    'BufferManager',
    'BufferStats',
    'normalize_tag',
    'get_relation_id',
    'get_block_num',
    'BlockGroup',
    'BlockGroupRegistry',
    'BLOCKS_PER_GROUP',
]

