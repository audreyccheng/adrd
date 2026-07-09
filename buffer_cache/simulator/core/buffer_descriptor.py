"""
Buffer Descriptor - Represents a single buffer in the buffer pool.

Minimal representation of PostgreSQL's BufferDesc with PBM extensions.
"""
from typing import Optional, Any


class BufferDescriptor:
    """
    Buffer descriptor with PBM-style extensions.
    
    Core fields match PostgreSQL's BufferDesc.
    Added: block_group pointer for PBM scan tracking.
    """
    
    __slots__ = [
        'buf_id', 'tag', 'refcount', 'usage_count', 'is_dirty',
        'block_group', 'last_access_time', 'access_count',
        'dirty_time'  # When page became dirty (for WAL flush cost modeling)
    ]
    
    def __init__(self, buf_id: int):
        # Buffer identification
        self.buf_id = buf_id
        
        # Page identification
        self.tag: Optional[tuple] = None  # (relation_id, block_num)
        
        # Buffer state (matches PostgreSQL)
        self.refcount = 0
        self.usage_count = 0
        self.is_dirty = False
        
        # PBM extensions
        self.block_group: Optional[Any] = None  # Pointer to BlockGroup
        self.last_access_time: float = 0.0  # For frequency tracking
        self.access_count: int = 0  # For frequency tracking
        
        # WAL flush cost modeling
        self.dirty_time: float = 0.0  # When page became dirty (0 = never/clean)
    
    def is_valid(self) -> bool:
        """Check if buffer contains a valid page."""
        return self.tag is not None
    
    def is_pinned(self) -> bool:
        """Check if buffer is currently pinned."""
        return self.refcount > 0
    
    def can_evict(self) -> bool:
        """Check if buffer can be evicted."""
        return self.refcount == 0
    
    def pin(self):
        """Increment reference count."""
        self.refcount += 1
    
    def unpin(self):
        """Decrement reference count."""
        assert self.refcount > 0, "Cannot unpin buffer with refcount 0"
        self.refcount -= 1
    
    def increment_usage(self, max_usage: int = 5):
        """Increment usage count (capped at max_usage)."""
        self.usage_count = min(self.usage_count + 1, max_usage)
    
    def decrement_usage(self):
        """Decrement usage count for clock-sweep."""
        if self.usage_count > 0:
            self.usage_count -= 1
    
    def reset(self):
        """Reset buffer to empty state."""
        self.tag = None
        self.refcount = 0
        self.usage_count = 0
        self.is_dirty = False
        self.block_group = None
        self.last_access_time = 0.0
        self.access_count = 0
        self.dirty_time = 0.0
    
    def __repr__(self):
        return (f"BufferDesc(id={self.buf_id}, tag={self.tag}, "
                f"ref={self.refcount}, usage={self.usage_count})")

