"""
Buffer Tag - Unique identifier for a database page.

Simplified for simulation: (relation_id, block_num)
"""


def normalize_tag(tag):
    """
    Normalize tag to (relation_id, block_num) tuple.
    
    Accepts:
    - Tuple (relation_id, block_num)
    - Integer (treated as block_num with relation=0)
    
    Returns: (relation_id, block_num)
    """
    if isinstance(tag, tuple):
        if len(tag) == 2:
            return tag
        elif len(tag) == 3:
            # (rel, fork, block) -> (rel, block)
            return (tag[0], tag[2])
        else:
            raise ValueError(f"Invalid tag tuple length: {len(tag)}")
    elif isinstance(tag, int):
        return (0, tag)
    else:
        raise ValueError(f"Invalid tag type: {type(tag)}")


def get_relation_id(tag) -> int:
    """Extract relation_id from tag."""
    return normalize_tag(tag)[0]


def get_block_num(tag) -> int:
    """Extract block_num from tag."""
    return normalize_tag(tag)[1]

